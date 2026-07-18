import json
import logging
from pathlib import Path
import modal

MODAL_VOLUME_NAME = "glance-data"
MODAL_GPU = "A100"
MODAL_TIMEOUT = 86400  # 24 hours
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

logger = logging.getLogger("train_align_eval")
logging.basicConfig(level=logging.INFO)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "peft",
        "Pillow",
        "tqdm",
        "scikit-learn",
        "bert-score"
    )
)

app = modal.App("glance-unified-pipeline")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)
REMOTE_DATA_DIR = "/data"

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT,
    volumes={REMOTE_DATA_DIR: volume},
)
def run_pipeline():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import numpy as np
    import bert_score

    logger.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    data_dir = Path(REMOTE_DATA_DIR)
    
    # --- UTILS ---
    def load_jsonl(path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f]
            
    def compute_f1(pred_text, gt_cats, gt_attrs):
        pred_lower = pred_text.lower()
        gt_all = [g.lower() for g in gt_cats + gt_attrs]
        if not gt_all: return 0.0
        
        hits = sum(1 for g in gt_all if g in pred_lower)
        precision = hits / len(pred_lower.split()) if pred_lower.split() else 0
        recall = hits / len(gt_all)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    # --- EVALUATION FUNCTION ---
    def evaluate(model_eval, tag, limit=None):
        logger.info(f"--- Evaluating: {tag} ---")
        test_data = load_jsonl(data_dir / "test_data.jsonl")
        if limit: test_data = test_data[:limit]
        
        model_eval.eval()
        total_f1 = 0
        all_preds = []
        all_refs = []
        
        for item in tqdm(test_data, desc=f"Eval {tag}"):
            try:
                img_path = item["messages"][0]["content"][0]["image"]
                raw_image = Image.open(img_path).convert("RGB")
                prompt_text = item["messages"][0]["content"][1]["text"]
                
                messages = [{"role": "user", "content": [{"type": "image", "image": raw_image}, {"type": "text", "text": prompt_text}]}]
                
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = {k: v.to(model_eval.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    gen_ids = model_eval.generate(**inputs, max_new_tokens=100)
                    
                gen_ids_trimmed = gen_ids[0][len(inputs["input_ids"][0]):]
                pred_text = processor.decode(gen_ids_trimmed, skip_special_tokens=True)
                
                gt_cats = item.get("ground_truth_categories", [])
                gt_attrs = item.get("ground_truth_attributes", [])
                
                f1 = compute_f1(pred_text, gt_cats, gt_attrs)
                total_f1 += f1
                all_preds.append(pred_text)
                all_refs.append(" ".join(gt_cats + gt_attrs))
            except Exception as e:
                continue
                
        avg_f1 = total_f1 / len(test_data)
        logger.info(f"[{tag}] Average Attribute F1: {avg_f1:.4f}")
        
        logger.info(f"[{tag}] Computing BERTScore...")
        P, R, F1 = bert_score.score(all_preds, all_refs, lang="en", rescale_with_baseline=True)
        avg_bertscore = F1.mean().item()
        logger.info(f"[{tag}] Average BERTScore (F1): {avg_bertscore:.4f}")
        
        return {"f1": avg_f1, "bert_score": avg_bertscore}

    # 1. EVALUATE BASELINE (Skipped)
    baseline_f1 = "Skipped"

    # --- SETUP LORA FOR SFT & GSPO ---
    model.gradient_checkpointing_enable()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(r=16, lora_alpha=32, target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    
    # Dataset Utils
    class VLMDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            msg = item["messages"]
            proc_msg = []
            for m in msg:
                content = []
                for chunk in m["content"]:
                    if chunk["type"] == "image":
                        try:
                            img = Image.open(chunk["image"]).convert("RGB")
                            content.append({"type": "image", "image": img})
                        except:
                            content.append({"type": "image", "image": Image.new("RGB", (1,1))})
                    else:
                        content.append(chunk)
                proc_msg.append({"role": m["role"], "content": content})
                
            inputs = processor.apply_chat_template(proc_msg, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt")
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and k in ["input_ids", "attention_mask", "labels", "mm_token_type_ids"] and v.ndim > 1 and v.shape[0] == 1 else v for k, v in inputs.items()}
            inputs["labels"] = inputs["input_ids"].clone()
            return inputs

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        
        batch_out = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100),
        }
        if "mm_token_type_ids" in batch[0]:
            mm_token_type_ids = [item["mm_token_type_ids"] for item in batch]
            batch_out["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
        if "pixel_values" in batch[0]:
            batch_out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
            if "image_grid_thw" in batch[0]:
                batch_out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
        return batch_out

    # 2. SFT PHASE
    logger.info("--- Starting SFT Phase ---")
    sft_data = load_jsonl(data_dir / "sft_data.jsonl")
    sft_loader = DataLoader(VLMDataset(sft_data), batch_size=2, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    model.train()
    for epoch in range(1): # 1 epoch for SFT to save time
        for batch in tqdm(sft_loader, desc=f"SFT Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            
    # 3. EVALUATE SFT
    sft_f1 = evaluate(model, "Baseline + SFT")

    # 4. GSPO ALIGNMENT PHASE (Skipped)
    gspo_data = []
    
    # Custom GSPO loop (Simplified Group Relative Policy Optimization)
    # G = 2 (generate 2 responses, calculate advantage, update)
    optimizer_rl = torch.optim.AdamW(model.parameters(), lr=5e-6)
    
    for idx, item in enumerate(tqdm(gspo_data, desc="GSPO Alignment")):
        try:
            img_path = item["messages"][0]["content"][0]["image"]
            raw_image = Image.open(img_path).convert("RGB")
            prompt_text = item["messages"][0]["content"][1]["text"]
            gt_cats = item.get("ground_truth_categories", [])
            gt_attrs = item.get("ground_truth_attributes", [])
            
            messages = [{"role": "user", "content": [{"type": "image", "image": raw_image}, {"type": "text", "text": prompt_text}]}]
            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate G=2 samples
            model.eval()
            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=1.0, num_return_sequences=2)
            
            rewards = []
            for i in range(2):
                gen_trimmed = gen_ids[i][len(inputs["input_ids"][0]):]
                pred_text = processor.decode(gen_trimmed, skip_special_tokens=True)
                rewards.append(compute_f1(pred_text, gt_cats, gt_attrs))
                
            r_mean = np.mean(rewards)
            r_std = np.std(rewards) + 1e-8
            advantages = [(r - r_mean) / r_std for r in rewards]
            
            # Policy update
            model.train()
            optimizer_rl.zero_grad()
            
            # Forward pass to get log probs for the generated sequences
            # Since VLM input includes images, we must pass the full sequence (prompt + gen)
            total_loss = 0
            for i in range(2):
                if advantages[i] == 0: continue
                full_ids = gen_ids[i:i+1] # [1, seq_len]
                # Rebuild batch for this specific sequence
                # We reuse pixel_values from inputs, and pass full_ids
                outputs = model(
                    input_ids=full_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )
                logits = outputs.logits # [1, seq_len, vocab]
                
                # Shift logits and labels for causal LM
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = full_ids[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Only compute loss over the generated tokens
                prompt_len = inputs["input_ids"].shape[1]
                gen_loss = token_losses[prompt_len-1:].mean() # average NLL of generated tokens
                
                # GSPO objective: - advantage * log_prob (where gen_loss is -log_prob)
                # So loss = advantage * gen_loss
                loss = advantages[i] * gen_loss
                total_loss += loss
                
            if total_loss != 0:
                total_loss.backward()
                optimizer_rl.step()
                
        except Exception as e:
            continue

    # 5. EVALUATE GSPO
    gspo_f1 = "Skipped"

    # SAVE RESULTS
    results = {
        "Baseline": baseline_f1,
        "SFT": sft_f1,
        "GSPO": gspo_f1
    }
    with open(data_dir / "final_evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    model.save_pretrained(str(data_dir / "lora_adapters_final"))
    processor.save_pretrained(str(data_dir / "lora_adapters_final"))
    volume.commit()
    
    logger.info("Pipeline Complete!")
    logger.info(f"Final Metrics: {results}")

@app.local_entrypoint()
def main():
    logger.info("Uploading datasets to Modal...")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    with volume.batch_upload(force=True) as batch:
        for f in ["sft_data.jsonl", "gspo_data.jsonl", "val_data.jsonl", "test_data.jsonl"]:
            p = data_dir / f
            if p.exists():
                batch.put_file(str(p), f"/{f}")
        
        img_dir = data_dir / "images"
        if img_dir.exists():
            for img_path in img_dir.glob("*.jpg"):
                batch.put_file(str(img_path), f"images/{img_path.name}")
                
    logger.info("Starting unified remote pipeline synchronously...")
    run_pipeline.remote()
    logger.info("Done! Metrics saved to Modal volume.")
