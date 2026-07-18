"""
Modal app for fine-tuning Qwen3-VL-8B-Thinking using LoRA.
"""
import json
import logging
from pathlib import Path

import modal

MODAL_VOLUME_NAME = "glance-data"
MODAL_GPU = "H100"
MODAL_TIMEOUT = 3600 * 5  # 5 hours
MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"

logger = logging.getLogger("finetune_modal")
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
    )
)

app = modal.App("glance-finetuner")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

REMOTE_DATA_DIR = "/data"

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT,
    volumes={REMOTE_DATA_DIR: volume},
)
def finetune():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm

    logger.info("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    
    # Load model in bfloat16 to save memory, prepare for LoRA
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    
    # Define LoRA config targeting attention and MLP layers
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    data_path = Path(REMOTE_DATA_DIR) / "finetune_data.jsonl"
    logger.info(f"Loading dataset from {data_path}")
    with open(data_path, 'r') as f:
        raw_data = [json.loads(line) for line in f]
        
    logger.info(f"Loaded {len(raw_data)} training examples.")
    
    # Custom Dataset
    class VLMDataset(Dataset):
        def __init__(self, data, processor):
            self.data = data
            self.processor = processor
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            item = self.data[idx]
            messages = item["messages"]
            
            # The JSONL has "image" as a string path (e.g. /data/images/file.jpg)
            # We need to load it as a PIL image for the processor
            # Find the image path in the messages and replace with PIL Image
            processed_messages = []
            for msg in messages:
                content = []
                for chunk in msg["content"]:
                    if chunk["type"] == "image":
                        img_path = chunk["image"]
                        try:
                            img = Image.open(img_path).convert("RGB")
                            content.append({"type": "image", "image": img})
                        except Exception as e:
                            logger.error(f"Failed to load image {img_path}: {e}")
                            # fallback to empty image (1x1 black) if missing
                            content.append({"type": "image", "image": Image.new("RGB", (1,1))})
                    else:
                        content.append(chunk)
                processed_messages.append({"role": msg["role"], "content": content})
            
            # Apply chat template
            inputs = self.processor.apply_chat_template(
                processed_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt"
            )
            
            # Squeeze batch dimension added by processor only for 1D sequence tensors
            inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) and k in ["input_ids", "attention_mask", "labels", "mm_token_type_ids"] and v.ndim > 1 and v.shape[0] == 1 else v for k, v in inputs.items()}
            
            # For causal LM, labels are typically the input_ids
            # We'll set labels = input_ids, and mask out the user prompt part below
            inputs["labels"] = inputs["input_ids"].clone()
            
            return inputs

    dataset = VLMDataset(raw_data, processor)
    
    # Custom collator to handle padding
    def collate_fn(batch):
        # We need to pad input_ids and labels to the max length in the batch
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        
        # Determine padding side and token
        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
        
        # Pad input_ids and attention_mask
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        
        # Pad labels with -100 so padded tokens are ignored in loss
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        batch_out = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
        
        if "mm_token_type_ids" in batch[0]:
            mm_token_type_ids = [item["mm_token_type_ids"] for item in batch]
            batch_out["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(mm_token_type_ids, batch_first=True, padding_value=0)
        
        # Handle pixel_values and image_grid_thw (if present)
        if "pixel_values" in batch[0]:
            # For Qwen3-VL, pixel_values might be 2D/3D and variable size per image,
            # but apply_chat_template usually returns them stacked or concatenated
            # If they are concatenated, we can just concat them across the batch
            batch_out["pixel_values"] = torch.cat([item["pixel_values"] for item in batch], dim=0)
            if "image_grid_thw" in batch[0]:
                batch_out["image_grid_thw"] = torch.cat([item["image_grid_thw"] for item in batch], dim=0)
                
        return batch_out

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training Loop
    epochs = 3
    model.train()
    
    logger.info("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            optimizer.zero_grad()
            
            # Move batch to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
    # Save adapter
    output_dir = Path(REMOTE_DATA_DIR) / "lora_adapters"
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    
    volume.commit()
    logger.info(f"Fine-tuning complete. LoRA adapters saved to {output_dir}")
    
@app.local_entrypoint()
def main():
    logger.info("Uploading dataset to Modal volume...")
    
    # Upload the JSONL dataset
    data_dir = Path(__file__).resolve().parent.parent / "data"
    finetune_file = data_dir / "finetune_data.jsonl"
    
    if not finetune_file.exists():
        logger.error(f"Finetune dataset not found at {finetune_file}")
        return
        
    images_dir = data_dir / "images"
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(finetune_file), "/finetune_data.jsonl")
        if images_dir.exists():
            for img_path in images_dir.glob("*.jpg"):
                batch.put_file(str(img_path), f"images/{img_path.name}")
        
    logger.info("Starting remote fine-tuning job...")
    finetune.remote()
    
    logger.info("Fine-tuning complete. LoRA adapters saved to /data/lora_adapters on the Modal Volume.")
