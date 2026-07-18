"""
Modal app for VLM captioning using Qwen3-VL-8B-Thinking on H100 GPU.
"""

import json
import logging
import sys
from pathlib import Path

import modal

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
CAPTIONS_FILE = DATA_DIR / "captions.json"
CAPTION_MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
MODAL_VOLUME_NAME = "glance-data"
MODAL_GPU = "H100"
MODAL_TIMEOUT = 3600 * 5

logger = logging.getLogger("caption_modal")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Modal image — includes all deps needed inside the remote container
# ---------------------------------------------------------------------------
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
        "qwen-vl-utils",
    )
)

app = modal.App("glance-captioner")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

REMOTE_DATA_DIR = "/data"


@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT,
    volumes={REMOTE_DATA_DIR: volume},
    scaledown_window=300,
)
def generate_captions():
    """Generate VLM captions for images using Qwen3-VL-8B-Thinking."""
    import torch
    from PIL import Image
    from tqdm import tqdm
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    logger.info("Loading Qwen3-VL model...")
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    adapters_path = Path(REMOTE_DATA_DIR) / "lora_adapters"
    if adapters_path.exists():
        logger.info(f"Found LoRA adapters at {adapters_path}. Loading...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapters_path))
        model.eval()
    else:
        logger.info("No LoRA adapters found, running base model.")

    annotations_path = Path(REMOTE_DATA_DIR) / "annotations.json"
    images_dir = Path(REMOTE_DATA_DIR) / "images"

    logger.info(f"Loading annotations from {annotations_path}")
    with open(annotations_path, "r") as f:
        metadata = json.load(f)

    prompt_text = (
        "Describe this fashion image in detail. Include:\n"
        "1) All visible clothing items with their exact colors, patterns, and fabric type\n"
        "2) The setting or environment (indoor/outdoor, office, street, park, home, event, etc.)\n"
        "3) The overall style or vibe (formal, casual, sporty, elegant, streetwear, etc.)\n"
        "4) Any accessories (bags, jewelry, hats, shoes, etc.)\n"
        "Be specific about colors (e.g., 'navy blue' not just 'blue') and patterns "
        "(e.g., 'pinstripe' not just 'striped')."
    )

    results = []

    logger.info(f"Generating captions for {len(metadata)} images...")
    for item in tqdm(metadata, desc="Captioning"):
        image_path = images_dir / item["file_name"]

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            results.append({
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "vlm_caption": "",
                "structured_caption": item.get("structured_caption", ""),
            })
            continue

        try:
            raw_image = Image.open(image_path).convert("RGB")

            # Build chat messages in the Qwen3-VL format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": raw_image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            # Use processor.apply_chat_template (the official Qwen3-VL pattern)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                top_p=0.95,
                top_k=20,
                temperature=1.0,
            )

            # Trim prompt tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            caption = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            results.append({
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "vlm_caption": caption,
                "structured_caption": item.get("structured_caption", ""),
            })
            logger.info(f"  [{item['file_name']}] caption: {caption[:80]}...")

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}", exc_info=True)
            results.append({
                "image_id": item["image_id"],
                "file_name": item["file_name"],
                "vlm_caption": "",
                "structured_caption": item.get("structured_caption", ""),
            })

    captions_path = Path(REMOTE_DATA_DIR) / "captions.json"
    with open(captions_path, "w") as f:
        json.dump(results, f, indent=2)

    volume.commit()
    logger.info(f"Saved {len([r for r in results if r['vlm_caption']])} captions to {captions_path}")
    return results


@app.local_entrypoint()
def main():
    logger.info("Uploading dataset to Modal volume...")

    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(IMAGES_DIR), "/images")
        batch.put_file(str(ANNOTATIONS_FILE), "/annotations.json")

    logger.info("Starting remote caption generation...")
    results = generate_captions.remote()

    good = len([r for r in results if r["vlm_caption"]])
    logger.info(f"Generated {good}/{len(results)} captions. Saving locally...")

    with open(CAPTIONS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Done! Saved locally to {CAPTIONS_FILE}")
