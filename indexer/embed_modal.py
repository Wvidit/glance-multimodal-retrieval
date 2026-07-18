"""
Modal app for generating embeddings using Marqo-FashionSigLIP on H100 GPU.
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
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VISUAL_EMBEDS_FILE = EMBEDDINGS_DIR / "visual_embeddings.npz"
CAPTION_EMBEDS_FILE = EMBEDDINGS_DIR / "caption_embeddings.npz"
IMAGE_IDS_FILE = EMBEDDINGS_DIR / "image_ids.json"
EMBED_MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
MODAL_VOLUME_NAME = "glance-data"
MODAL_GPU = "H100"
MODAL_TIMEOUT = 3600

logger = logging.getLogger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "open-clip-torch",
        "Pillow",
        "tqdm",
        "numpy",
        "transformers",
    )
    .add_local_file(str(Path(__file__).resolve().parent.parent / "config.py"), remote_path="/root/config.py")
)

app = modal.App("glance-embedder")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

REMOTE_DATA_DIR = "/data"

@app.function(
    image=image,
    gpu=MODAL_GPU,
    timeout=MODAL_TIMEOUT,
    volumes={REMOTE_DATA_DIR: volume},
    scaledown_window=300,
)
def generate_embeddings():
    """Generate visual and text embeddings using Marqo-FashionSigLIP."""
    import numpy as np
    import open_clip
    import torch
    from PIL import Image
    from tqdm import tqdm

    logger.info("Loading FashionSigLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(EMBED_MODEL_ID)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(EMBED_MODEL_ID)

    # We read from captions if available (which also has structured_caption)
    # If not, we fall back to annotations
    captions_path = Path(REMOTE_DATA_DIR) / "captions.json"
    if not captions_path.exists():
        captions_path = Path(REMOTE_DATA_DIR) / "annotations.json"
        
    logger.info(f"Loading metadata from {captions_path}")
    with open(captions_path, 'r') as f:
        metadata = json.load(f)

    images_dir = Path(REMOTE_DATA_DIR) / "images"
    
    image_ids = []
    visual_embeds = []
    caption_embeds = []

    batch_size = 64
    
    logger.info(f"Generating embeddings for {len(metadata)} items in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(metadata), batch_size), desc="Embedding"):
        batch_meta = metadata[i:i+batch_size]
        
        # 1. Process Images
        images = []
        valid_items = []
        for item in batch_meta:
            img_path = images_dir / item["file_name"]
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(preprocess_val(img))
                    valid_items.append(item)
                except Exception as e:
                    logger.error(f"Error loading {img_path}: {e}")
                    
        if not images:
            continue
            
        image_input = torch.stack(images).to(device)
        
        # 2. Process Text (VLM caption if exists, else structured)
        texts = []
        for item in valid_items:
            # Prefer VLM caption, fallback to structured caption
            text = item.get("vlm_caption", item.get("structured_caption", ""))
            texts.append(text)
            
        text_input = tokenizer(texts).to(device)
        
        # 3. Generate embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            
            # 4. Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        # Store results
        visual_embeds.append(image_features.cpu().numpy())
        caption_embeds.append(text_features.cpu().numpy())
        image_ids.extend([item["image_id"] for item in valid_items])
        
    # Concatenate all batches
    final_visual = np.vstack(visual_embeds).astype(np.float32)
    final_caption = np.vstack(caption_embeds).astype(np.float32)
    
    logger.info(f"Generated visual embeddings shape: {final_visual.shape}")
    logger.info(f"Generated caption embeddings shape: {final_caption.shape}")
    
    # Save to remote volume
    out_dir = Path(REMOTE_DATA_DIR) / "embeddings"
    out_dir.mkdir(exist_ok=True)
    
    np.savez_compressed(out_dir / "visual_embeddings.npz", embeddings=final_visual)
    np.savez_compressed(out_dir / "caption_embeddings.npz", embeddings=final_caption)
    
    with open(out_dir / "image_ids.json", 'w') as f:
        json.dump(image_ids, f)
        
    volume.commit()
    logger.info("Embeddings generated and saved to volume.")
    
    return final_visual, final_caption, image_ids


@app.local_entrypoint()
def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Uploading dataset to Modal volume...")
    
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(IMAGES_DIR), "/images")
        
        if CAPTIONS_FILE.exists():
            batch.put_file(str(CAPTIONS_FILE), "/captions.json")
        if ANNOTATIONS_FILE.exists():
            batch.put_file(str(ANNOTATIONS_FILE), "/annotations.json")
            
    logger.info("Starting remote embedding generation...")
    visual, caption, ids = generate_embeddings.remote()
    
    logger.info("Saving embeddings locally...")
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    np.savez_compressed(VISUAL_EMBEDS_FILE, embeddings=visual)
    np.savez_compressed(CAPTION_EMBEDS_FILE, embeddings=caption)
    
    with open(IMAGE_IDS_FILE, 'w') as f:
        json.dump(ids, f)
        
    logger.info(f"Done! Embeddings saved to {EMBEDDINGS_DIR}")
