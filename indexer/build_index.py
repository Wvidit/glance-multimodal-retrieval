"""
Builds the ChromaDB vector index locally from generated embeddings.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import chromadb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ANNOTATIONS_FILE,
    CAPTION_COLLECTION,
    CAPTION_EMBEDS_FILE,
    CAPTIONS_FILE,
    CHROMA_DIR,
    IMAGE_IDS_FILE,
    VISUAL_COLLECTION,
    VISUAL_EMBEDS_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_index():
    if not VISUAL_EMBEDS_FILE.exists() or not CAPTION_EMBEDS_FILE.exists():
        logger.error("Embedding files not found. Run embed_modal.py first.")
        return
        
    if not IMAGE_IDS_FILE.exists():
        logger.error("Image IDs file not found.")
        return
        
    # Load embeddings
    logger.info("Loading embeddings from disk...")
    visual_embeds = np.load(VISUAL_EMBEDS_FILE)["embeddings"]
    caption_embeds = np.load(CAPTION_EMBEDS_FILE)["embeddings"]
    
    with open(IMAGE_IDS_FILE, 'r') as f:
        image_ids_list = json.load(f)
        
    # Load metadata (prefer captions if available)
    metadata_path = CAPTIONS_FILE if CAPTIONS_FILE.exists() else ANNOTATIONS_FILE
    logger.info(f"Loading metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata_list = json.load(f)
        
    # Create lookup map
    meta_map = {str(item["image_id"]): item for item in metadata_list}
    
    # Initialize ChromaDB
    logger.info(f"Initializing ChromaDB at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing if any
    try:
        client.delete_collection(VISUAL_COLLECTION)
        logger.info(f"Deleted existing collection {VISUAL_COLLECTION}")
    except Exception:
        pass
        
    try:
        client.delete_collection(CAPTION_COLLECTION)
        logger.info(f"Deleted existing collection {CAPTION_COLLECTION}")
    except Exception:
        pass
        
    visual_col = client.create_collection(name=VISUAL_COLLECTION, metadata={"hnsw:space": "cosine"})
    caption_col = client.create_collection(name=CAPTION_COLLECTION, metadata={"hnsw:space": "cosine"})
    
    # Prepare data in batches for Chroma
    batch_size = 5000
    total = len(image_ids_list)
    
    logger.info(f"Inserting {total} items into ChromaDB collections...")
    
    for i in tqdm(range(0, total, batch_size), desc="Inserting into ChromaDB"):
        batch_ids = [str(x) for x in image_ids_list[i:i+batch_size]]
        batch_visual = visual_embeds[i:i+batch_size].tolist()
        batch_caption = caption_embeds[i:i+batch_size].tolist()
        
        # Prepare metadata formatting for ChromaDB (must be strings, ints, floats)
        batch_meta = []
        for img_id in batch_ids:
            meta = meta_map.get(img_id, {})
            clean_meta = {
                "image_id": str(img_id),
                "file_name": meta.get("file_name", ""),
                "categories": ",".join(meta.get("categories", [])),
                "attributes": ",".join(meta.get("attributes", [])),
                "vlm_caption": meta.get("vlm_caption", ""),
                "structured_caption": meta.get("structured_caption", "")
            }
            batch_meta.append(clean_meta)
            
        visual_col.add(
            ids=batch_ids,
            embeddings=batch_visual,
            metadatas=batch_meta
        )
        
        caption_col.add(
            ids=batch_ids,
            embeddings=batch_caption,
            metadatas=batch_meta
        )
        
    logger.info("Done building index!")
    logger.info(f"Visual collection count: {visual_col.count()}")
    logger.info(f"Caption collection count: {caption_col.count()}")


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index from embeddings.")
    args = parser.parse_args()
    build_index()


if __name__ == "__main__":
    main()
