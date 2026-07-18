"""
Shared configuration for the Glance Fashion Retrieval system.
"""
from pathlib import Path

# ──────────────────────────────────────────────
# Model identifiers
# ──────────────────────────────────────────────
CAPTION_MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
EMBED_MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
EMBED_MODEL_PRETRAINED = "Marqo/marqo-fashionSigLIP"
RERANK_MODEL_ID = "Salesforce/blip2-flan-t5-xl"

# ──────────────────────────────────────────────
# Directory layout
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CAPTIONS_FILE = DATA_DIR / "captions.json"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
VISUAL_EMBEDS_FILE = EMBEDDINGS_DIR / "visual_embeddings.npz"
CAPTION_EMBEDS_FILE = EMBEDDINGS_DIR / "caption_embeddings.npz"
IMAGE_IDS_FILE = EMBEDDINGS_DIR / "image_ids.json"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ──────────────────────────────────────────────
# Embedding dimensions (FashionSigLIP ViT-B/16)
# ──────────────────────────────────────────────
EMBED_DIM = 768

# ──────────────────────────────────────────────
# ChromaDB collection names
# ──────────────────────────────────────────────
VISUAL_COLLECTION = "visual_embeddings"
CAPTION_COLLECTION = "caption_embeddings"

# ──────────────────────────────────────────────
# Retrieval hyper-parameters
# ──────────────────────────────────────────────
TOP_K_RETRIEVAL = 50        # candidates from vector search
TOP_K_FINAL = 10            # final results after re-ranking
ALPHA_VISUAL = 0.4          # weight for visual similarity
ALPHA_CAPTION = 0.5         # weight for caption similarity
ALPHA_METADATA = 0.1        # weight for metadata keyword boost

# ──────────────────────────────────────────────
# Modal configuration
# ──────────────────────────────────────────────
MODAL_VOLUME_NAME = "glance-data"
MODAL_GPU = "H100"
MODAL_TIMEOUT = 3600        # 1 hour

# ──────────────────────────────────────────────
# Dataset curation
# ──────────────────────────────────────────────
TARGET_SUBSET_SIZE = 8500   # number of images to use
FASHIONPEDIA_TRAIN_IMAGES_URL = (
    "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
)
FASHIONPEDIA_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/ifashionist-dataset/annotations/"
    "instances_attributes_train2020.json"
)
