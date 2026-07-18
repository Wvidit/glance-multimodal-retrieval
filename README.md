# Glance: Multimodal Fashion & Context Retrieval

An intelligent fashion image search engine that retrieves images from a database based on natural language descriptions. Goes beyond vanilla CLIP by understanding **what** someone is wearing, **where** they are, and the **vibe** of their attire.

## Architecture

```
Query: "A red tie and a white shirt in a formal setting"
                            │
                   ┌────────▼─────────┐
                   │  FashionSigLIP    │
                   │  Text Encoder     │
                   └────────┬─────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Visual   │  │ Caption  │  │ Metadata │
        │ Search   │  │ Search   │  │ Keyword  │
        │ (0.4)    │  │ (0.5)    │  │ Boost    │
        │          │  │          │  │ (0.1)    │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             └──────────┬───┘──────────────┘
                        ▼
              ┌──────────────────┐
              │  Hybrid Scoring  │
              │  & Merging       │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  BLIP-2 ITM      │
              │  Re-ranking      │
              │  (cross-encoder) │
              └────────┬─────────┘
                       ▼
                 Top-k Results
```

### Why This Beats Vanilla CLIP

| CLIP Limitation | Our Solution |
|---|---|
| Poor compositionality ("red shirt blue pants" ≠ "blue shirt red pants") | VLM captions explicitly describe attribute-object bindings; caption similarity catches this |
| Misses fine-grained fashion attributes (fabric, pattern, neckline) | **Marqo-FashionSigLIP** trained on 800K+ fashion image-text pairs |
| Weak at spatial/contextual reasoning | **BLIP-2 cross-encoder** re-ranker with full cross-attention |
| Generic embeddings | Domain-specific encoder + attribute-aware captions |

## Models Used

| Component | Model | Purpose |
|---|---|---|
| **Encoder** | [Marqo-FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP) | Fashion-domain visual & text embeddings (57% better MRR than FashionCLIP) |
| **Captioner** | [LLaVA-v1.6-Mistral-7B](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) | Rich natural-language fashion descriptions per image |
| **Re-ranker** | [BLIP-2-FlanT5-XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl) | Cross-encoder image-text matching for compositional queries |
| **Vector DB** | [ChromaDB](https://docs.trychroma.com/) | Lightweight persistent vector storage with metadata filtering |

## Dataset

[Fashionpedia](https://fashionpedia.github.io/home/index.html) — 27 apparel categories, 19 parts, 294 fine-grained attributes. We curate a diverse ~3,000 image subset from the training set.

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[all]"
```

### 2. Download & Curate Dataset

```bash
python -m indexer.download_dataset --num_images 3000
```

### 3. Generate Captions (Modal H100)

```bash
modal run indexer/caption_modal.py
```

### 4. Generate Embeddings (Modal H100)

```bash
modal run indexer/embed_modal.py
```

### 5. Build Vector Index

```bash
python -m indexer.build_index
```

### 6. Search!

```bash
# Basic search
python -m retriever.retrieve "A person in a bright yellow raincoat" --top_k 5

# Search with BLIP-2 re-ranking
python -m retriever.retrieve "A red tie and a white shirt in a formal setting" --top_k 5

# Skip re-ranking (faster, CPU only)
python -m retriever.retrieve "Casual weekend outfit" --top_k 5 --no_rerank

# Show result images
python -m retriever.retrieve "Professional business attire" --top_k 5 --show_images
```

### 7. Run Evaluation

```bash
python -m evaluation.evaluate --output data/evaluation_report.html
```

## Project Structure

```
glance/
├── config.py                    # Shared constants (models, paths, weights)
├── pyproject.toml               # Dependencies
├── README.md
│
├── indexer/                     # Part A: The Indexer
│   ├── download_dataset.py      # Download & curate Fashionpedia subset
│   ├── caption_modal.py         # Modal H100: VLM captioning (LLaVA)
│   ├── embed_modal.py           # Modal H100: FashionSigLIP embeddings
│   └── build_index.py           # Build ChromaDB vector index
│
├── retriever/                   # Part B: The Retriever
│   ├── search.py                # Hybrid retrieval engine
│   └── retrieve.py              # CLI entry point + optional BLIP-2 re-ranking
│
├── evaluation/
│   └── evaluate.py              # Run benchmark queries, generate HTML report
│
└── data/                        # Generated at runtime
    ├── images/                  # Downloaded Fashionpedia images
    ├── captions.json            # VLM + structured captions
    ├── annotations.json         # Curated metadata
    ├── embeddings/              # .npz embedding files
    ├── chroma_db/               # ChromaDB persistent storage
    └── evaluation_report.html   # Evaluation results
```

## Hybrid Retrieval Details

The retrieval score for each candidate image is:

```
score = α × visual_sim + β × caption_sim + γ × metadata_boost
```

Where:
- `α = 0.4` — Cosine similarity between query embedding and image visual embedding
- `β = 0.5` — Cosine similarity between query embedding and caption text embedding
- `γ = 0.1` — Keyword overlap between query and structured metadata (categories + attributes)

The higher weight on caption similarity (0.5 vs 0.4) compensates for CLIP's compositionality weakness — VLM captions explicitly bind attributes to objects (e.g., "a **red** tie paired with a **white** button-down shirt"), making text-to-text matching more accurate for multi-attribute queries.

## Scalability

- **ChromaDB** supports millions of vectors with HNSW indexing
- **Embedding generation** can be parallelized across Modal containers
- **Hybrid retrieval** is O(log n) with ANN search + O(k) for re-ranking
- To scale to 1M+ images: swap ChromaDB for Qdrant/Milvus, shard embeddings, cache model on Modal

## Compute Budget

| Task | GPU | Time | Cost |
|---|---|---|---|
| VLM Captioning (3K images) | H100 | ~30-45 min | ~$2-3 |
| Embedding Generation | H100 | ~5 min | ~$0.35 |
| Testing & iteration | H100 | ~30 min | ~$2 |
| **Total** | | | **~$5-6** |

## Future Work

### Adding Locations & Weather
- Augment VLM captioning prompt to emphasize location/weather cues
- Add weather-specific attributes to structured metadata
- Fine-tune on geo-tagged fashion datasets (e.g., StreetStyle)
- Add location-aware embeddings using CLIP + place recognition models

### Improving Precision
- Fine-tune FashionSigLIP with LoRA on Fashionpedia (hard negative contrastive loss)
- Train a lightweight attribute classifier head for explicit attribute prediction
- Use ensemble of multiple re-rankers (BLIP-2 + cross-encoder fine-tuned on fashion)
- Implement query expansion with LLM-generated paraphrases
