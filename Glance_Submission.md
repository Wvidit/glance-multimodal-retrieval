# Glance ML Internship Assignment: Multimodal Fashion & Context Retrieval

## 1. Approaches
To build a system capable of searching fashion images by specific clothing attributes, environment context, and style vibe, several approaches were evaluated:

**A. Zero-Shot Vision-Language Models (e.g., CLIP, SigLIP)**
*   **How it works:** Encodes both images and search queries into a shared multi-modal embedding space. Retrieval is performed using cosine similarity.
*   **Trade-offs:** Highly scalable and fast. However, standard CLIP models struggle severely with compositionality (e.g., distinguishing "red shirt, blue pants" from "blue shirt, red pants") and often ignore fine-grained fashion terminology.
*   **When to use:** Good as a baseline for broad semantic searches, but insufficient for strict attribute constraints.

**B. Object Detection + Attribute Classification (e.g., Faster R-CNN)**
*   **How it works:** Detects bounding boxes for clothing items and runs a classifier to predict color, fabric, and style for each box.
*   **Trade-offs:** Highly accurate and compositional. However, the pipeline is extremely rigid, relies heavily on predefined bounding-box datasets, and scales poorly to open-vocabulary natural language queries.
*   **When to use:** Ideal for structured e-commerce databases, but fails for contextual "vibe" or "environment" searches.

**C. Hybrid Semantic Retrieval (The Chosen Approach)**
*   **How it works:** Fuses the strengths of state-of-the-art vision encoders (FashionSigLIP) with Large Vision-Language Models (Qwen3-VL) to extract both visual embeddings and highly detailed text captions for each image.
*   **Trade-offs:** Requires more upfront compute during the indexing phase (VLM inference), but provides unparalleled retrieval accuracy for compositional and context-aware queries.
*   **When to use:** Best for complex, multi-attribute, open-vocabulary semantic search.

---

## 2. Short Write-up on Chosen Approach

The implemented system leverages a **Hybrid Semantic Retrieval Pipeline**. 

### The Indexing Architecture
For every image in the dataset, the indexer extracts two representations:
1.  **Visual Embeddings:** We use `Marqo-FashionSigLIP`, a variant of SigLIP explicitly fine-tuned on fashion datasets, to generate a 768-dimensional visual embedding. This captures the raw visual semantics of the image.
2.  **Semantic Captions:** We deploy `Qwen3-VL-8B-Thinking` (fine-tuned using LoRA on the Fashionpedia ontology) to generate an incredibly detailed structured caption. It explicitly identifies environments, vibes, clothing patterns, colors, and accessories. These captions are embedded into text vectors.

Both embeddings, along with the raw metadata, are stored in **ChromaDB**.

### The Retrieval Architecture
When a user issues a natural language query (e.g., "A red tie and a white shirt in a formal setting"):
1.  The query is embedded using the FashionSigLIP text encoder.
2.  The engine performs an approximate nearest neighbor search across both the visual embedding collection and the VLM caption embedding collection.
3.  A **Hybrid Scoring Function** is applied:
    *   `Score = (α * Visual Similarity) + (β * Caption Similarity) + (γ * Metadata Boost)`
    *   The Metadata Boost heavily penalizes candidates that lack the specific colors or objects explicitly mentioned in the query, solving CLIP's compositionality failure.

This architecture handles fashion queries flawlessly because the VLM acts as an extremely powerful feature extractor for environment and style, while FashionSigLIP handles fine-grained visual matching.

---

## 3. Codebase Link
The complete codebase is documented and structured cleanly. 
(Insert GitHub URL here)

*Note: The codebase utilizes Modal serverless infrastructure to scale VLM inference and fine-tuning across H100 GPUs.*

---

## 4. Approaches for Future Work

### A. Adding Locations (Cities, Places) and Weather
*   **Geospatial & Temporal Indexing:** If the dataset contains GPS coordinates or timestamps, we can integrate a spatial-temporal vector database (like Qdrant or Milvus with geo-filtering).
*   **VLM Prompt Expansion:** We can explicitly prompt the indexing VLM to infer weather (e.g., "sunny", "raining", "winter snow") and architectural location cues (e.g., "Parisian streets", "beachfront") from the image background. The retrieval engine can then weight these tokens heavily when the user asks for "winter coats for a trip to New York."

### B. Improving Precision
*   **Cross-Encoder Re-Ranking:** While the current system uses bi-encoders (SigLIP) for fast retrieval, precision can be significantly improved by applying a heavy Cross-Encoder (like ALBEF or BLIP-2 ITM) to the top-50 results. The cross-encoder can attend to the interaction between query words and image patches, entirely eliminating compositionality errors (e.g., filtering out the "blue shirt, red pants" false positives).
*   **Hard Negative Mining:** Fine-tuning the FashionSigLIP encoder using triplet loss with hard negatives (images that contain the exact same items but in swapped colors) will force the visual embeddings to become strictly compositional at the bi-encoder level.
