"""
Hybrid retrieval module for the Glance Fashion Retrieval system.

Combines three signals to rank fashion images against a text query:
  1. Visual similarity  – FashionSigLIP text→image embedding distance in ChromaDB.
  2. Caption similarity  – FashionSigLIP text→caption embedding distance in ChromaDB.
  3. Metadata keyword boost – token overlap between query and structured metadata.

Usage:
    from retriever.search import HybridSearcher
    searcher = HybridSearcher()
    results = searcher.search("red cocktail dress", top_k=5)
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project imports – add project root so ``config`` is importable everywhere.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ALPHA_CAPTION,
    ALPHA_METADATA,
    ALPHA_VISUAL,
    CAPTION_COLLECTION,
    CHROMA_DIR,
    EMBED_MODEL_ID,
    TOP_K_FINAL,
    TOP_K_RETRIEVAL,
    VISUAL_COLLECTION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single retrieval result with per-signal scores."""

    image_id: str
    file_name: str
    score: float
    visual_score: float
    caption_score: float
    metadata_score: float
    vlm_caption: str = ""
    structured_caption: str = ""
    categories: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hybrid searcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """Hybrid retrieval engine over ChromaDB visual + caption collections.

    Parameters
    ----------
    chroma_dir : Path | str | None
        Override for the ChromaDB persistence directory.  Defaults to the
        value defined in ``config.py``.
    """

    def __init__(self, chroma_dir: Optional[Path | str] = None) -> None:
        self._chroma_dir = Path(chroma_dir) if chroma_dir else CHROMA_DIR

        # --- ChromaDB --------------------------------------------------
        self._init_chroma()

        # --- FashionSigLIP text encoder --------------------------------
        self._init_text_encoder()

    # -----------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------

    def _init_chroma(self) -> None:
        """Connect to the persisted ChromaDB store and load both collections."""
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self._chroma_dir))

            self._visual_col = self._client.get_collection(name=VISUAL_COLLECTION)
            logger.info(
                "Loaded visual collection '%s' (%d items)",
                VISUAL_COLLECTION,
                self._visual_col.count(),
            )

            self._caption_col = self._client.get_collection(name=CAPTION_COLLECTION)
            logger.info(
                "Loaded caption collection '%s' (%d items)",
                CAPTION_COLLECTION,
                self._caption_col.count(),
            )
        except Exception as exc:
            logger.error("Failed to initialise ChromaDB: %s", exc)
            raise RuntimeError(
                f"Could not open ChromaDB at {self._chroma_dir}. "
                "Make sure the indexer has been run first."
            ) from exc

    def _init_text_encoder(self) -> None:
        """Load the FashionSigLIP model and tokenizer for text encoding."""
        try:
            import open_clip
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            model, _, preprocess = open_clip.create_model_and_transforms(
                EMBED_MODEL_ID
            )
            self._model = model.to(self._device).eval()
            self._tokenizer = open_clip.get_tokenizer(EMBED_MODEL_ID)
            self._preprocess = preprocess

            logger.info(
                "FashionSigLIP text encoder loaded on %s", self._device
            )
        except Exception as exc:
            logger.error("Failed to load FashionSigLIP: %s", exc)
            raise RuntimeError(
                "Could not load the FashionSigLIP text encoder. "
                "Ensure open_clip_torch is installed."
            ) from exc

    # -----------------------------------------------------------------
    # Query encoding
    # -----------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a text query into a normalised embedding vector.

        Parameters
        ----------
        query : str
            Natural-language search query.

        Returns
        -------
        np.ndarray
            1-D float32 embedding of shape ``(EMBED_DIM,)``.
        """
        import torch

        tokens = self._tokenizer([query]).to(self._device)
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype(np.float32).flatten()

    # -----------------------------------------------------------------
    # Metadata boosting
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_metadata_boost(query: str, metadata: dict) -> float:
        """Compute keyword-overlap score between *query* and metadata fields.

        The score is the fraction of unique query tokens that appear in the
        combined (lowercased) categories + attributes of the item, yielding a
        value in ``[0.0, 1.0]``.

        Parameters
        ----------
        query : str
            The user's search query.
        metadata : dict
            Metadata dict as stored in ChromaDB.  Expected keys include
            ``"categories"`` and ``"attributes"`` (comma-separated strings).

        Returns
        -------
        float
            Overlap fraction between 0 and 1.
        """
        # Tokenise query into lowercase words (strip punctuation).
        query_tokens = set(re.findall(r"[a-z]+", query.lower()))
        if not query_tokens:
            return 0.0

        # Build a set of metadata tokens from categories + attributes.
        raw_categories = metadata.get("categories", "")
        raw_attributes = metadata.get("attributes", "")
        meta_text = f"{raw_categories} {raw_attributes}".lower()
        meta_tokens = set(re.findall(r"[a-z]+", meta_text))

        if not meta_tokens:
            return 0.0

        overlap = query_tokens & meta_tokens
        return len(overlap) / len(query_tokens)

    # -----------------------------------------------------------------
    # Core search
    # -----------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
    ) -> list[SearchResult]:
        """Run hybrid retrieval for *query* and return ranked results.

        Steps
        -----
        1. Encode *query* with FashionSigLIP.
        2. Retrieve ``TOP_K_RETRIEVAL`` nearest neighbours from both the
           visual and caption ChromaDB collections.
        3. Merge candidates by ``image_id``.
        4. Compute per-candidate scores and a weighted final score.
        5. Return the top *top_k* results sorted by descending score.

        Parameters
        ----------
        query : str
            Natural-language fashion search query.
        top_k : int
            Number of final results to return.

        Returns
        -------
        list[SearchResult]
            Ranked search results.
        """
        logger.info("Searching for: '%s'  (top_k=%d)", query, top_k)

        # 1. Encode the query text.
        query_embedding = self._encode_query(query)
        query_list = query_embedding.tolist()

        # 2. Query both collections.
        visual_results = self._visual_col.query(
            query_embeddings=[query_list],
            n_results=TOP_K_RETRIEVAL,
            include=["distances", "metadatas"],
        )
        caption_results = self._caption_col.query(
            query_embeddings=[query_list],
            n_results=TOP_K_RETRIEVAL,
            include=["distances", "metadatas"],
        )

        # 3. Merge candidates keyed by image_id.
        candidates: dict[str, dict] = {}

        # -- Visual hits ------------------------------------------------
        if visual_results and visual_results["ids"]:
            for img_id, dist, meta in zip(
                visual_results["ids"][0],
                visual_results["distances"][0],
                visual_results["metadatas"][0],
            ):
                candidates[img_id] = {
                    "visual_dist": dist,
                    "caption_dist": None,
                    "metadata": meta or {},
                }

        # -- Caption hits -----------------------------------------------
        if caption_results and caption_results["ids"]:
            for img_id, dist, meta in zip(
                caption_results["ids"][0],
                caption_results["distances"][0],
                caption_results["metadatas"][0],
            ):
                if img_id in candidates:
                    candidates[img_id]["caption_dist"] = dist
                    # Merge metadata (caption collection may have richer data).
                    candidates[img_id]["metadata"].update(meta or {})
                else:
                    candidates[img_id] = {
                        "visual_dist": None,
                        "caption_dist": dist,
                        "metadata": meta or {},
                    }

        if not candidates:
            logger.warning("No candidates found for query: '%s'", query)
            return []

        # 4. Score each candidate.
        scored: list[SearchResult] = []
        for img_id, info in candidates.items():
            meta = info["metadata"]

            visual_score = (
                max(0.0, 1.0 - info["visual_dist"])
                if info["visual_dist"] is not None
                else 0.0
            )
            caption_score = (
                max(0.0, 1.0 - info["caption_dist"])
                if info["caption_dist"] is not None
                else 0.0
            )
            metadata_score = self._compute_metadata_boost(query, meta)

            final_score = (
                ALPHA_VISUAL * visual_score
                + ALPHA_CAPTION * caption_score
                + ALPHA_METADATA * metadata_score
            )

            # Parse list-like metadata stored as comma-separated strings.
            categories = [
                c.strip()
                for c in meta.get("categories", "").split(",")
                if c.strip()
            ]
            attributes = [
                a.strip()
                for a in meta.get("attributes", "").split(",")
                if a.strip()
            ]

            scored.append(
                SearchResult(
                    image_id=img_id,
                    file_name=meta.get("file_name", f"{img_id}.jpg"),
                    score=final_score,
                    visual_score=visual_score,
                    caption_score=caption_score,
                    metadata_score=metadata_score,
                    vlm_caption=meta.get("vlm_caption", ""),
                    structured_caption=meta.get("structured_caption", ""),
                    categories=categories,
                    attributes=attributes,
                )
            )

        # 5. Sort descending by final score and return top-k.
        scored.sort(key=lambda r: r.score, reverse=True)
        results = scored[:top_k]

        logger.info(
            "Returning %d results (best score=%.4f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results
