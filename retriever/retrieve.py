"""
CLI entry point for the Glance Fashion Retrieval system.

Runs hybrid search (visual + caption + metadata), optionally re-ranks the
candidates with BLIP-2 image–text matching, then prints a formatted results
table and can display / save result images.

Usage examples:
    # Basic search
    python -m retriever.retrieve "red cocktail dress"

    # Show images, skip re-ranking
    python -m retriever.retrieve "blue denim jacket" --show_images --no_rerank

    # Save top-3 results to disk
    python -m retriever.retrieve "summer floral skirt" --top_k 3 --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    IMAGES_DIR,
    RERANK_MODEL_ID,
    TOP_K_FINAL,
    TOP_K_RETRIEVAL,
)
from retriever.search import HybridSearcher, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLIP-2 re-ranking
# ---------------------------------------------------------------------------

def _rerank_with_blip2(
    results: list[SearchResult],
    query: str,
    images_dir: Path = IMAGES_DIR,
) -> list[SearchResult]:
    """Re-rank *results* using BLIP-2 image–text matching (ITM) scores.

    If the model cannot be loaded (e.g. no GPU, missing weights) the function
    falls back gracefully and returns the original list unchanged.

    Parameters
    ----------
    results : list[SearchResult]
        Candidates from hybrid retrieval.
    query : str
        The original search query.
    images_dir : Path
        Directory containing the source images.

    Returns
    -------
    list[SearchResult]
        Results re-sorted by BLIP-2 ITM score (descending).
    """
    try:
        import torch
        from PIL import Image
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info(
            "Loading BLIP-2 re-ranker (%s) on %s …", RERANK_MODEL_ID, device
        )
        processor = Blip2Processor.from_pretrained(RERANK_MODEL_ID)
        model = Blip2ForConditionalGeneration.from_pretrained(
            RERANK_MODEL_ID, torch_dtype=dtype
        ).to(device)
        model.eval()

        itm_scores: list[float] = []
        for res in results:
            img_path = images_dir / res.file_name
            if not img_path.exists():
                logger.warning("Image not found for re-ranking: %s", img_path)
                itm_scores.append(0.0)
                continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(
                images=image, text=query, return_tensors="pt"
            ).to(device, dtype)

            with torch.no_grad():
                # Use the image-text matching head.  BLIP-2 exposes ITM via
                # the ``itm_score`` output when using ``forward`` with the
                # ``use_itm_head`` flag on the vision-language model, but the
                # HuggingFace API varies across versions.  A portable
                # approach: generate a short answer to "Does this image match
                # the query?" and use the logit of the first generated token
                # as a proxy confidence score.
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                # The logit of the very first generated token gives a usable
                # relative confidence signal across candidates.
                first_token_logits = outputs.scores[0][0]
                score = first_token_logits.max().item()

            itm_scores.append(score)
            logger.debug(
                "  BLIP-2 score for %s: %.4f", res.file_name, score
            )

        # Attach ITM score and re-sort.
        for res, itm in zip(results, itm_scores):
            res.score = itm

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("BLIP-2 re-ranking complete.")
        return results

    except Exception as exc:
        logger.warning(
            "BLIP-2 re-ranking failed (%s). Falling back to hybrid scores.",
            exc,
        )
        return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_results_table(results: list[SearchResult], query: str) -> None:
    """Print a nicely formatted results table to stdout."""
    cap_width = 40
    header = f"{'Rank':>4} │ {'Score':>7} │ {'Image':<20} │ Caption (truncated)"
    sep = "─" * 5 + "┼" + "─" * 9 + "┼" + "─" * 22 + "┼" + "─" * (cap_width + 2)

    print(f"\n  Query: \"{query}\"\n")
    print(f"  {header}")
    print(f"  {sep}")

    for rank, res in enumerate(results, start=1):
        caption = res.vlm_caption or res.structured_caption or ""
        if len(caption) > cap_width:
            caption = caption[: cap_width - 3] + "..."
        print(
            f"  {rank:>4} │ {res.score:>7.4f} │ {res.file_name:<20} │ {caption}"
        )

    print()


def _show_images(
    results: list[SearchResult],
    query: str,
    images_dir: Path = IMAGES_DIR,
) -> None:
    """Display result images in a matplotlib grid."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        logger.warning("matplotlib / Pillow not installed – cannot display images.")
        return

    n = len(results)
    cols = min(n, 5)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flat  # type: ignore[union-attr]

    for idx, (ax, res) in enumerate(zip(axes, results)):
        img_path = images_dir / res.file_name
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center")
        ax.set_title(f"#{idx + 1}  score={res.score:.3f}", fontsize=9)
        ax.axis("off")

    # Hide unused axes.
    for ax in list(axes)[n:]:
        ax.set_visible(False)

    fig.suptitle(f'Query: "{query}"', fontsize=12)
    plt.tight_layout()
    plt.show()


def _save_results(
    results: list[SearchResult],
    query: str,
    output_dir: Path,
    images_dir: Path = IMAGES_DIR,
) -> None:
    """Copy result images and write a ``results.json`` to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for rank, res in enumerate(results, start=1):
        src = images_dir / res.file_name
        dst = output_dir / res.file_name
        if src.exists():
            shutil.copy2(src, dst)
        manifest.append(
            {
                "rank": rank,
                "image_id": res.image_id,
                "file_name": res.file_name,
                "score": round(res.score, 6),
                "visual_score": round(res.visual_score, 6),
                "caption_score": round(res.caption_score, 6),
                "metadata_score": round(res.metadata_score, 6),
                "vlm_caption": res.vlm_caption,
                "structured_caption": res.structured_caption,
                "categories": res.categories,
                "attributes": res.attributes,
            }
        )

    results_json = output_dir / "results.json"
    results_json.write_text(
        json.dumps({"query": query, "results": manifest}, indent=2),
        encoding="utf-8",
    )
    logger.info("Results saved to %s", output_dir)
    print(f"  Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Glance Fashion Retrieval – search for fashion images by text.",
    )
    parser.add_argument(
        "query",
        type=str,
        help="Natural-language search query.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_FINAL,
        help=f"Number of results to return (default: {TOP_K_FINAL}).",
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Skip BLIP-2 re-ranking and use hybrid scores only.",
    )
    parser.add_argument(
        "--show_images",
        action="store_true",
        help="Display result images in a matplotlib window.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Save result images and metadata to this directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the end-to-end retrieval pipeline from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()

    # 1. Hybrid search.
    searcher = HybridSearcher()
    results = searcher.search(args.query, top_k=TOP_K_RETRIEVAL)

    if not results:
        print("No results found.")
        return

    # 2. Optional BLIP-2 re-ranking.
    if not args.no_rerank:
        results = _rerank_with_blip2(results, args.query)

    # Trim to final top_k after re-ranking.
    results = results[: args.top_k]

    # 3. Display results table.
    _print_results_table(results, args.query)

    # 4. Optional image display.
    if args.show_images:
        _show_images(results, args.query)

    # 5. Optional save.
    if args.output_dir:
        _save_results(results, args.query, Path(args.output_dir))


if __name__ == "__main__":
    main()
