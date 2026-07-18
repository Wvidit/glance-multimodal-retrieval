"""
Evaluation script for the Glance Fashion Retrieval system.

Runs a fixed set of five benchmark queries, collects retrieval results, and
generates a self-contained HTML report with embedded base64 images and score
tables.

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --top_k 10 --output report.html --no_rerank
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import logging
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, IMAGES_DIR, TOP_K_FINAL
from retriever.search import HybridSearcher, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark queries
# ---------------------------------------------------------------------------

EVAL_QUERIES: list[str] = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def _image_to_base64_uri(image_path: Path, max_size: int = 300) -> str:
    """Return a base64 data-URI string for *image_path*.

    The image is resized so the longest edge is at most *max_size* pixels to
    keep the HTML file size manageable.
    """
    try:
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img.thumbnail((max_size, max_size))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception as exc:
        logger.warning("Could not encode image %s: %s", image_path, exc)
        return ""


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: #f4f6f9; color: #222; padding: 2rem;
}
h1 { text-align: center; margin-bottom: 0.5rem; }
.subtitle { text-align: center; color: #666; margin-bottom: 2rem; font-size: 0.95rem; }
table.summary {
  margin: 0 auto 2.5rem; border-collapse: collapse; width: auto;
}
table.summary th, table.summary td {
  padding: 0.5rem 1rem; border: 1px solid #d0d5dd; text-align: left;
}
table.summary th { background: #e2e6ec; }
.query-section { margin-bottom: 3rem; }
.query-title {
  font-size: 1.15rem; font-weight: 600; margin-bottom: 1rem;
  padding: 0.6rem 1rem; background: #fff; border-left: 4px solid #4a7cff;
  border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.results-grid {
  display: flex; flex-wrap: wrap; gap: 1rem; justify-content: flex-start;
}
.result-card {
  background: #fff; border-radius: 8px; overflow: hidden; width: 200px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.08); display: flex; flex-direction: column;
}
.result-card img { width: 100%; height: 200px; object-fit: cover; }
.result-card .info { padding: 0.6rem; font-size: 0.82rem; }
.result-card .info .rank { font-weight: 700; color: #4a7cff; }
.result-card .info .score { font-weight: 600; }
.result-card .info .caption {
  margin-top: 0.3rem; color: #555; line-height: 1.35;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
  overflow: hidden;
}
"""


def _build_summary_table(
    all_results: dict[str, list[SearchResult]],
) -> str:
    """Build an HTML summary table showing each query and its top-1 score."""
    rows = ""
    for idx, (query, results) in enumerate(all_results.items(), start=1):
        top_score = results[0].score if results else 0.0
        top_file = results[0].file_name if results else "—"
        rows += (
            f"<tr>"
            f"<td>{idx}</td>"
            f"<td>{html_mod.escape(query)}</td>"
            f"<td>{top_score:.4f}</td>"
            f"<td>{html_mod.escape(top_file)}</td>"
            f"<td>{len(results)}</td>"
            f"</tr>\n"
        )
    return (
        '<table class="summary">\n'
        "<tr><th>#</th><th>Query</th><th>Top-1 Score</th>"
        "<th>Top-1 Image</th><th>Results</th></tr>\n"
        f"{rows}</table>\n"
    )


def _build_query_section(
    idx: int,
    query: str,
    results: list[SearchResult],
    images_dir: Path,
) -> str:
    """Build the HTML block for one evaluation query."""
    cards = ""
    for rank, res in enumerate(results, start=1):
        img_path = images_dir / res.file_name
        data_uri = _image_to_base64_uri(img_path) if img_path.exists() else ""
        img_tag = (
            f'<img src="{data_uri}" alt="{html_mod.escape(res.file_name)}">'
            if data_uri
            else '<div style="width:100%;height:200px;background:#eee;'
            'display:flex;align-items:center;justify-content:center;'
            'color:#999;">No image</div>'
        )
        caption_text = html_mod.escape(
            res.vlm_caption or res.structured_caption or ""
        )
        cards += (
            '<div class="result-card">\n'
            f"  {img_tag}\n"
            '  <div class="info">\n'
            f'    <span class="rank">#{rank}</span> '
            f'<span class="score">Score: {res.score:.4f}</span><br>\n'
            f"    V:{res.visual_score:.3f} C:{res.caption_score:.3f} "
            f"M:{res.metadata_score:.3f}<br>\n"
            f'    <div class="caption">{caption_text}</div>\n'
            "  </div>\n"
            "</div>\n"
        )

    return (
        '<div class="query-section">\n'
        f'  <div class="query-title">Query {idx}: "{html_mod.escape(query)}"</div>\n'
        f'  <div class="results-grid">\n{cards}  </div>\n'
        "</div>\n"
    )


def _generate_html_report(
    all_results: dict[str, list[SearchResult]],
    output_path: Path,
    images_dir: Path,
) -> None:
    """Write the full evaluation report to *output_path*."""
    summary = _build_summary_table(all_results)
    sections = ""
    for idx, (query, results) in enumerate(all_results.items(), start=1):
        sections += _build_query_section(idx, query, results, images_dir)

    html_content = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
        "<title>Glance – Evaluation Report</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n<body>\n"
        "<h1>Glance Fashion Retrieval – Evaluation Report</h1>\n"
        '<p class="subtitle">Hybrid retrieval results for 5 benchmark queries</p>\n'
        f"{summary}\n{sections}\n"
        "</body>\n</html>\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------


def _print_console_results(all_results: dict[str, list[SearchResult]]) -> None:
    """Print a concise summary of all evaluation results to stdout."""
    for idx, (query, results) in enumerate(all_results.items(), start=1):
        print(f"\n{'═' * 70}")
        print(f"  Query {idx}: \"{query}\"")
        print(f"{'─' * 70}")
        cap_w = 35
        header = f"  {'Rank':>4} │ {'Score':>7} │ {'V':>5} │ {'C':>5} │ {'M':>5} │ {'Image':<18} │ Caption"
        print(header)
        print(
            "  "
            + "─" * 5
            + "┼"
            + "─" * 9
            + "┼"
            + "─" * 7
            + "┼"
            + "─" * 7
            + "┼"
            + "─" * 7
            + "┼"
            + "─" * 20
            + "┼"
            + "─" * (cap_w + 2)
        )
        for rank, res in enumerate(results, start=1):
            caption = res.vlm_caption or res.structured_caption or ""
            if len(caption) > cap_w:
                caption = caption[: cap_w - 3] + "..."
            print(
                f"  {rank:>4} │ {res.score:>7.4f} │ "
                f"{res.visual_score:>5.3f} │ {res.caption_score:>5.3f} │ "
                f"{res.metadata_score:>5.3f} │ {res.file_name:<18} │ {caption}"
            )
    print(f"\n{'═' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Glance retrieval pipeline on benchmark queries.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results per query (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DATA_DIR / "evaluation_report.html"),
        help="Path for the HTML evaluation report (default: data/evaluation_report.html).",
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Skip BLIP-2 re-ranking.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full evaluation pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()
    output_path = Path(args.output)

    # --- Initialise searcher ------------------------------------------------
    searcher = HybridSearcher()

    # --- Optionally load BLIP-2 re-ranker once ------------------------------
    rerank_fn = None
    if not args.no_rerank:
        try:
            # Import the re-ranking function from the retrieve module.
            from retriever.retrieve import _rerank_with_blip2

            rerank_fn = _rerank_with_blip2
            logger.info("BLIP-2 re-ranker available.")
        except Exception as exc:
            logger.warning(
                "Could not load BLIP-2 re-ranker (%s). Proceeding without re-ranking.",
                exc,
            )

    # --- Run each query -----------------------------------------------------
    all_results: dict[str, list[SearchResult]] = {}

    for query in EVAL_QUERIES:
        logger.info("Running query: '%s'", query)
        results = searcher.search(query, top_k=args.top_k)

        if rerank_fn is not None and results:
            results = rerank_fn(results, query)
            results = results[: args.top_k]

        all_results[query] = results

    # --- Console output -----------------------------------------------------
    _print_console_results(all_results)

    # --- HTML report --------------------------------------------------------
    _generate_html_report(all_results, output_path, IMAGES_DIR)
    print(f"  HTML report saved to: {output_path}")


if __name__ == "__main__":
    main()
