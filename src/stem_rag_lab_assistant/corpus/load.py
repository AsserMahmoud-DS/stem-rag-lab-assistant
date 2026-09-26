"""Corpus loading — run OpenDataLoader hybrid on data/*.pdf → loaded_data/*.json. (P0.2)

Requires the hybrid backend to be running:
    opendataloader-pdf-hybrid --port 5002 --enrich-picture-description --enrich-formula
"""

import json
import logging
from pathlib import Path

import opendataloader_pdf

from stem_rag_lab_assistant.config import DATA_DIR, LOADED_DATA_DIR

logger = logging.getLogger(__name__)


def _collect_pdf_paths(data_dir: Path) -> list[str]:
    """Return sorted list of absolute PDF file paths from data_dir."""
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")
    return [str(p.resolve()) for p in pdfs]


def _count_captions(loaded_json_path: Path) -> int:
    """Recursively count 'caption' elements in an OpenDataLoader JSON output."""
    with open(loaded_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    count = 0
    stack = list(doc.get("kids", []))

    while stack:
        element = stack.pop()
        if not isinstance(element, dict):
            continue
        if element.get("type") == "caption":
            count += 1
        for key in ("kids", "list items", "rows"):
            children = element.get(key)
            if isinstance(children, list):
                stack.extend(children)

    return count


def _count_descriptions(loaded_json_path: Path) -> int:
    """Recursively count elements with an AI description field (from Docling --enrich-picture-description).

    Note: ODL outputs "description" or "alt" on picture elements depending on
    backend version; our images.json schema maps these to "ai_description" to
    avoid collision with graph entity descriptions.
    """
    with open(loaded_json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    count = 0
    stack = list(doc.get("kids", []))

    while stack:
        element = stack.pop()
        if not isinstance(element, dict):
            continue
        if element.get("description") or element.get("alt"):
            count += 1
        for key in ("kids", "list items", "rows"):
            children = element.get(key)
            if isinstance(children, list):
                stack.extend(children)

    return count


def run_opendataloader(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Convert all PDFs in data_dir with OpenDataLoader hybrid mode.

    Requires the hybrid backend running on localhost:5002:
        opendataloader-pdf-hybrid --port 5002 --enrich-picture-description --enrich-formula

    Batching all PDFs in a single convert() call — each invocation spawns a
    JVM process, so repeated calls are slow (per OpenDataLoader docs).

    Returns {doc_id: {"captions": int, "descriptions": int}} for the exit-gate
    check (P0.2).
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    output_dir = Path(output_dir) if output_dir is not None else LOADED_DATA_DIR

    pdf_paths = _collect_pdf_paths(data_dir)
    logger.info("Running OpenDataLoader hybrid on %d PDFs → %s", len(pdf_paths), output_dir)

    opendataloader_pdf.convert(
        input_path=pdf_paths,
        output_dir=str(output_dir),
        format="json",
        image_output="external",
        hybrid="docling-fast",
        hybrid_mode="full",
        hybrid_url="http://localhost:5002",
    )

    counts: dict[str, dict[str, int]] = {}
    for pdf_path_str in pdf_paths:
        pdf_path = Path(pdf_path_str)
        doc_id = pdf_path.stem
        json_path = output_dir / f"{doc_id}.json"
        if json_path.exists():
            n_captions = _count_captions(json_path)
            n_descriptions = _count_descriptions(json_path)
            counts[doc_id] = {"captions": n_captions, "descriptions": n_descriptions}
            logger.info("  %s: %d captions, %d ai_descriptions", doc_id, n_captions, n_descriptions)
        else:
            logger.warning("  %s: output JSON not found at %s", doc_id, json_path)
            counts[doc_id] = {"captions": -1, "descriptions": -1}

    total_captions = sum(v["captions"] for v in counts.values() if v["captions"] > 0)
    total_descriptions = sum(v["descriptions"] for v in counts.values() if v["descriptions"] > 0)
    logger.info("Totals across %d docs: %d captions, %d ai_descriptions", len(pdf_paths), total_captions, total_descriptions)

    return counts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    counts = run_opendataloader()
    print("\nCaption + ai_description counts per document:")
    for doc_id, v in sorted(counts.items()):
        cap_flag = " ⚠ NEAR-ZERO" if v["captions"] <= 1 else ""
        desc_flag = " ⚠ NEAR-ZERO" if v["descriptions"] <= 1 else ""
        print(f"  {doc_id}: captions={v['captions']}{cap_flag}, ai_descriptions={v['descriptions']}{desc_flag}")
