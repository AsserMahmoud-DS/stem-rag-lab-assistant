"""Corpus loading — run OpenDataLoader hybrid on data/*.pdf → loaded_data/*.json. (P0.2)

Requires the hybrid backend to be running:
    opendataloader-pdf-hybrid --port 5002 --enrich-picture-description --enrich-formula
"""

import hashlib
import json
import logging
import os
from pathlib import Path

import opendataloader_pdf

from stem_rag_lab_assistant.config import DATA_DIR, LOADED_DATA_DIR

logger = logging.getLogger(__name__)

# Local cache mapping doc_id -> source content hash, so --only-missing can skip
# PDFs that were already parsed without re-running the (slow) ODL/JVM step.
_MANIFEST_NAME = "_opendataloader_manifest.json"


def _file_sha256(path: Path) -> str:
    """Content hash of a file (same scheme as corpus.ingest)."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return f"sha256:{sha.hexdigest()}"


def _load_manifest(output_dir: Path) -> dict:
    path = output_dir / _MANIFEST_NAME
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": 1, "docs": {}}


def _save_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / _MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


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
    only_missing: bool = False,
) -> dict[str, dict[str, int]]:
    """Convert PDFs in data_dir with OpenDataLoader hybrid mode.

    Requires the hybrid backend running on localhost:5002:
        opendataloader-pdf-hybrid --port 5002 --enrich-picture-description --enrich-formula

    Batching PDFs in a single convert() call — each invocation spawns a JVM
    process, so repeated calls are slow (per OpenDataLoader docs).

    When ``only_missing`` is True, PDFs whose output JSON already exists **and**
    whose content hash matches the stored manifest are skipped, so adding a few
    PDFs does not force a full re-parse. The manifest
    (``_opendataloader_manifest.json``) records each doc's source content hash.

    Returns {doc_id: {"captions": int, "descriptions": int}} for the exit-gate
    computed across all PDFs (skipped and converted).
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    output_dir = Path(output_dir) if output_dir is not None else LOADED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = _collect_pdf_paths(data_dir)
    manifest = _load_manifest(output_dir)
    manifest_docs = manifest.setdefault("docs", {})

    to_convert: list[str] = []
    for pdf_path_str in pdf_paths:
        pdf_path = Path(pdf_path_str)
        if only_missing:
            doc_id = pdf_path.stem
            json_path = output_dir / f"{doc_id}.json"
            entry = manifest_docs.get(doc_id, {})
            if json_path.exists() and entry.get("content_hash") == _file_sha256(pdf_path):
                logger.info("  %s: unchanged, skipping (--only-missing)", doc_id)
                continue
        to_convert.append(pdf_path_str)

    if to_convert:
        logger.info(
            "Running OpenDataLoader hybrid on %d PDFs (%d skipped) → %s",
            len(to_convert), len(pdf_paths) - len(to_convert), output_dir,
        )
        opendataloader_pdf.convert(
            input_path=to_convert,
            output_dir=str(output_dir),
            format="json",
            image_output="external",
            hybrid="docling-fast",
            hybrid_mode="full",
            hybrid_url="http://localhost:5002",
        )
    else:
        logger.info("OpenDataLoader: all %d PDFs up to date, nothing to convert", len(pdf_paths))

    for pdf_path_str in to_convert:
        pdf_path = Path(pdf_path_str)
        manifest_docs[pdf_path.stem] = {
            "content_hash": _file_sha256(pdf_path),
            "source_mtime": format(os.path.getmtime(pdf_path), ".0f"),
        }
    _save_manifest(output_dir, manifest)

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
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Parse dataset PDFs with OpenDataLoader hybrid mode.")
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="skip PDFs whose output JSON already exists and whose PDF is unchanged",
    )
    counts = run_opendataloader(only_missing=parser.parse_args().only_missing)
    print("\nCaption + ai_description counts per document:")
    for doc_id, v in sorted(counts.items()):
        cap_flag = " ⚠ NEAR-ZERO" if v["captions"] <= 1 else ""
        desc_flag = " ⚠ NEAR-ZERO" if v["descriptions"] <= 1 else ""
        print(f"  {doc_id}: captions={v['captions']}{cap_flag}, ai_descriptions={v['descriptions']}{desc_flag}")
