"""Chunking — walk ODL element list, produce chunks per test_chunking_contract.py. (P0.3)"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stem_rag_lab_assistant.config import get_config, to_relative_path

_ELEM_TYPES_TEXT = frozenset({"paragraph", "heading", "caption", "formula"})
_ELEM_TYPES_IMAGE = frozenset({"image"})
_ELEM_TYPES_COMPOSITE = frozenset({"table", "list"})
# struct elements that wrap kids without their own text:
_ELEM_TYPES_WRAPPER = frozenset({"text block", "header", "footer", "table cell"})


def _assert_unique_chunk_ids(chunks: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for chunk in chunks:
        cid = chunk["chunk_id"]
        if cid in seen:
            raise ValueError(f"Duplicate chunk_id: {cid}")
        seen.add(cid)


def _extract_cell_text(cell: dict[str, Any]) -> str:
    """Recursively extract text from a table cell's kids."""
    parts: list[str] = []
    stack = list(cell.get("kids", []))
    while stack:
        el = stack.pop()
        if not isinstance(el, dict):
            continue
        if "content" in el and el.get("content", "").strip():
            parts.append(el["content"].strip())
        for key in ("kids", "list items", "rows", "cells"):
            children = el.get(key)
            if isinstance(children, list):
                stack.extend(children)
    return " ".join(parts)


def _extract_table_text(rows: list[dict[str, Any]]) -> str:
    row_texts: list[str] = []
    for row in rows:
        cells = row.get("cells", [])
        cell_texts = [_extract_cell_text(c) for c in cells]
        row_texts.append(" | ".join(c for c in cell_texts if c))
    return "\n".join(t for t in row_texts if t)


def _extract_list_text(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        content = item.get("content", "").strip()
        lines.append(f"- {content}" if content else "")
        lines.append(_extract_nested_text(item.get("kids", [])))
    return "\n".join(l for l in lines if l)


def _extract_nested_text(kids: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    stack = list(kids)
    while stack:
        el = stack.pop()
        if not isinstance(el, dict):
            continue
        if "content" in el and el.get("content", "").strip():
            parts.append(el["content"].strip())
        for key in ("kids", "list items", "rows", "cells"):
            children = el.get(key)
            if isinstance(children, list):
                stack.extend(children)
    return " ".join(parts)


def _walk_elements(
    kids: list[dict[str, Any]],
    result: list[tuple[str, str, int | None, list[float] | None, dict[str, Any]]],
    doc_id: str,
    img_counter: dict[str, int] | None = None,
) -> None:
    """In-order walk of ODL element tree → (type, text, page, bbox, extra) tuples.

    extra dict may carry:
      - image_id: str          — stable ID, always set for image elements
      - linked_image_id: str   — same as image_id (used by chunker for linking)
      - ai_description: str    — from image element's "description" or "alt"
    """
    if img_counter is None:
        img_counter = {"count": 0}

    for element in kids:
        if not isinstance(element, dict):
            continue

        elem_type = element.get("type", "")
        text = element.get("content", "")

        page_raw = element.get("page number")
        page = int(page_raw) if page_raw is not None else None
        bbox = element.get("bounding box")

        if elem_type in _ELEM_TYPES_IMAGE:
            extra: dict[str, Any] = {}
            img_id = f"{doc_id}::img_{img_counter['count']:05d}"
            img_counter["count"] += 1
            extra["image_id"] = img_id
            extra["linked_image_id"] = img_id
            desc = element.get("description") or element.get("alt")
            if desc:
                extra["ai_description"] = desc
            result.append((elem_type, desc or "", page, bbox, extra))
            continue

        if elem_type in _ELEM_TYPES_TEXT:
            extra: dict[str, Any] = {}
            if elem_type == "caption":
                linked_id = element.get("linked content id")
                if linked_id is not None:
                    extra["linked_image_id"] = f"{doc_id}::img_{int(linked_id):05d}"
            result.append((elem_type, text, page, bbox, extra))
            continue

        if elem_type in _ELEM_TYPES_COMPOSITE:
            if elem_type == "table":
                table_text = _extract_table_text(element.get("rows", []))
            else:
                table_text = _extract_list_text(element.get("list items", []))
            result.append((elem_type, table_text, page, bbox, {}))
            continue

        if elem_type in _ELEM_TYPES_WRAPPER:
            _walk_elements(element.get("kids", []), result, doc_id)
            continue

        if elem_type == "list item":
            content = element.get("content", "").strip()
            if element.get("kids"):
                nested = _extract_nested_text(element["kids"])
                if content and nested:
                    text = f"- {content}\n{nested}"
                elif nested:
                    text = nested
                else:
                    text = f"- {content}" if content else ""
                result.append((elem_type, text, page, bbox, {}))
            else:
                result.append((elem_type, content, page, bbox, {}))
            continue


def _make_chunk(
    *,
    doc_id: str,
    text: str,
    order: int,
    page: int | None,
    n_pages: int,
    elem_type: str,
    bbox: list[float] | None,
    file_path: str,
    image_ids: list[str],
) -> dict[str, Any]:
    return {
        "chunk_id": f"{doc_id}::ch_{order:05d}",
        "doc_id": doc_id,
        "text": text,
        "embedding": [],
        "metadata": {
            "chunk_order_index": order,
            "page_number": page or 0,
            "doc_n_pages": n_pages,
            "element_type": elem_type,
            "file_path": file_path,
            "bbox": bbox,
            "linked_image_ids": list(dict.fromkeys(image_ids)),  # dedup, preserve order
        },
    }


def _chunk_single_doc(
    doc: dict[str, Any],
    doc_path: Path,
    doc_id: str,
    pdf_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Convert one ODL JSON doc into a list of chunk dicts.

    Splits on page boundaries and on exceeding the chunk-size limit.  Caption text and image
    ai_description text are merged into the enclosing chunk; image linkage is
    recorded via linked_image_ids.  Image elements without description/alt are
    skipped (no text to contribute).

    Args:
        doc: ODL JSON document dict.
        doc_path: Path to the ODL JSON (used as fallback file_path).
        doc_id: Stable document ID (PDF stem, e.g. "Lab1").
        pdf_path: Path to the source PDF (used for metadata.file_path).
    """
    cfg = get_config()
    n_pages = doc.get("number of pages", 1)
    file_path = to_relative_path(pdf_path) if pdf_path else to_relative_path(doc_path)

    flat: list[tuple[str, str, int | None, list[float] | None, dict[str, Any]]] = []
    img_counter: dict[str, int] = {"count": 0}
    _walk_elements(doc.get("kids", []), flat, doc_id, img_counter)

    chunks: list[dict[str, Any]] = []
    text_buffer: list[str] = []
    current_page: int | None = None
    current_elem_type: str | None = None
    current_bbox: list[float] | None = None
    linked_image_ids: list[str] = []

    def _flush(override_text: str | None = None) -> None:
        nonlocal text_buffer, current_page, current_elem_type, current_bbox, linked_image_ids
        combined = override_text if override_text is not None else " ".join(text_buffer)
        if not combined.strip():
            text_buffer = []
            linked_image_ids = []
            return
        chunks.append(
            _make_chunk(
                doc_id=doc_id,
                text=combined,
                order=len(chunks),
                page=current_page,
                n_pages=n_pages,
                elem_type=current_elem_type or "paragraph",
                bbox=current_bbox,
                file_path=file_path,
                image_ids=linked_image_ids,
            )
        )
        text_buffer = []
        linked_image_ids = []

    for elem_type, text, page, bbox, extra in flat:
        if not text.strip():
            continue

        # flush on page boundary
        if current_page is not None and page != current_page and text_buffer:
            _flush()

        if not text_buffer:
            current_page = page
            current_elem_type = elem_type if elem_type != "image" else "paragraph"
            current_bbox = bbox

        if extra.get("linked_image_id"):
            linked_image_ids.append(extra["linked_image_id"])

        text_buffer.append(text)
        combined = " ".join(text_buffer)

        if len(combined) > cfg.chunk_size:
            if len(text_buffer) == 1:
                _flush(override_text=combined)
                continue
            pre_overflow = " ".join(text_buffer[:-1])
            if pre_overflow.strip():
                _flush(override_text=pre_overflow)
            overlap_text = ""
            if cfg.chunk_overlap > 0 and len(pre_overflow) > cfg.chunk_overlap:
                overlap_text = pre_overflow[-cfg.chunk_overlap:]
            text_buffer = [overlap_text] if overlap_text.strip() else []
            text_buffer.append(text)
            current_page = page
            current_elem_type = (
                elem_type if elem_type not in _ELEM_TYPES_IMAGE else current_elem_type or "paragraph"
            )
            current_bbox = bbox

    _flush()
    _assert_unique_chunk_ids(chunks)
    return chunks
