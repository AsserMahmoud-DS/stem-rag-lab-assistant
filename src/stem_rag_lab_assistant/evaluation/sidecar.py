"""Sidecar — runs_sidecar.jsonl recording retrieved_chunks + attached_image_ids per (question, method). (P9.5)"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SIDECAR_PATH = Path(__file__).resolve().parents[3] / "runs" / "runs_sidecar.jsonl"


def log_cell(
    question_id: str,
    method: str,
    retrieved_chunk_ids: list[str],
    attached_image_ids: list[str],
    answer: str,
    latency_ms: float,
    sidecar_path: Path | None = None,
) -> None:
    """Append one JSON line to the sidecar log for a single (question, method) cell.

    Args:
        question_id: e.g. "Q001"
        method: e.g. "naive", "hybrid", "hybrid_graph"
        retrieved_chunk_ids: list of chunk_id strings retrieved
        attached_image_ids: list of image_id strings attached
        answer: the generated answer text
        latency_ms: total end-to-end latency in milliseconds
        sidecar_path: override path
    """
    p = sidecar_path or _SIDECAR_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "question_id": question_id,
        "method": method,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "attached_image_ids": attached_image_ids,
        "answer_preview": (answer or "")[:200],
        "latency_ms": round(latency_ms, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.debug("Sidecar: %s / %s logged", question_id, method)
