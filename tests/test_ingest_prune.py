"""Tests for incremental ingest: pruning removed docs + unchanged-doc skip."""

from __future__ import annotations

import json

from stem_rag_lab_assistant.corpus import ingest


def _odl_doc(name: str) -> dict:
    return {
        "file name": name,
        "number of pages": 1,
        "kids": [
            {
                "type": "heading",
                "content": "Title",
                "page number": 1,
                "bounding box": [0, 0, 10, 10],
            },
            {
                "type": "paragraph",
                "content": "Some technical content about bridge circuits.",
                "page number": 1,
                "bounding box": [0, 10, 10, 20],
            },
        ],
    }


def _seed_pdfs_and_outputs(tmp_path, names: list[str]):
    data_dir = tmp_path / "data"
    loaded_dir = tmp_path / "loaded"
    data_dir.mkdir()
    loaded_dir.mkdir()
    for name in names:
        (data_dir / f"{name}.pdf").write_bytes(f"%PDF {name}".encode())
        (loaded_dir / f"{name}.json").write_text(
            json.dumps(_odl_doc(f"{name}.pdf")), encoding="utf-8"
        )
    return data_dir, loaded_dir


def test_run_ingest_prunes_removed_docs(tmp_path, monkeypatch) -> None:
    data_dir, loaded_dir = _seed_pdfs_and_outputs(tmp_path, ["A", "B"])
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(ingest, "CHUNKS_JSON_PATH", chunks_path)
    chunks_path.write_text(
        json.dumps({"version": 1, "docs": {"C": {"doc_id": "C", "n_chunks": 0, "chunks": []}}}),
        encoding="utf-8",
    )

    ingest.run_ingest(data_dir=data_dir, loaded_dir=loaded_dir)

    saved = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert set(saved["docs"]) == {"A", "B"}
    assert saved["docs"]["A"]["n_chunks"] >= 1


def test_run_ingest_skips_unchanged_doc(tmp_path, monkeypatch) -> None:
    data_dir, loaded_dir = _seed_pdfs_and_outputs(tmp_path, ["A"])
    chunks_path = tmp_path / "chunks.json"
    monkeypatch.setattr(ingest, "CHUNKS_JSON_PATH", chunks_path)

    ingest.run_ingest(data_dir=data_dir, loaded_dir=loaded_dir)

    # Tamper a chunk; a re-chunk would overwrite it, a content-hash skip will not.
    seeded = json.loads(chunks_path.read_text(encoding="utf-8"))
    seeded["docs"]["A"]["chunks"][0]["text"] = "SENTINEL"
    chunks_path.write_text(json.dumps(seeded), encoding="utf-8")

    ingest.run_ingest(data_dir=data_dir, loaded_dir=loaded_dir)

    saved = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert saved["docs"]["A"]["chunks"][0]["text"] == "SENTINEL"
