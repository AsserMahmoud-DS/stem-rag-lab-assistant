"""Tests for images.json construction (manifest sidecar must be ignored)."""

from __future__ import annotations

import json

from stem_rag_lab_assistant.corpus import images


def _odl_doc_with_image(doc_id: str) -> dict:
    return {
        "file name": f"{doc_id}.pdf",
        "number of pages": 1,
        "kids": [
            {
                "type": "image",
                "page number": 1,
                "bounding box": [0, 0, 10, 10],
                "source": f"{doc_id}_images/image_0.png",
                "format": "png",
                "alt": "a schematic",
            }
        ],
    }


def _seed(loaded_dir, doc_ids: list[str]) -> None:
    loaded_dir.mkdir()
    for doc_id in doc_ids:
        (loaded_dir / f"{doc_id}.json").write_text(
            json.dumps(_odl_doc_with_image(doc_id)), encoding="utf-8"
        )
    # Sidecar that must NOT be treated as a document.
    (loaded_dir / "_opendataloader_manifest.json").write_text(
        json.dumps({"version": 1, "docs": {"A": {"content_hash": "sha256:x"}}}),
        encoding="utf-8",
    )


def test_run_images_ignores_underscore_sidecar(tmp_path, monkeypatch) -> None:
    loaded_dir = tmp_path / "loaded"
    _seed(loaded_dir, ["A"])

    chunks_path = tmp_path / "chunks.json"
    chunks_path.write_text(json.dumps({"version": 1, "docs": {}}), encoding="utf-8")
    monkeypatch.setattr(images, "CHUNKS_JSON_PATH", chunks_path)
    monkeypatch.setattr(images, "IMAGES_JSON_PATH", tmp_path / "images.json")

    data = images.run_images(loaded_dir=loaded_dir)

    assert len(data["images"]) == 1
    assert {img["doc_id"] for img in data["images"]} == {"A"}
    assert data["images"][0]["image_id"] == "A::img_00000"
