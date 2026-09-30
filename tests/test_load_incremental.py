"""Tests for incremental OpenDataLoader parsing (corpus/load.py --only-missing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stem_rag_lab_assistant.corpus import load


def _write_pdf(path: Path, content: bytes = b"%PDF-1.4 fake") -> None:
    path.write_bytes(content)


def _output_doc() -> dict:
    return {"kids": [{"type": "paragraph", "content": "hello"}]}


@pytest.fixture
def fake_convert(monkeypatch):
    """Replace ODL convert() with a recorder that writes minimal output JSONs."""
    calls: list[list[str]] = []

    def _convert(input_path, output_dir, **kwargs):
        calls.append([Path(p).stem for p in input_path])
        for p in input_path:
            (Path(output_dir) / f"{Path(p).stem}.json").write_text(
                json.dumps(_output_doc()), encoding="utf-8"
            )

    monkeypatch.setattr(load.opendataloader_pdf, "convert", _convert)
    return calls


def test_first_pass_converts_all(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    _write_pdf(data_dir / "B.pdf")

    counts = load.run_opendataloader(data_dir, out_dir)

    assert fake_convert == [["A", "B"]]
    assert set(counts) == {"A", "B"}


def test_only_missing_skips_unchanged(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    _write_pdf(data_dir / "B.pdf")

    load.run_opendataloader(data_dir, out_dir)
    counts = load.run_opendataloader(data_dir, out_dir, only_missing=True)

    assert fake_convert == [["A", "B"]]  # no second convert call
    assert set(counts) == {"A", "B"}


def test_only_missing_converts_new_pdf(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    load.run_opendataloader(data_dir, out_dir)

    _write_pdf(data_dir / "C.pdf")
    load.run_opendataloader(data_dir, out_dir, only_missing=True)

    assert fake_convert[-1] == ["C"]


def test_only_missing_reconverts_changed_pdf(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    load.run_opendataloader(data_dir, out_dir)

    _write_pdf(data_dir / "A.pdf", b"%PDF-1.4 changed")
    load.run_opendataloader(data_dir, out_dir, only_missing=True)

    assert fake_convert[-1] == ["A"]


def test_only_missing_reconverts_when_output_deleted(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    load.run_opendataloader(data_dir, out_dir)

    (out_dir / "A.json").unlink()
    load.run_opendataloader(data_dir, out_dir, only_missing=True)

    assert fake_convert[-1] == ["A"]


def test_default_reconverts_everything(tmp_path, fake_convert) -> None:
    data_dir, out_dir = tmp_path / "data", tmp_path / "loaded"
    data_dir.mkdir()
    out_dir.mkdir()
    _write_pdf(data_dir / "A.pdf")
    _write_pdf(data_dir / "B.pdf")

    load.run_opendataloader(data_dir, out_dir)
    load.run_opendataloader(data_dir, out_dir)

    assert fake_convert == [["A", "B"], ["A", "B"]]
