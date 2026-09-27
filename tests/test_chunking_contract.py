from pathlib import Path

import pytest

from stem_rag_lab_assistant.corpus.chunking import _assert_unique_chunk_ids, _chunk_single_doc


def test_chunk_single_doc_contract_and_metadata_shape() -> None:
    doc = {
        "file name": "sample_ece.pdf",
        "number of pages": 2,
        "kids": [
            {
                "type": "heading",
                "content": "Ohm's Law",
                "page number": 1,
                "bounding box": [10, 10, 100, 20],
            },
            {
                "type": "paragraph",
                "content": "Voltage equals current times resistance.",
                "page number": "2",
                "bounding box": [10, 30, 100, 60],
            },
            {
                "type": "image",
                "page number": 2,
            },
            {
                "type": "paragraph",
                "content": "   ",
                "page number": 2,
            },
        ],
    }

    chunks = _chunk_single_doc(
        doc=doc,
        doc_path=Path("loaded_data/sample_ece.json"),
        doc_id="sample_ece",
        pdf_path=Path("data/sample_ece.pdf"),
    )

    assert len(chunks) == 2

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(set(chunk_ids)) == len(chunk_ids)

    for i, chunk in enumerate(chunks):
        assert chunk["text"]
        assert isinstance(chunk["chunk_id"], str)
        assert isinstance(chunk["doc_id"], str)
        assert isinstance(chunk["metadata"]["chunk_order_index"], int)
        assert chunk["doc_id"] == "sample_ece"
        assert chunk["metadata"]["chunk_order_index"] == i
        assert "page_number" in chunk["metadata"]
        assert "doc_n_pages" in chunk["metadata"]
        assert "element_type" in chunk["metadata"]

    assert chunks[0]["metadata"]["page_number"] == 1
    assert chunks[1]["metadata"]["page_number"] == 2
    assert chunks[0]["metadata"]["doc_n_pages"] == 2
    assert chunks[1]["metadata"]["doc_n_pages"] == 2


def test_image_description_not_merged_into_chunk_text() -> None:
    doc = {
        "file name": "figs.pdf",
        "number of pages": 1,
        "kids": [
            {
                "type": "paragraph",
                "content": "The bridge balances when the ratio arms are equal.",
                "page number": 1,
                "bounding box": [10, 10, 100, 30],
            },
            {
                "type": "image",
                "description": "A schematic of a Wheatstone bridge with four resistors.",
                "page number": 1,
                "bounding box": [10, 40, 100, 90],
            },
        ],
    }

    chunks = _chunk_single_doc(
        doc=doc,
        doc_path=Path("loaded_data/figs.json"),
        doc_id="figs",
        pdf_path=Path("dataset/figs.pdf"),
    )

    combined = " ".join(c["text"] for c in chunks)
    assert "schematic of a Wheatstone bridge" not in combined

    linked = [c for c in chunks if c["metadata"]["linked_image_ids"]]
    assert linked, "image should stay linked to its positional host chunk"
    assert "figs::img_00000" in linked[0]["metadata"]["linked_image_ids"]


def test_chunk_id_uniqueness_guard_raises_on_duplicates() -> None:
    duplicate_chunks = [
        {
            "chunk_id": "docA::ch_00000",
            "text": "x",
            "doc_id": "docA",
            "metadata": {
                "chunk_order_index": 0,
                "file_path": "docA.pdf",
                "bbox": None,
                "page_number": 1,
                "doc_n_pages": 1,
                "element_type": "paragraph",
            },
        },
        {
            "chunk_id": "docA::ch_00000",
            "text": "y",
            "doc_id": "docA",
            "metadata": {
                "chunk_order_index": 1,
                "file_path": "docA.pdf",
                "bbox": None,
                "page_number": 1,
                "doc_n_pages": 1,
                "element_type": "paragraph",
            },
        },
    ]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        _assert_unique_chunk_ids(duplicate_chunks)
