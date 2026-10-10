"""Tests for tokenization, offset tracking, and span alignment."""

from structured_ie.pipeline.alignment import (
    align_entities_to_tokens,
    convert_glirel_prediction_to_relation,
    tokenize_with_offsets,
)
from structured_ie.schemas.models import Entity


def test_tokenize_with_offsets() -> None:
    text = "Alfa Logistyka S.A. ma NIP 5252525252."
    spans = tokenize_with_offsets(text)

    # Reconstruct text through token surfaces and spaces
    for span in spans:
        assert text[span.start : span.end] == span.text

    token_texts = [s.text for s in spans]
    assert "Alfa" in token_texts
    assert "Logistyka" in token_texts
    assert "S" in token_texts or "S.A." in token_texts or "A" in token_texts
    assert "5252525252" in token_texts


def test_align_entities_to_tokens_multi_word() -> None:
    text = "Firma Alfa Logistyka zatrudnia Jana Kowalskiego."
    spans = tokenize_with_offsets(text)
    # text tokens:
    # 0: Firma, 1: Alfa, 2: Logistyka, 3: zatrudnia, 4: Jana, 5: Kowalskiego, 6: .

    start1 = text.index("Alfa Logistyka")
    ent1 = Entity(
        text="Alfa Logistyka",
        label="company",
        score=0.95,
        start=start1,
        end=start1 + len("Alfa Logistyka"),
    )
    start2 = text.index("Jana Kowalskiego")
    ent2 = Entity(
        text="Jana Kowalskiego",
        label="person",
        score=0.98,
        start=start2,
        end=start2 + len("Jana Kowalskiego"),
    )

    glirel_ner = align_entities_to_tokens([ent1, ent2], spans)

    assert len(glirel_ner) == 2
    # Alfa Logistyka corresponds to tokens 1 and 2
    assert glirel_ner[0][0] == 1
    assert glirel_ner[0][1] == 2
    assert glirel_ner[0][2] == "company"
    assert glirel_ner[0][3] == "Alfa Logistyka"

    # Jana Kowalskiego corresponds to tokens 4 and 5
    assert glirel_ner[1][0] == 4
    assert glirel_ner[1][1] == 5
    assert glirel_ner[1][2] == "person"
    assert glirel_ner[1][3] == "Jana Kowalskiego"


def test_convert_glirel_prediction_to_relation() -> None:
    text = "Alfa Logistyka zatrudnia Jana Kowalskiego."
    spans = tokenize_with_offsets(text)
    tokens = [s.text for s in spans]

    ent_company = Entity(
        text="Alfa Logistyka",
        label="company",
        score=0.95,
        start=0,
        end=14,
    )
    ent_person = Entity(
        text="Jana Kowalskiego",
        label="person",
        score=0.98,
        start=25,
        end=41,
    )
    entities = [ent_company, ent_person]
    glirel_ner = align_entities_to_tokens(entities, spans)

    # Mock raw prediction as returned by GLiREL:
    # head: Jana Kowalskiego (tokens [3, 4] -> slice [3, 5])
    # tail: Alfa Logistyka (tokens [0, 1] -> slice [0, 2])
    raw_glirel_pred = {
        "head_pos": [glirel_ner[1][0], glirel_ner[1][1] + 1],
        "tail_pos": [glirel_ner[0][0], glirel_ner[0][1] + 1],
        "head_text": ["Jana", "Kowalskiego"],
        "tail_text": ["Alfa", "Logistyka"],
        "label": "works for",
        "score": 0.9412,
    }

    relation = convert_glirel_prediction_to_relation(
        prediction=raw_glirel_pred,
        entities=entities,
        glirel_ner=glirel_ner,
        tokens=tokens,
    )

    assert relation.source == "Jana Kowalskiego"
    assert relation.relation == "works for"
    assert relation.target == "Alfa Logistyka"
    assert relation.score == 0.9412
    assert relation.source_span == (25, 41)
    assert relation.target_span == (0, 14)
