"""Tests for EntityPostprocessor normalization, span adjustments, and disambiguation."""

from structured_ie.postprocessing.entity_cleaner import EntityPostprocessor
from structured_ie.schemas.models import Entity


def test_clean_identifier_trims_prefix_and_adjusts_span() -> None:
    text = "Firma posiada NIP: 5251234567 oraz numer REGON 140987654."
    raw_entities = [
        Entity(text="NIP: 5251234567", label="NIP", score=0.85, start=14, end=29),
        Entity(text="REGON 140987654", label="REGON", score=0.91, start=41, end=56),
    ]

    postprocessor = EntityPostprocessor()
    cleaned = postprocessor.postprocess(entities=raw_entities, text=text)

    assert len(cleaned) == 2
    nip_ent, regon_ent = cleaned[0], cleaned[1]

    assert nip_ent.text == "5251234567"
    assert nip_ent.label == "NIP"
    assert text[nip_ent.start : nip_ent.end] == "5251234567"

    assert regon_ent.text == "140987654"
    assert regon_ent.label == "REGON"
    assert text[regon_ent.start : regon_ent.end] == "140987654"


def test_clean_identifier_discards_trigger_words_without_digits() -> None:
    text = "Numer NIP oraz numer REGON zostały zweryfikowane."
    raw_entities = [
        Entity(text="NIP", label="NIP", score=0.45, start=6, end=9),
        Entity(text="REGON", label="REGON", score=0.92, start=21, end=26),
    ]

    postprocessor = EntityPostprocessor()
    cleaned = postprocessor.postprocess(entities=raw_entities, text=text)

    # Both trigger words should be dropped because they lack digits
    assert len(cleaned) == 0


def test_disambiguate_tax_identifier_lengths() -> None:
    text = "W rejestrze: 7891234567 oraz 360123456."
    # 7891234567 is 10 digits (NIP), but mistakenly labeled REGON by zero-shot model
    # 360123456 is 9 digits (REGON), but mistakenly labeled NIP
    raw_entities = [
        Entity(text="7891234567", label="REGON", score=0.72, start=13, end=23),
        Entity(text="360123456", label="NIP", score=0.68, start=29, end=38),
    ]

    postprocessor = EntityPostprocessor(disambiguate_lengths=True)
    cleaned = postprocessor.postprocess(entities=raw_entities, text=text)

    assert len(cleaned) == 2
    assert cleaned[0].text == "7891234567"
    assert cleaned[0].label == "NIP"  # Corrected to NIP
    assert cleaned[1].text == "360123456"
    assert cleaned[1].label == "REGON"  # Corrected to REGON


def test_clean_company_descriptor_prefixes() -> None:
    text = "Spółka Apex IT Services Sp. z o.o. rozpoczęła działalność."
    raw_entities = [
        Entity(
            text="Spółka Apex IT Services Sp. z o.o.",
            label="company",
            score=0.90,
            start=0,
            end=34,
        ),
    ]

    postprocessor = EntityPostprocessor(clean_company_descriptors=True)
    cleaned = postprocessor.postprocess(entities=raw_entities, text=text)

    assert len(cleaned) == 1
    assert cleaned[0].text == "Apex IT Services Sp. z o.o."
    assert text[cleaned[0].start : cleaned[0].end] == "Apex IT Services Sp. z o.o."
