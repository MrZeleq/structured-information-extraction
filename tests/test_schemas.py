"""Tests for normalized entity, relation, and result schemas."""

import json

import pytest
from pydantic import ValidationError

from structured_ie.schemas.models import (
    Entity,
    ExtractionResult,
    LabeledSample,
    Relation,
)


def test_entity_schema_valid() -> None:
    entity = Entity(
        text="XYZ Broker S.A.",
        label="company",
        score=0.97,
        start=0,
        end=15,
    )
    assert entity.text == "XYZ Broker S.A."
    assert entity.label == "company"
    assert entity.score == 0.97
    assert entity.start == 0
    assert entity.end == 15

    d = entity.to_dict()
    assert d == {
        "text": "XYZ Broker S.A.",
        "label": "company",
        "score": 0.97,
        "start": 0,
        "end": 15,
    }


def test_entity_schema_validation_error() -> None:
    # Invalid score (> 1.0)
    with pytest.raises(ValidationError):
        Entity(text="Foo", label="bar", score=1.5, start=0, end=3)

    # Negative offset
    with pytest.raises(ValidationError):
        Entity(text="Foo", label="bar", score=0.9, start=-1, end=3)


def test_relation_schema_valid() -> None:
    rel = Relation(
        source="XYZ Broker S.A.",
        relation="has NIP",
        target="9876543210",
        score=0.9321,
        source_span=(0, 15),
        target_span=(20, 30),
    )
    assert rel.source == "XYZ Broker S.A."
    assert rel.relation == "has NIP"
    assert rel.target == "9876543210"
    assert rel.score == 0.9321

    d = rel.to_dict()
    assert d["source"] == "XYZ Broker S.A."
    assert d["relation"] == "has NIP"
    assert d["target"] == "9876543210"
    assert d["score"] == 0.9321
    assert d["source_span"] == [0, 15]
    assert d["target_span"] == [20, 30]


def test_extraction_result_json_serialization() -> None:
    ent1 = Entity(text="Acme Corp", label="company", score=0.95, start=0, end=9)
    ent2 = Entity(text="1234567890", label="NIP", score=0.99, start=15, end=25)
    rel = Relation(source="Acme Corp", relation="has NIP", target="1234567890", score=0.92)

    res = ExtractionResult(
        text="Acme Corp NIP: 1234567890",
        entities=[ent1, ent2],
        relations=[rel],
    )

    d = res.to_dict()
    assert len(d["entities"]) == 2
    assert len(d["relations"]) == 1
    assert d["entities"][0]["text"] == "Acme Corp"
    assert d["relations"][0]["relation"] == "has NIP"

    # Ensure JSON serializable
    dumped = json.dumps(d)
    assert "Acme Corp" in dumped


def test_labeled_sample_schema() -> None:
    sample = LabeledSample(
        text="Sample text",
        expected_entities=[Entity(text="Sample", label="example", score=1.0, start=0, end=6)],
        expected_relations=[],
    )
    assert sample.text == "Sample text"
    assert len(sample.expected_entities) == 1
