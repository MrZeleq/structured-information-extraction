"""Tests for RelationConstraint, ExtractionSchema, and RelationFilter."""

from structured_ie.postprocessing.relation_filter import RelationFilter
from structured_ie.schemas.constraints import (
    DEFAULT_BUSINESS_SCHEMA,
    ExtractionSchema,
    RelationConstraint,
)
from structured_ie.schemas.models import Entity, Relation


def test_relation_constraint_validation() -> None:
    constraint = RelationConstraint(
        relation_name="has NIP",
        allowed_source_types={"company"},
        allowed_target_types={"NIP"},
        functional_target=True,
    )

    assert constraint.is_valid("company", "NIP") is True
    assert constraint.is_valid("NIP", "company") is False
    assert constraint.is_valid("person", "NIP") is False
    assert constraint.is_valid("company", "REGON") is False


def test_custom_extraction_schema() -> None:
    schema = ExtractionSchema()
    schema.add_constraint(
        RelationConstraint(
            relation_name="employs",
            allowed_source_types={"company"},
            allowed_target_types={"person"},
        )
    )

    assert schema.is_valid_pair("employs", "company", "person") is True
    assert schema.is_valid_pair("employs", "person", "company") is False
    # Unconstrained relations pass by default
    assert schema.is_valid_pair("unregistered_relation", "any", "other") is True


def test_relation_filter_rejects_schema_violations() -> None:
    entities = [
        Entity(text="Alpha S.A.", label="company", score=0.9, start=0, end=10),
        Entity(text="1234567890", label="NIP", score=0.9, start=15, end=25),
        Entity(text="987654321", label="REGON", score=0.9, start=30, end=39),
    ]

    candidate_relations = [
        Relation(source="Alpha S.A.", relation="has NIP", target="1234567890", score=0.8),
        # Inverted relation: NIP -> company (invalid)
        Relation(source="1234567890", relation="has NIP", target="Alpha S.A.", score=0.7),
        # Cross-identifier relation: REGON -> NIP (invalid)
        Relation(source="987654321", relation="has NIP", target="1234567890", score=0.6),
    ]

    rel_filter = RelationFilter(schema=DEFAULT_BUSINESS_SCHEMA)
    filtered = rel_filter.filter(relations=candidate_relations, entities=entities)

    assert len(filtered) == 1
    assert filtered[0].source == "Alpha S.A."
    assert filtered[0].relation == "has NIP"
    assert filtered[0].target == "1234567890"


def test_relation_filter_functional_target_deduplication() -> None:
    entities = [
        Entity(text="Alpha S.A.", label="company", score=0.95, start=0, end=10),
        Entity(text="Beta Sp. z o.o.", label="company", score=0.90, start=20, end=35),
        Entity(text="5251234567", label="NIP", score=0.92, start=40, end=50),
    ]

    # Two companies competing to claim the same NIP
    candidate_relations = [
        Relation(source="Alpha S.A.", relation="has NIP", target="5251234567", score=0.85),
        Relation(source="Beta Sp. z o.o.", relation="has NIP", target="5251234567", score=0.45),
    ]

    rel_filter = RelationFilter(schema=DEFAULT_BUSINESS_SCHEMA)
    filtered = rel_filter.filter(relations=candidate_relations, entities=entities)

    # Only the higher-scoring association (Alpha S.A.) should be retained
    assert len(filtered) == 1
    assert filtered[0].source == "Alpha S.A."
    assert filtered[0].score == 0.85
