"""Tests for ExtractionPipeline orchestration with mocked extractors."""

from structured_ie.ner.base import BaseNERExtractor
from structured_ie.pipeline.extraction_pipeline import ExtractionPipeline
from structured_ie.relations.base import BaseRelationExtractor
from structured_ie.schemas.models import Entity, Relation


class DummyNER(BaseNERExtractor):
    def __init__(self, entities: list[Entity] | None = None) -> None:
        self.entities = entities or []

    def extract_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[Entity]:
        thresh = threshold if threshold is not None else 0.5
        return [e for e in self.entities if e.score >= thresh]


class DummyRE(BaseRelationExtractor):
    def __init__(self, relations: list[Relation] | None = None) -> None:
        self.relations = relations or []

    def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[Relation]:
        thresh = threshold if threshold is not None else 0.5
        return [r for r in self.relations if r.score >= thresh]


def test_pipeline_orchestration_success() -> None:
    text = "Firma XYZ zatrudnia Jana Kowalskiego."
    mock_entities = [
        Entity(text="XYZ", label="company", score=0.9, start=6, end=9),
        Entity(text="Jana Kowalskiego", label="person", score=0.95, start=20, end=36),
    ]
    mock_relations = [
        Relation(source="Jana Kowalskiego", relation="works for", target="XYZ", score=0.88),
    ]

    pipeline = ExtractionPipeline(
        ner_extractor=DummyNER(mock_entities),
        relation_extractor=DummyRE(mock_relations),
    )

    result = pipeline.extract(
        text=text,
        entity_labels=["company", "person"],
        relation_labels=["works for"],
        entity_threshold=0.5,
        relation_threshold=0.5,
    )

    assert result.text == text
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.entities[0].text == "XYZ"
    assert result.relations[0].relation == "works for"


def test_pipeline_skips_re_when_fewer_than_two_entities() -> None:
    text = "Tylko jedna firma Acme."
    mock_entities = [
        Entity(text="Acme", label="company", score=0.9, start=17, end=21),
    ]

    re_called = False

    class SpyRE(BaseRelationExtractor):
        def extract_relations(
            self,
            text: str,
            entities: list[Entity],
            labels: list[str],
            threshold: float = 0.5,
        ) -> list[Relation]:
            nonlocal re_called
            re_called = True
            return []

    pipeline = ExtractionPipeline(
        ner_extractor=DummyNER(mock_entities),
        relation_extractor=SpyRE(),
    )

    result = pipeline.extract(
        text=text,
        entity_labels=["company"],
        relation_labels=["works for"],
    )

    assert len(result.entities) == 1
    assert len(result.relations) == 0
    assert not re_called


def test_pipeline_with_postprocessing_and_filtering() -> None:
    text = "Firma Pol-Trans Sp. z o.o. posiada NIP: 5251234567."
    raw_entities = [
        Entity(text="Firma Pol-Trans Sp. z o.o.", label="company", score=0.9, start=0, end=26),
        Entity(text="NIP: 5251234567", label="NIP", score=0.88, start=35, end=50),
    ]
    raw_relations = [
        # Valid relation
        Relation(
            source="Pol-Trans Sp. z o.o.", relation="has NIP", target="5251234567", score=0.85
        ),
        # Inverted relation (should be pruned by relation_filter)
        Relation(
            source="5251234567", relation="has NIP", target="Pol-Trans Sp. z o.o.", score=0.75
        ),
    ]

    from structured_ie.postprocessing.entity_cleaner import EntityPostprocessor
    from structured_ie.postprocessing.relation_filter import RelationFilter
    from structured_ie.schemas.constraints import DEFAULT_BUSINESS_SCHEMA

    pipeline = ExtractionPipeline(
        ner_extractor=DummyNER(raw_entities),
        relation_extractor=DummyRE(raw_relations),
        postprocessor=EntityPostprocessor(),
        relation_filter=RelationFilter(DEFAULT_BUSINESS_SCHEMA),
    )

    res = pipeline.extract(
        text=text,
        entity_labels=["company", "NIP"],
        relation_labels=["has NIP"],
        apply_postprocessing=True,
    )

    # Entities should be cleaned: company descriptor stripped, NIP prefix stripped
    assert len(res.entities) == 2
    assert res.entities[0].text == "Pol-Trans Sp. z o.o."
    assert res.entities[1].text == "5251234567"

    # Only valid relation should remain
    assert len(res.relations) == 1
    assert res.relations[0].source == "Pol-Trans Sp. z o.o."
    assert res.relations[0].target == "5251234567"
