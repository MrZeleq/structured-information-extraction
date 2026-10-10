"""Tests for evaluation dataset loading, integrity, and pipeline benchmarking."""

from pathlib import Path

from structured_ie.evaluation.dataset import load_dataset_from_json
from structured_ie.evaluation.evaluator import PipelineEvaluator
from structured_ie.schemas.models import ExtractionResult


def test_dataset_loading_and_structure() -> None:
    dataset_file = Path("data/synthetic_cases.json")
    assert dataset_file.exists(), "data/synthetic_cases.json should exist"

    samples = load_dataset_from_json(dataset_file)
    assert len(samples) == 20, f"Dataset should contain exactly 20 scenarios, got {len(samples)}"

    for sample in samples:
        assert isinstance(sample.text, str) and len(sample.text) > 0
        assert isinstance(sample.expected_entities, list)
        assert isinstance(sample.expected_relations, list)


def test_dataset_substring_offsets_exact() -> None:
    samples = load_dataset_from_json("data/synthetic_cases.json")

    for idx, sample in enumerate(samples, 1):
        for ent in sample.expected_entities:
            actual_substring = sample.text[ent.start : ent.end]
            assert actual_substring == ent.text, (
                f"Sample {idx} offset error for '{ent.text}': "
                f"text[{ent.start}:{ent.end}] gave '{actual_substring}'"
            )


def test_dataset_relations_reference_existing_entities() -> None:
    samples = load_dataset_from_json("data/synthetic_cases.json")

    for idx, sample in enumerate(samples, 1):
        entity_texts = {e.text for e in sample.expected_entities}
        for rel in sample.expected_relations:
            assert rel.source in entity_texts, (
                f"Sample {idx}: relation source '{rel.source}' not in expected entities {entity_texts}"
            )
            assert rel.target in entity_texts, (
                f"Sample {idx}: relation target '{rel.target}' not in expected entities {entity_texts}"
            )


def test_evaluator_runs_on_synthetic_dataset_with_mock() -> None:
    samples = load_dataset_from_json("data/synthetic_cases.json")

    # Mock pipeline that returns expected entities and relations
    class PerfectMockPipeline:
        def extract(self, text: str, entity_labels: list[str], relation_labels: list[str]):
            # Find matching sample
            for s in samples:
                if s.text == text:
                    return ExtractionResult(
                        text=text,
                        entities=s.expected_entities,
                        relations=s.expected_relations,
                    )
            return ExtractionResult(text=text)

    evaluator = PipelineEvaluator(
        entity_labels=["company", "NIP", "REGON"],
        relation_labels=["has NIP", "has REGON"],
    )

    result = evaluator.evaluate(pipeline=PerfectMockPipeline(), samples=samples)
    assert result.total_samples == len(samples)
    assert result.entity_micro.precision == 1.0
    assert result.entity_micro.recall == 1.0
    assert result.entity_micro.f1 == 1.0
    assert result.relation_micro.precision == 1.0
    assert result.relation_micro.recall == 1.0
    assert result.relation_micro.f1 == 1.0


def test_dataset_contains_only_simplified_labels() -> None:
    samples = load_dataset_from_json("data/synthetic_cases.json")
    allowed_entity_labels = {"company", "NIP", "REGON"}
    allowed_relation_labels = {"has NIP", "has REGON"}

    for idx, sample in enumerate(samples, 1):
        for ent in sample.expected_entities:
            assert ent.label in allowed_entity_labels, (
                f"Sample {idx}: entity '{ent.text}' has unexpected label '{ent.label}'"
            )
        for rel in sample.expected_relations:
            assert rel.relation in allowed_relation_labels, (
                f"Sample {idx}: relation has unexpected label '{rel.relation}'"
            )
