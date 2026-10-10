"""Tests for evaluation metrics and PipelineEvaluator."""

from structured_ie.evaluation.evaluator import PipelineEvaluator
from structured_ie.evaluation.metrics import (
    compute_entity_metrics,
    compute_relation_metrics,
)
from structured_ie.schemas.models import (
    Entity,
    ExtractionResult,
    LabeledSample,
    Relation,
)


def test_entity_metrics_perfect() -> None:
    expected = [
        Entity(text="Google", label="company", score=1.0, start=0, end=6),
        Entity(text="Alice", label="person", score=1.0, start=10, end=15),
    ]
    predicted = [
        Entity(text="Google", label="company", score=0.9, start=0, end=6),
        Entity(text="Alice", label="person", score=0.85, start=10, end=15),
    ]

    scores = compute_entity_metrics(predicted, expected, match_mode="exact")
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
    assert scores.tp == 2
    assert scores.fp == 0
    assert scores.fn == 0


def test_entity_metrics_partial() -> None:
    expected = [
        Entity(text="Google", label="company", score=1.0, start=0, end=6),
        Entity(text="Alice", label="person", score=1.0, start=10, end=15),
    ]
    predicted = [
        Entity(text="Google", label="company", score=0.9, start=0, end=6),
        Entity(text="Bob", label="person", score=0.8, start=20, end=23),  # FP
    ]

    scores = compute_entity_metrics(predicted, expected, match_mode="exact")
    assert scores.tp == 1
    assert scores.fp == 1
    assert scores.fn == 1
    assert scores.precision == 0.5
    assert scores.recall == 0.5
    assert scores.f1 == 0.5


def test_relation_metrics_exact() -> None:
    expected = [
        Relation(source="Alice", relation="works for", target="Google", score=1.0),
    ]
    predicted = [
        Relation(source="alice", relation="works for", target="google", score=0.92),
    ]

    scores = compute_relation_metrics(predicted, expected)
    assert scores.tp == 1
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0


def test_pipeline_evaluator_run() -> None:
    sample = LabeledSample(
        text="Sample text",
        expected_entities=[Entity(text="ABC", label="company", score=1.0, start=0, end=3)],
        expected_relations=[],
    )

    class MockPipeline:
        def extract(self, text, entity_labels, relation_labels):
            return ExtractionResult(
                text=text,
                entities=[Entity(text="ABC", label="company", score=0.9, start=0, end=3)],
                relations=[],
            )

    evaluator = PipelineEvaluator(
        entity_labels=["company"],
        relation_labels=["works for"],
    )

    result = evaluator.evaluate(pipeline=MockPipeline(), samples=[sample])
    assert result.total_samples == 1
    assert result.entity_micro.f1 == 1.0
    assert result.relation_micro.f1 == 1.0
    summary_text = result.format_summary()
    assert "IE EVALUATION REPORT" in summary_text


def test_entity_metrics_relaxed_substring() -> None:
    expected = [
        Entity(text="Pol-Trans Sp. z o.o.", label="company", score=1.0, start=35, end=55),
    ]
    # Predicted is a superstring
    predicted = [
        Entity(
            text="Przedsiębiorstwo Handlowo-Usługowe Pol-Trans Sp. z o.o.",
            label="company",
            score=0.85,
            start=0,
            end=55,
        ),
    ]

    exact_scores = compute_entity_metrics(predicted, expected, match_mode="exact")
    assert exact_scores.tp == 0

    relaxed_scores = compute_entity_metrics(predicted, expected, match_mode="relaxed")
    assert relaxed_scores.tp == 1
    assert relaxed_scores.precision == 1.0
    assert relaxed_scores.recall == 1.0


def test_relation_metrics_relaxed() -> None:
    expected = [
        Relation(source="Pol-Trans Sp. z o.o.", relation="has NIP", target="5251234567", score=1.0),
    ]
    predicted = [
        Relation(
            source="Przedsiębiorstwo Pol-Trans Sp. z o.o.",
            relation="has NIP",
            target="5251234567",
            score=0.88,
        ),
    ]

    exact_scores = compute_relation_metrics(predicted, expected, match_mode="exact")
    assert exact_scores.tp == 0

    relaxed_scores = compute_relation_metrics(predicted, expected, match_mode="relaxed")
    assert relaxed_scores.tp == 1
    assert relaxed_scores.precision == 1.0
    assert relaxed_scores.recall == 1.0
