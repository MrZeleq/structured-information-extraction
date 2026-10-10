"""Dataset evaluator for comparing different Information Extraction approaches."""

import logging
from dataclasses import dataclass
from typing import Any

from structured_ie.evaluation.metrics import (
    MetricScores,
    _calculate_prf,
    compute_entity_metrics,
    compute_relation_metrics,
)
from structured_ie.schemas.models import ExtractionResult, LabeledSample

logger = logging.getLogger(__name__)


@dataclass
class DatasetEvaluationResult:
    """Consolidated evaluation report across multiple samples."""

    total_samples: int
    entity_micro: MetricScores
    relation_micro: MetricScores
    per_sample_results: list[dict[str, Any]]

    def format_summary(self) -> str:
        """Render a readable summary table."""
        lines = [
            "=" * 60,
            f"IE EVALUATION REPORT (Total Samples: {self.total_samples})",
            "=" * 60,
            "Task                | Precision |  Recall   |    F1     | TP / FP / FN",
            "-" * 60,
            f"Entities (NER)      |  {self.entity_micro.precision:6.2%}   |  {self.entity_micro.recall:6.2%}   |  {self.entity_micro.f1:6.2%}   | {self.entity_micro.tp:2d} / {self.entity_micro.fp:2d} / {self.entity_micro.fn:2d}",
            f"Relations (RE)      |  {self.relation_micro.precision:6.2%}   |  {self.relation_micro.recall:6.2%}   |  {self.relation_micro.f1:6.2%}   | {self.relation_micro.tp:2d} / {self.relation_micro.fp:2d} / {self.relation_micro.fn:2d}",
            "=" * 60,
        ]
        return "\n".join(lines)


class PipelineEvaluator:
    """Evaluates any extraction pipeline over a dataset of labeled business samples."""

    def __init__(
        self,
        entity_labels: list[str],
        relation_labels: list[str],
        match_mode: str = "exact",
    ) -> None:
        """Initialize evaluator.

        Args:
            entity_labels: Entity categories candidate list.
            relation_labels: Relation types candidate list.
            match_mode: Matching mode for entities ('exact' or 'span').
        """
        self.entity_labels = entity_labels
        self.relation_labels = relation_labels
        self.match_mode = match_mode

    def evaluate(
        self,
        pipeline: Any,
        samples: list[LabeledSample],
    ) -> DatasetEvaluationResult:
        """Run evaluation over a dataset of labeled samples.

        Args:
            pipeline: Object with an `extract(text, entity_labels, relation_labels)` method.
            samples: List of LabeledSample instances.

        Returns:
            DatasetEvaluationResult with micro scores and per-sample logs.
        """
        total_ent_tp = 0
        total_ent_fp = 0
        total_ent_fn = 0

        total_rel_tp = 0
        total_rel_fp = 0
        total_rel_fn = 0

        sample_details: list[dict[str, Any]] = []

        for idx, sample in enumerate(samples):
            logger.info("Evaluating sample %d/%d...", idx + 1, len(samples))
            result: ExtractionResult = pipeline.extract(
                text=sample.text,
                entity_labels=self.entity_labels,
                relation_labels=self.relation_labels,
            )

            ent_m = compute_entity_metrics(
                predicted=result.entities,
                expected=sample.expected_entities,
                match_mode=self.match_mode,
            )
            rel_m = compute_relation_metrics(
                predicted=result.relations,
                expected=sample.expected_relations,
                match_mode=self.match_mode,
            )

            total_ent_tp += ent_m.tp
            total_ent_fp += ent_m.fp
            total_ent_fn += ent_m.fn

            total_rel_tp += rel_m.tp
            total_rel_fp += rel_m.fp
            total_rel_fn += rel_m.fn

            sample_details.append(
                {
                    "index": idx,
                    "text": sample.text,
                    "predicted_entities": [e.to_dict() for e in result.entities],
                    "expected_entities": [e.to_dict() for e in sample.expected_entities],
                    "entity_metrics": ent_m.to_dict(),
                    "predicted_relations": [r.to_dict() for r in result.relations],
                    "expected_relations": [r.to_dict() for r in sample.expected_relations],
                    "relation_metrics": rel_m.to_dict(),
                }
            )

        entity_micro = _calculate_prf(total_ent_tp, total_ent_fp, total_ent_fn)
        relation_micro = _calculate_prf(total_rel_tp, total_rel_fp, total_rel_fn)

        return DatasetEvaluationResult(
            total_samples=len(samples),
            entity_micro=entity_micro,
            relation_micro=relation_micro,
            per_sample_results=sample_details,
        )
