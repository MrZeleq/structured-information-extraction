"""Precision, Recall, and F1 calculation for entities and relations."""

from dataclasses import dataclass

from structured_ie.schemas.models import Entity, Relation


@dataclass(frozen=True)
class MetricScores:
    """Evaluation metrics summary."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def _calculate_prf(tp: int, fp: int, fn: int) -> MetricScores:
    """Safely compute Precision, Recall, and F1 avoiding division by zero."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return MetricScores(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )


def compute_entity_metrics(
    predicted: list[Entity],
    expected: list[Entity],
    match_mode: str = "exact",
) -> MetricScores:
    """Calculate Precision, Recall, and F1 for predicted entities against gold entities.

    Args:
        predicted: Extracted Entity objects.
        expected: Gold-standard expected Entity objects.
        match_mode:
            - 'exact': Matches normalized text and label (default).
            - 'span': Matches exact character offsets (start, end) and label.
            - 'relaxed': Matches label and allows substring containment or span overlap.

    Returns:
        MetricScores dataclass.
    """
    if match_mode == "span":
        pred_set = {(e.start, e.end, e.label.strip().lower()) for e in predicted}
        gold_set = {(e.start, e.end, e.label.strip().lower()) for e in expected}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
    elif match_mode in ("relaxed", "partial"):
        matched_gold_indices: set[int] = set()
        tp = 0
        for p in predicted:
            p_text = p.text.strip().lower()
            p_label = p.label.strip().lower()
            for g_idx, g in enumerate(expected):
                if g_idx in matched_gold_indices:
                    continue
                g_text = g.text.strip().lower()
                g_label = g.label.strip().lower()
                # Substring containment or span overlap
                spans_overlap = not (p.end <= g.start or p.start >= g.end)
                if p_label == g_label and (
                    p_text == g_text or p_text in g_text or g_text in p_text or spans_overlap
                ):
                    matched_gold_indices.add(g_idx)
                    tp += 1
                    break
        fp = len(predicted) - tp
        fn = len(expected) - tp
    else:
        # Default 'exact' matches text surface form and label
        pred_set = {(e.text.strip().lower(), e.label.strip().lower()) for e in predicted}
        gold_set = {(e.text.strip().lower(), e.label.strip().lower()) for e in expected}
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

    return _calculate_prf(tp, fp, fn)


def compute_relation_metrics(
    predicted: list[Relation],
    expected: list[Relation],
    match_mode: str = "exact",
) -> MetricScores:
    """Calculate Precision, Recall, and F1 for relation triplets.

    Args:
        predicted: Extracted Relation objects.
        expected: Gold-standard expected Relation objects.
        match_mode:
            - 'exact': Matches exact (source, relation, target) with case normalization (default).
            - 'relaxed': Matches relation label and permits substring containment on source and target.

    Returns:
        MetricScores dataclass.
    """
    if match_mode in ("relaxed", "partial"):
        matched_gold_indices: set[int] = set()
        tp = 0
        for p in predicted:
            p_src = p.source.strip().lower()
            p_rel = p.relation.strip().lower()
            p_tgt = p.target.strip().lower()
            for g_idx, g in enumerate(expected):
                if g_idx in matched_gold_indices:
                    continue
                g_src = g.source.strip().lower()
                g_rel = g.relation.strip().lower()
                g_tgt = g.target.strip().lower()
                src_match = p_src == g_src or p_src in g_src or g_src in p_src
                tgt_match = p_tgt == g_tgt or p_tgt in g_tgt or g_tgt in p_tgt
                if p_rel == g_rel and src_match and tgt_match:
                    matched_gold_indices.add(g_idx)
                    tp += 1
                    break
        fp = len(predicted) - tp
        fn = len(expected) - tp
        return _calculate_prf(tp, fp, fn)

    pred_set = {
        (
            r.source.strip().lower(),
            r.relation.strip().lower(),
            r.target.strip().lower(),
        )
        for r in predicted
    }
    gold_set = {
        (
            r.source.strip().lower(),
            r.relation.strip().lower(),
            r.target.strip().lower(),
        )
        for r in expected
    }

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return _calculate_prf(tp, fp, fn)
