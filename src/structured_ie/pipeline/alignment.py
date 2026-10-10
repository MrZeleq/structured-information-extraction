"""Tokenization and entity span alignment utilities between GLiNER and GLiREL."""

import re
from dataclasses import dataclass

from structured_ie.schemas.models import Entity, Relation


@dataclass(frozen=True)
class TokenSpan:
    """Token surface string with character offsets and token index."""

    text: str
    start: int
    end: int
    index: int


def tokenize_with_offsets(text: str) -> list[TokenSpan]:
    """Tokenize text while recording exact character start and end offsets.

    Uses a word-and-punctuation regex compatible with GLiREL token boundaries:
    r'\\w+(?:[-_]\\w+)*|\\S'.
    """
    token_spans: list[TokenSpan] = []
    pattern = re.compile(r"\w+(?:[-_]\w+)*|\S")

    for idx, match in enumerate(pattern.finditer(text)):
        token_spans.append(
            TokenSpan(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                index=idx,
            )
        )
    return token_spans


def align_entities_to_tokens(
    entities: list[Entity],
    token_spans: list[TokenSpan],
) -> list[list[int | str]]:
    """Convert character-based Entity objects into token-based GLiREL NER spans.

    GLiREL expects:
        [[start_token_idx, end_token_idx, label, text], ...]
    where start_token_idx and end_token_idx are 0-indexed, inclusive token positions.

    Args:
        entities: List of normalized Entity objects with character offsets.
        token_spans: Tokenized text with character offsets.

    Returns:
        List of lists in GLiREL format.
    """
    if not entities or not token_spans:
        return []

    glirel_ner: list[list[int | str]] = []

    for entity in entities:
        # Find all tokens overlapping the entity character span [entity.start, entity.end)
        overlapping_tokens = [
            t for t in token_spans if t.end > entity.start and t.start < entity.end
        ]

        if overlapping_tokens:
            start_token_idx = overlapping_tokens[0].index
            end_token_idx = overlapping_tokens[-1].index
        else:
            # Fallback to closest token by midpoint
            midpoint = (entity.start + entity.end) / 2.0
            closest_token = min(
                token_spans,
                key=lambda t: abs(((t.start + t.end) / 2.0) - midpoint),
            )
            start_token_idx = closest_token.index
            end_token_idx = closest_token.index

        glirel_ner.append([start_token_idx, end_token_idx, entity.label, entity.text])

    return glirel_ner


def convert_glirel_prediction_to_relation(
    prediction: dict,
    entities: list[Entity],
    glirel_ner: list[list[int | str]],
    tokens: list[str],
) -> Relation:
    """Convert a single raw GLiREL prediction dictionary into a normalized Relation.

    GLiREL returns:
        {
            'head_pos': [head_start, head_end_exclusive],
            'tail_pos': [tail_start, tail_end_exclusive],
            'head_text': list[str],
            'tail_text': list[str],
            'label': str,
            'score': float
        }

    This function matches the token span indices back to the original Entity objects
    to preserve true surface text and precise character offsets.
    """
    head_pos = prediction.get("head_pos", [0, 0])
    tail_pos = prediction.get("tail_pos", [0, 0])
    label = str(prediction.get("label", ""))
    score = float(prediction.get("score", 0.0))

    # Convert Python slice indices [start, end_exclusive] to GLiREL input format [start, end_inclusive]
    head_token_span = (head_pos[0], head_pos[1] - 1)
    tail_token_span = (tail_pos[0], tail_pos[1] - 1)

    # Build index to map token span back to Entity
    head_entity: Entity | None = None
    tail_entity: Entity | None = None

    for ent, ner_span in zip(entities, glirel_ner, strict=False):
        span_tuple = (ner_span[0], ner_span[1])
        if span_tuple == head_token_span and head_entity is None:
            head_entity = ent
        if span_tuple == tail_token_span and tail_entity is None:
            tail_entity = ent

    # Resolve source text and span
    if head_entity is not None:
        source_text = head_entity.text
        source_span = (head_entity.start, head_entity.end)
    else:
        raw_head = prediction.get("head_text", [])
        source_text = (
            " ".join(raw_head)
            if isinstance(raw_head, list)
            else " ".join(tokens[head_pos[0] : head_pos[1]])
        )
        source_span = None

    # Resolve target text and span
    if tail_entity is not None:
        target_text = tail_entity.text
        target_span = (tail_entity.start, tail_entity.end)
    else:
        raw_tail = prediction.get("tail_text", [])
        target_text = (
            " ".join(raw_tail)
            if isinstance(raw_tail, list)
            else " ".join(tokens[tail_pos[0] : tail_pos[1]])
        )
        target_span = None

    return Relation(
        source=source_text,
        relation=label,
        target=target_text,
        score=score,
        source_span=source_span,
        target_span=target_span,
    )
