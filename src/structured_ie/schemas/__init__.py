"""Data schemas for structured information extraction."""

from structured_ie.schemas.models import (
    Entity,
    ExtractionResult,
    LabeledSample,
    Relation,
)

__all__ = ["Entity", "Relation", "ExtractionResult", "LabeledSample"]
