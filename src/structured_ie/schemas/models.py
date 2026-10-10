"""Normalized data models for entities, relations, and extraction results."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel):
    """Normalized named entity with text, label, confidence, and character offsets."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Extracted entity surface form text")
    label: str = Field(description="Entity type/class label")
    score: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    start: int = Field(ge=0, description="Start character offset in original text")
    end: int = Field(ge=0, description="End character offset in original text")

    def to_dict(self) -> dict[str, Any]:
        """Convert to basic dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "start": self.start,
            "end": self.end,
        }


class Relation(BaseModel):
    """Normalized relation between source and target entities."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Surface text of the head/source entity")
    relation: str = Field(description="Relation predicate type")
    target: str = Field(description="Surface text of the tail/target entity")
    score: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    source_span: tuple[int, int] | None = Field(
        default=None,
        description="Character start and end span of source entity in text",
    )
    target_span: tuple[int, int] | None = Field(
        default=None,
        description="Character start and end span of target entity in text",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to basic dictionary."""
        d: dict[str, Any] = {
            "source": self.source,
            "relation": self.relation,
            "target": self.target,
            "score": round(self.score, 4),
        }
        if self.source_span is not None:
            d["source_span"] = list(self.source_span)
        if self.target_span is not None:
            d["target_span"] = list(self.target_span)
        return d


class ExtractionResult(BaseModel):
    """Combined output containing extracted entities and relations for an input text."""

    text: str = Field(description="Original input text")
    entities: list[Entity] = Field(default_factory=list, description="Extracted entities")
    relations: list[Relation] = Field(default_factory=list, description="Extracted relations")

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary format."""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
        }


class LabeledSample(BaseModel):
    """Gold-standard sample for evaluation and testing."""

    text: str
    expected_entities: list[Entity] = Field(default_factory=list)
    expected_relations: list[Relation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
