"""Abstract base class for Relation Extraction models."""

from abc import ABC, abstractmethod

from structured_ie.schemas.models import Entity, Relation


class BaseRelationExtractor(ABC):
    """Abstract interface for Relation Extraction models.

    Enables plug-and-play comparison between GLiREL, fine-tuned relation classifiers,
    or LLM-based relation extractors.
    """

    @abstractmethod
    def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[Relation]:
        """Extract relations between extracted entities.

        Args:
            text: Original input text string.
            entities: Pre-extracted Entity objects.
            labels: Candidate relation types (e.g. 'works for', 'has NIP').
            threshold: Confidence threshold for predicting relations.

        Returns:
            List of normalized Relation objects.
        """
        ...
