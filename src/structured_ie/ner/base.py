"""Abstract base class for Named Entity Recognition extractors."""

from abc import ABC, abstractmethod

from structured_ie.schemas.models import Entity


class BaseNERExtractor(ABC):
    """Abstract interface for NER extractors.

    Enables plug-and-play comparison between GLiNER, NuNER, fine-tuned encoders,
    or LLM-based entity extractors.
    """

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[Entity]:
        """Extract entities from a single text using schema labels.

        Args:
            text: Input text string.
            labels: List of entity types to identify.
            threshold: Confidence threshold for filtering predictions.

        Returns:
            List of normalized Entity objects.
        """
        ...

    def batch_extract_entities(
        self,
        texts: list[str],
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[list[Entity]]:
        """Extract entities for a list of texts. Default implementation iterates."""
        return [self.extract_entities(text=t, labels=labels, threshold=threshold) for t in texts]
