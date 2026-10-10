"""GLiNER Named Entity Recognition extractor wrapper."""

import logging
from typing import Any

from gliner import GLiNER

from structured_ie.config import DEFAULT_CONFIG
from structured_ie.ner.base import BaseNERExtractor
from structured_ie.schemas.models import Entity

logger = logging.getLogger(__name__)


class GLiNERExtractor(BaseNERExtractor):
    """Zero-shot NER extractor using GLiNER."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        flat_ner: bool = True,
        model: Any | None = None,
    ) -> None:
        """Initialize GLiNER extractor.

        Args:
            model_name: Hugging Face repository ID for GLiNER checkpoint.
            device: Computing device ('cpu', 'cuda', etc.).
            flat_ner: If True, resolves nested/overlapping spans to highest scoring entity.
            model: Optional preloaded GLiNER instance (useful for testing and dependency injection).
        """
        self.model_name = model_name or DEFAULT_CONFIG.default_gliner_model
        self.device = device or DEFAULT_CONFIG.default_device
        self.flat_ner = flat_ner

        if model is not None:
            self.model = model
        else:
            logger.info("Loading GLiNER model '%s' on device '%s'...", self.model_name, self.device)
            self.model = GLiNER.from_pretrained(self.model_name)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            logger.info("GLiNER model loaded successfully.")

    def extract_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float | None = None,
    ) -> list[Entity]:
        """Extract entities using zero-shot labels.

        Args:
            text: Input string.
            labels: List of target entity classes.
            threshold: Confidence threshold (defaults to config default).

        Returns:
            List of normalized Entity objects sorted by character start position.
        """
        if not text.strip() or not labels:
            return []

        score_threshold = (
            threshold if threshold is not None else DEFAULT_CONFIG.default_entity_threshold
        )

        logger.debug(
            "Extracting entities for text (len=%d) with %d labels at threshold %.2f",
            len(text),
            len(labels),
            score_threshold,
        )

        raw_predictions = self.model.predict_entities(
            text,
            labels,
            flat_ner=self.flat_ner,
            threshold=score_threshold,
        )

        entities: list[Entity] = []
        for raw in raw_predictions:
            ent = Entity(
                text=str(raw["text"]).strip(),
                label=str(raw["label"]).strip(),
                score=float(raw["score"]),
                start=int(raw["start"]),
                end=int(raw["end"]),
            )
            entities.append(ent)

        # Sort entities by start offset, then by end offset
        entities.sort(key=lambda e: (e.start, e.end))
        return entities
