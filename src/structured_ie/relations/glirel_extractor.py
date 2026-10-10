"""GLiREL Zero-Shot Relation Extraction wrapper."""

import logging
from typing import Any

from glirel import GLiREL

from structured_ie.config import DEFAULT_CONFIG
from structured_ie.pipeline.alignment import (
    align_entities_to_tokens,
    convert_glirel_prediction_to_relation,
    tokenize_with_offsets,
)
from structured_ie.relations.base import BaseRelationExtractor
from structured_ie.schemas.models import Entity, Relation

logger = logging.getLogger(__name__)


def _patch_glirel_hub_compatibility() -> None:
    """Ensure GLiREL._from_pretrained has default values for proxies and resume_download.

    Upstream GLiREL 1.2.1 omitted default values for `proxies` and `resume_download` in
    `_from_pretrained`. In huggingface_hub>=0.23, `from_pretrained` no longer passes these
    parameters by default, leading to a TypeError unless defaults are provided.
    """
    orig = getattr(GLiREL, "_from_pretrained", None)
    if orig is None or getattr(GLiREL, "_hub_patched", False):
        return

    def _compat_from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None = None,
        cache_dir: Any = None,
        force_download: bool = False,
        proxies: dict | None = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Any = None,
        map_location: str = "cpu",
        strict: bool = False,
        **kwargs: Any,
    ) -> Any:
        return orig(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
            map_location=map_location,
            strict=strict,
            **kwargs,
        )

    GLiREL._from_pretrained = classmethod(_compat_from_pretrained)
    GLiREL._hub_patched = True


_patch_glirel_hub_compatibility()


class GLiRELExtractor(BaseRelationExtractor):
    """Zero-shot Relation Extractor using GLiREL."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        flat_ner: bool = True,
        top_k: int = -1,
        model: Any | None = None,
    ) -> None:
        """Initialize GLiREL extractor.

        Args:
            model_name: Hugging Face repository ID for GLiREL checkpoint.
            device: Computing device ('cpu', 'cuda', etc.).
            flat_ner: Whether to enforce flat NER constraints in evaluator.
            top_k: Maximum relation predictions per entity pair (-1 for all above threshold).
            model: Optional preloaded GLiREL instance (useful for testing and dependency injection).
        """
        self.model_name = model_name or DEFAULT_CONFIG.default_glirel_model
        self.device = device or DEFAULT_CONFIG.default_device
        self.flat_ner = flat_ner
        self.top_k = top_k

        if model is not None:
            self.model = model
        else:
            logger.info("Loading GLiREL model '%s' on device '%s'...", self.model_name, self.device)
            self.model = GLiREL.from_pretrained(self.model_name)
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            logger.info("GLiREL model loaded successfully.")

    def extract_relations(
        self,
        text: str,
        entities: list[Entity],
        labels: list[str],
        threshold: float | None = None,
    ) -> list[Relation]:
        """Extract relations between entities given relation labels.

        Args:
            text: Original text string.
            entities: Extracted Entity objects. At least 2 entities are required.
            labels: Candidate relation types.
            threshold: Confidence threshold for relation classification.

        Returns:
            List of normalized Relation objects sorted by score descending.
        """
        if len(entities) < 2 or not labels or not text.strip():
            logger.debug("Skipping relation extraction: fewer than 2 entities or no labels.")
            return []

        score_threshold = (
            threshold if threshold is not None else DEFAULT_CONFIG.default_relation_threshold
        )

        # 1. Tokenize text with exact character offset tracking
        token_spans = tokenize_with_offsets(text)
        tokens = [t.text for t in token_spans]

        # 2. Align character-based Entity spans into token-based GLiREL NER spans
        glirel_ner = align_entities_to_tokens(entities=entities, token_spans=token_spans)

        logger.debug(
            "Predicting relations for %d entities (%d tokens) across %d relation labels...",
            len(entities),
            len(tokens),
            len(labels),
        )

        # 3. Call GLiREL predict_relations
        # Passing tokens: list[str] avoids GLiREL string-slicing bug
        raw_predictions = self.model.predict_relations(
            tokens,
            labels,
            flat_ner=self.flat_ner,
            threshold=score_threshold,
            ner=glirel_ner,
            top_k=self.top_k,
        )

        # 4. Convert and normalize predictions back to Relation models
        relations: list[Relation] = []
        for pred in raw_predictions:
            rel = convert_glirel_prediction_to_relation(
                prediction=pred,
                entities=entities,
                glirel_ner=glirel_ner,
                tokens=tokens,
            )
            # Avoid self-relations if source equals target
            if rel.source != rel.target:
                relations.append(rel)

        # Sort relations by confidence score descending
        relations.sort(key=lambda r: r.score, reverse=True)
        return relations
