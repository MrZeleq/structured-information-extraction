"""Specialized end-to-end pipeline for extracting brokers, clients, and their tax identifiers."""

import logging
import time

from structured_ie.ner.base import BaseNERExtractor
from structured_ie.ner.gliner_extractor import GLiNERExtractor
from structured_ie.normalization.text_normalizer import TextNormalizer
from structured_ie.postprocessing.broker_client_resolver import (
    BrokerClientExtraction,
    BrokerClientResolver,
)
from structured_ie.postprocessing.entity_cleaner import EntityPostprocessor
from structured_ie.schemas.models import Entity, ExtractionResult

logger = logging.getLogger(__name__)


class BrokerClientPipeline:
    """Pipeline extracting broker, client, and NIP associations using two-stage neuro-symbolic resolution."""

    def __init__(
        self,
        ner_extractor: BaseNERExtractor,
        postprocessor: EntityPostprocessor | None = None,
        resolver: BrokerClientResolver | None = None,
    ) -> None:
        """Initialize broker-client pipeline.

        Args:
            ner_extractor: Extractor implementing BaseNERExtractor (typically GLiNER).
            postprocessor: Optional EntityPostprocessor for normalizing NIPs and company names.
            resolver: Optional BrokerClientResolver for role disambiguation and NIP binding.
        """
        self.ner_extractor = ner_extractor
        self.postprocessor = postprocessor or EntityPostprocessor()
        self.resolver = resolver or BrokerClientResolver()

    @classmethod
    def from_pretrained(
        cls,
        gliner_model_name: str | None = None,
        device: str | None = None,
        flat_ner: bool = True,
    ) -> "BrokerClientPipeline":
        """Convenience factory creating a ready-to-use pipeline with pre-trained models."""
        ner = GLiNERExtractor(model_name=gliner_model_name, device=device, flat_ner=flat_ner)
        postprocessor = EntityPostprocessor()
        resolver = BrokerClientResolver()
        return cls(ner_extractor=ner, postprocessor=postprocessor, resolver=resolver)

    def _extract_and_clean_entities(self, text: str, threshold: float) -> list[Entity]:
        """Extract entities via neural NER and augment with OCR-tolerant NIP candidate regexes."""
        # Stage 1: High-precision zero-shot entity extraction
        raw_entities = self.ner_extractor.extract_entities(
            text=text,
            labels=["company", "NIP"],
            threshold=threshold,
        )

        # Stage 1b: Pattern-based NIP augmentation for OCR resilience (e.g. 'N1P:', spaced NIPs)
        nip_candidates = TextNormalizer.find_nip_candidates(text)
        existing_spans = {(e.start, e.end) for e in raw_entities if e.label == "NIP"}
        for cand in nip_candidates:
            if not any(
                abs(cand.start - s) <= 5 and abs(cand.end - e) <= 5 for s, e in existing_spans
            ):
                raw_entities.append(
                    Entity(
                        text=cand.normalized_nip,
                        label="NIP",
                        score=0.99,
                        start=cand.start,
                        end=cand.end,
                    )
                )

        # Stage 2: Post-processing and character-offset normalization
        return self.postprocessor.postprocess(entities=raw_entities, text=text)

    def extract(
        self,
        text: str,
        entity_labels: list[str] | None = None,
        relation_labels: list[str] | None = None,
        entity_threshold: float | None = None,
        relation_threshold: float | None = None,
    ) -> ExtractionResult:
        """Extract broker, client, and NIP associations from raw text.

        Complies with standard ExtractionPipeline interface for seamless evaluation.

        Args:
            text: Input document text.
            entity_labels: Optional entity labels (defaults to ['company', 'NIP']).
            relation_labels: Optional relation labels (unused, resolved deterministically).
            entity_threshold: Confidence threshold for NER.
            relation_threshold: Confidence threshold for RE (unused).

        Returns:
            Standard ExtractionResult containing re-typed entities ('broker', 'client', 'NIP')
            and exact relations ('has NIP', 'broker for').
        """
        ent_thresh = entity_threshold if entity_threshold is not None else 0.25

        t0 = time.perf_counter()
        cleaned_entities = self._extract_and_clean_entities(text=text, threshold=ent_thresh)

        # Stage 3: Neuro-symbolic role classification and NIP binding
        resolved_entities, relations = self.resolver.resolve(
            entities=cleaned_entities,
            text=text,
        )
        t_end = time.perf_counter()
        logger.debug(
            "BrokerClientPipeline extracted %d entities and %d relations in %.3fs",
            len(resolved_entities),
            len(relations),
            t_end - t0,
        )

        return ExtractionResult(
            text=text,
            entities=resolved_entities,
            relations=relations,
        )

    def extract_structured(
        self,
        text: str,
        entity_threshold: float | None = None,
    ) -> BrokerClientExtraction:
        """Extract high-level business entity model with broker and client party objects.

        Args:
            text: Input document text.
            entity_threshold: Optional confidence threshold for NER.

        Returns:
            BrokerClientExtraction containing BrokerParty and ClientParty with exact NIPs.
        """
        ent_thresh = entity_threshold if entity_threshold is not None else 0.25
        cleaned_entities = self._extract_and_clean_entities(text=text, threshold=ent_thresh)
        return self.resolver.extract_structured(entities=cleaned_entities, text=text)
