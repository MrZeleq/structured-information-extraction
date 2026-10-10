"""End-to-end Information Extraction pipeline combining NER and Relation Extraction."""

import logging
import time

from structured_ie.config import DEFAULT_CONFIG
from structured_ie.ner.base import BaseNERExtractor
from structured_ie.ner.gliner_extractor import GLiNERExtractor
from structured_ie.postprocessing.entity_cleaner import EntityPostprocessor
from structured_ie.postprocessing.relation_filter import RelationFilter
from structured_ie.relations.base import BaseRelationExtractor
from structured_ie.relations.glirel_extractor import GLiRELExtractor
from structured_ie.schemas.constraints import DEFAULT_BUSINESS_SCHEMA, ExtractionSchema
from structured_ie.schemas.models import ExtractionResult

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """Composite pipeline executing NER followed by Relation Extraction."""

    def __init__(
        self,
        ner_extractor: BaseNERExtractor,
        relation_extractor: BaseRelationExtractor,
        postprocessor: EntityPostprocessor | None = None,
        relation_filter: RelationFilter | None = None,
    ) -> None:
        """Initialize extraction pipeline with modular extractors and filters.

        Args:
            ner_extractor: Extractor implementing BaseNERExtractor.
            relation_extractor: Extractor implementing BaseRelationExtractor.
            postprocessor: Optional EntityPostprocessor for cleaning entity spans.
            relation_filter: Optional RelationFilter for schema validation and deduplication.
        """
        self.ner_extractor = ner_extractor
        self.relation_extractor = relation_extractor
        self.postprocessor = postprocessor
        self.relation_filter = relation_filter

    @classmethod
    def from_pretrained(
        cls,
        gliner_model_name: str | None = None,
        glirel_model_name: str | None = None,
        device: str | None = None,
        flat_ner: bool = True,
        enable_postprocessing: bool = True,
        schema: ExtractionSchema | None = None,
    ) -> "ExtractionPipeline":
        """Convenience constructor instantiating default GLiNER and GLiREL models."""
        ner = GLiNERExtractor(model_name=gliner_model_name, device=device, flat_ner=flat_ner)
        re = GLiRELExtractor(model_name=glirel_model_name, device=device, flat_ner=flat_ner)
        postprocessor = EntityPostprocessor() if enable_postprocessing else None
        target_schema = schema if schema is not None else DEFAULT_BUSINESS_SCHEMA
        rel_filter = RelationFilter(schema=target_schema) if enable_postprocessing else None
        return cls(
            ner_extractor=ner,
            relation_extractor=re,
            postprocessor=postprocessor,
            relation_filter=rel_filter,
        )

    def extract(
        self,
        text: str,
        entity_labels: list[str],
        relation_labels: list[str],
        entity_threshold: float | None = None,
        relation_threshold: float | None = None,
        apply_postprocessing: bool = True,
    ) -> ExtractionResult:
        """Execute the two-stage extraction pipeline.

        1. Extract entities using NER extractor with `entity_labels`.
        2. Convert and align entity representations.
        3. Extract relations between candidate entities with `relation_labels`.
        4. Return normalized ExtractionResult.

        Args:
            text: Input text string.
            entity_labels: Target named entity classes for NER.
            relation_labels: Target relation types for RE.
            entity_threshold: Optional confidence threshold for NER.
            relation_threshold: Optional confidence threshold for RE.

        Returns:
            ExtractionResult containing extracted entities and relations.
        """
        ent_thresh = (
            entity_threshold
            if entity_threshold is not None
            else DEFAULT_CONFIG.default_entity_threshold
        )
        rel_thresh = (
            relation_threshold
            if relation_threshold is not None
            else DEFAULT_CONFIG.default_relation_threshold
        )

        t0 = time.perf_counter()
        logger.info("Starting NER stage for text length %d...", len(text))
        entities = self.ner_extractor.extract_entities(
            text=text,
            labels=entity_labels,
            threshold=ent_thresh,
        )
        t_ner = time.perf_counter()
        logger.info(
            "NER completed in %.3fs: %d raw entities identified.",
            t_ner - t0,
            len(entities),
        )

        # 2. Entity post-processing & disambiguation
        if apply_postprocessing and self.postprocessor is not None:
            entities = self.postprocessor.postprocess(entities=entities, text=text)
            logger.debug(
                "Post-processed entities: %d entities retained after normalization.",
                len(entities),
            )

        relations = []
        if len(entities) >= 2 and relation_labels:
            logger.info("Starting Relation Extraction stage with %d entities...", len(entities))
            relations = self.relation_extractor.extract_relations(
                text=text,
                entities=entities,
                labels=relation_labels,
                threshold=rel_thresh,
            )
            t_re = time.perf_counter()
            logger.info(
                "Relation Extraction completed in %.3fs: %d raw relations extracted.",
                t_re - t_ner,
                len(relations),
            )

            # 3. Schema validation & relation constraint filtering
            if apply_postprocessing and self.relation_filter is not None:
                relations = self.relation_filter.filter(relations=relations, entities=entities)
                logger.debug(
                    "Post-filtered relations: %d relations retained after schema validation.",
                    len(relations),
                )
        else:
            logger.info(
                "Skipping Relation Extraction: insufficient entities (%d) or empty relation labels.",
                len(entities),
            )

        return ExtractionResult(
            text=text,
            entities=entities,
            relations=relations,
        )
