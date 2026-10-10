"""Schema-based relation filtering and cardinality constraint enforcement."""

import logging

from structured_ie.schemas.constraints import DEFAULT_BUSINESS_SCHEMA, ExtractionSchema
from structured_ie.schemas.models import Entity, Relation

logger = logging.getLogger(__name__)


class RelationFilter:
    """Enforces domain, range, and functional constraints on extracted relations.

    Key responsibilities:
    1. Schema Type Validation: Rejects relations where source/target entity labels
       violate the declared schema (e.g. rejecting 'REGON -> has NIP -> Company').
    2. Inversion Protection: Rejects inverted directions (head/tail swap).
    3. Functional Dependency Deduplication: When a relation specifies functional_target=True
       (such as an official tax identifier belonging to only one legal entity),
       multiple competing source candidates for the same identifier are pruned to the
       highest-scoring candidate.
    """

    def __init__(self, schema: ExtractionSchema | None = None) -> None:
        """Initialize relation filter with an extraction schema.

        Args:
            schema: ExtractionSchema defining valid type pairings and constraints.
                Defaults to DEFAULT_BUSINESS_SCHEMA.
        """
        self.schema = schema if schema is not None else DEFAULT_BUSINESS_SCHEMA

    def filter(
        self,
        relations: list[Relation],
        entities: list[Entity],
    ) -> list[Relation]:
        """Filter and deduplicate candidate relations against the schema.

        Args:
            relations: Extracted Relation objects from relation extraction model.
            entities: Available Entity objects (with verified types).

        Returns:
            Filtered list of valid Relation objects.
        """
        if not relations or not entities:
            return []

        # Map entity surface forms to their corresponding entity labels
        # In case of duplicate text with different labels, keep latest or highest score
        entity_label_map: dict[str, str] = {}
        for ent in entities:
            entity_label_map[ent.text] = ent.label

        type_valid_relations: list[Relation] = []

        for rel in relations:
            src_label = entity_label_map.get(rel.source)
            tgt_label = entity_label_map.get(rel.target)

            if src_label is None or tgt_label is None:
                # If either endpoint cannot be verified in current entities, skip
                continue

            # Check validity against schema
            if not self.schema.is_valid_pair(
                relation_name=rel.relation,
                source_label=src_label,
                target_label=tgt_label,
            ):
                logger.debug(
                    "Filtered out relation '%s' (%s -> %s) violating schema.",
                    rel.relation,
                    src_label,
                    tgt_label,
                )
                continue

            type_valid_relations.append(rel)

        # Apply functional cardinality constraints (e.g. 1 company per NIP/REGON)
        deduplicated = self._apply_functional_constraints(type_valid_relations)
        deduplicated.sort(key=lambda r: r.score, reverse=True)
        return deduplicated

    def _apply_functional_constraints(self, relations: list[Relation]) -> list[Relation]:
        """Prune multi-association collisions based on functional dependencies."""
        result: list[Relation] = []

        # Group by relation predicate
        by_predicate: dict[str, list[Relation]] = {}
        for r in relations:
            by_predicate.setdefault(r.relation, []).append(r)

        for rel_name, rel_list in by_predicate.items():
            constraint = self.schema.get_constraint(rel_name)
            if constraint is None or not constraint.functional_target:
                result.extend(rel_list)
                continue

            # For functional_target=True: each target string must belong to at most ONE source.
            # E.g. NIP '5251234567' can be claimed by only one company. Pick highest score.
            best_by_target: dict[str, Relation] = {}
            for r in rel_list:
                existing = best_by_target.get(r.target)
                if existing is None or r.score > existing.score:
                    best_by_target[r.target] = r

            result.extend(best_by_target.values())

        return result
