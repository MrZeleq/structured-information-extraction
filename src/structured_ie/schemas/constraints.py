"""Declarative constraints and schema definitions for information extraction."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RelationConstraint:
    """Type and cardinality constraints for a relation predicate.

    Attributes:
        relation_name: Name of the relation (e.g. 'has NIP').
        allowed_source_types: Set of allowed entity labels for head/source entity.
        allowed_target_types: Set of allowed entity labels for tail/target entity.
        functional_target: When True, a target entity can belong to at most one source
            entity (e.g. unique identifiers like NIP/REGON belong to a single company).
            If multiple sources are predicted for the same target, the highest scoring
            is preserved.
        functional_source: When True, a source entity can link to at most one target entity.
    """

    relation_name: str
    allowed_source_types: set[str]
    allowed_target_types: set[str]
    functional_target: bool = False
    functional_source: bool = False

    def is_valid(self, source_label: str, target_label: str) -> bool:
        """Check whether the given entity type pair is valid under this constraint."""
        src_ok = not self.allowed_source_types or source_label in self.allowed_source_types
        tgt_ok = not self.allowed_target_types or target_label in self.allowed_target_types
        return src_ok and tgt_ok


@dataclass
class ExtractionSchema:
    """Schema containing relational constraints across entity and relation types."""

    constraints: dict[str, RelationConstraint] = field(default_factory=dict)

    def add_constraint(self, constraint: RelationConstraint) -> "ExtractionSchema":
        """Register a constraint for a relation predicate."""
        self.constraints[constraint.relation_name] = constraint
        return self

    def get_constraint(self, relation_name: str) -> RelationConstraint | None:
        """Retrieve constraint definition for a given relation, or None if unconstrained."""
        return self.constraints.get(relation_name)

    def is_valid_pair(self, relation_name: str, source_label: str, target_label: str) -> bool:
        """Check whether relation between source and target labels is permitted."""
        constraint = self.get_constraint(relation_name)
        if constraint is None:
            return True
        return constraint.is_valid(source_label, target_label)


# Default business schema for Polish corporate entities and tax identifiers
DEFAULT_BUSINESS_SCHEMA = ExtractionSchema(
    constraints={
        "has NIP": RelationConstraint(
            relation_name="has NIP",
            allowed_source_types={"company", "broker", "client"},
            allowed_target_types={"NIP"},
            functional_target=True,
        ),
        "has REGON": RelationConstraint(
            relation_name="has REGON",
            allowed_source_types={"company", "broker", "client"},
            allowed_target_types={"REGON"},
            functional_target=True,
        ),
        "broker for": RelationConstraint(
            relation_name="broker for",
            allowed_source_types={"company", "broker"},
            allowed_target_types={"company", "client"},
            functional_target=False,
        ),
        "works for": RelationConstraint(
            relation_name="works for",
            allowed_source_types={"person"},
            allowed_target_types={"company", "broker", "client"},
            functional_target=False,
        ),
    }
)
