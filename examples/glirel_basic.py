#!/usr/bin/env python3
"""Phase 2 Example: Zero-shot Relation Extraction using GLiREL.

Extracts relationships between pre-identified entities without needing fine-tuning.
"""

import json
import logging

from structured_ie.relations.glirel_extractor import GLiRELExtractor
from structured_ie.schemas.models import Entity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    print("=" * 70)
    print("GLiREL Zero-Shot Relation Extraction (Phase 2)")
    print("=" * 70)

    # Initialize GLiREL extractor (uses default 'jackboyla/glirel-large-v0')
    extractor = GLiRELExtractor(top_k=1)

    text = (
        "Piotr Zieliński reprezentujący spółkę Beta Trade sp. z o.o. o NIP 1234567890 "
        "podpisał długoterminową umowę o współpracę z brokerem Omega Broker Sp. k."
    )

    # Pre-defined known entities (as would come from an upstream NER step)
    entities = [
        Entity(text="Piotr Zieliński", label="person", score=0.99, start=0, end=15),
        Entity(text="Beta Trade sp. z o.o.", label="company", score=0.98, start=33, end=54),
        Entity(text="1234567890", label="NIP", score=0.97, start=61, end=71),
        Entity(text="Omega Broker Sp. k.", label="company", score=0.96, start=126, end=145),
    ]

    relation_labels = [
        "works for",
        "has NIP",
        "broker for",
        "insured by",
        "contact person",
    ]

    print("\nInput Text:")
    print(f"  {text}")
    print("\nPre-identified Entities:")
    for ent in entities:
        print(f"  - [{ent.label}] '{ent.text}' ({ent.start}..{ent.end})")
    print(f"\nCandidate Relation Labels:\n  {relation_labels}\n")

    # Extract relations
    relations = extractor.extract_relations(
        text=text,
        entities=entities,
        labels=relation_labels,
        threshold=0.3,
    )

    print(f"Extracted {len(relations)} relations:")
    for rel in relations:
        print(f"  '{rel.source}' --[{rel.relation}]--> '{rel.target}' (score: {rel.score:.4f})")

    print("\nNormalized JSON Representation:")
    normalized_list = [r.to_dict() for r in relations]
    print(json.dumps(normalized_list, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
