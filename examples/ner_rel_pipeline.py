#!/usr/bin/env python3
"""Phase 3 Example: Combined End-to-End NER → Relation Extraction Pipeline.

Demonstrates the full pipeline across 5 synthetic business scenarios,
including edge cases where NER succeeds but relation extraction fails or remains ambiguous.
"""

import json
import logging

try:
    from examples.synthetic_data import (
        SYNTHETIC_ENTITY_LABELS,
        SYNTHETIC_RELATION_LABELS,
        get_synthetic_samples,
    )
except ModuleNotFoundError:
    from synthetic_data import (
        SYNTHETIC_ENTITY_LABELS,
        SYNTHETIC_RELATION_LABELS,
        get_synthetic_samples,
    )
from structured_ie.pipeline.extraction_pipeline import ExtractionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    print("=" * 80)
    print("COMBINED NER → RELATION EXTRACTION PIPELINE (Phase 3)")
    print("=" * 80)

    # Initialize the complete end-to-end pipeline
    pipeline = ExtractionPipeline.from_pretrained()

    samples = get_synthetic_samples()

    for idx, sample in enumerate(samples, 1):
        desc = sample.metadata.get("description", f"Sample {idx}")
        difficulty = sample.metadata.get("difficulty", "standard")

        print("\n" + "=" * 80)
        print(f"Scenario {idx} [{difficulty.upper()}]: {desc}")
        print("=" * 80)
        print(f'Text:\n  "{sample.text}"')

        # Run extraction pipeline
        result = pipeline.extract(
            text=sample.text,
            entity_labels=SYNTHETIC_ENTITY_LABELS,
            relation_labels=SYNTHETIC_RELATION_LABELS,
            entity_threshold=0.35,
            relation_threshold=0.25,
        )

        print(f"\n[1] Extracted Entities ({len(result.entities)}):")
        for ent in result.entities:
            print(
                f"    - [{ent.label:13s}] {ent.text:25s} (score: {ent.score:.3f}, offsets: {ent.start}..{ent.end})"
            )

        print(f"\n[2] Extracted Relations ({len(result.relations)}):")
        if result.relations:
            for rel in result.relations:
                print(
                    f"    - '{rel.source}' --[{rel.relation}]--> '{rel.target}' (score: {rel.score:.3f})"
                )
        else:
            print("    (No relations met confidence threshold)")

        print("\n[3] Normalized Output JSON:")
        output_dict = {
            "entities": [
                {"text": e.text, "label": e.label, "score": round(e.score, 4)}
                for e in result.entities
            ],
            "relations": [
                {
                    "source": r.source,
                    "relation": r.relation,
                    "target": r.target,
                    "score": round(r.score, 4),
                }
                for r in result.relations
            ],
        }
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
