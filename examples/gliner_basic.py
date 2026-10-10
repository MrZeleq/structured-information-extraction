#!/usr/bin/env python3
"""Phase 1 Example: Zero-shot Named Entity Recognition using GLiNER.

Extracts companies, persons, NIP tax identifiers, and policy numbers
from synthetic business text.
"""

import json
import logging

from structured_ie.ner.gliner_extractor import GLiNERExtractor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    print("=" * 70)
    print("GLiNER Zero-Shot Named Entity Recognition (Phase 1)")
    print("=" * 70)

    # Initialize GLiNER extractor (uses default 'urchade/gliner_multi-v2.1')
    extractor = GLiNERExtractor()

    text = (
        "Spółka Alfa Logistyka S.A. o numerze NIP 5252525252 zawarła polisę "
        "POL-2024-88412 za pośrednictwem brokera Quantum Re. "
        "Głównym koordynatorem ze strony ubezpieczonego jest Tomasz Kamiński."
    )

    labels = [
        "company",
        "person",
        "NIP",
        "policy number",
    ]

    print("\nInput Text:")
    print(f"  {text}")
    print(f"\nTarget Entity Labels: {labels}\n")

    # Extract entities (threshold 0.25 captures zero-shot numeric identifiers like NIP)
    entities = extractor.extract_entities(text=text, labels=labels, threshold=0.25)

    print(f"Extracted {len(entities)} entities:")
    for ent in entities:
        print(
            f"  [{ent.label:13s}] {ent.text:25s} (score: {ent.score:.4f}, span: {ent.start}..{ent.end})"
        )

    print("\nNormalized JSON Representation:")
    normalized_list = [e.to_dict() for e in entities]
    print(json.dumps(normalized_list, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
