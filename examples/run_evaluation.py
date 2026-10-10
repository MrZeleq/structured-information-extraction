#!/usr/bin/env python3
"""Execute full evaluation of the Information Extraction pipeline on synthetic test cases.

Calculates Precision, Recall, and F1 for both Entity Extraction and Relation Extraction,
displays a summary table, and saves the detailed benchmark results to JSON.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from structured_ie.evaluation.dataset import load_dataset_from_json
from structured_ie.evaluation.evaluator import PipelineEvaluator
from structured_ie.pipeline.extraction_pipeline import ExtractionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluator")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation over labeled synthetic test cases."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_cases.json",
        help="Path to labeled test cases JSON file (default: data/synthetic_cases.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/latest_evaluation_report.json",
        help="Path to save evaluation output report JSON (default: data/latest_evaluation_report.json)",
    )
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for NER extraction (default: 0.25)",
    )
    parser.add_argument(
        "--relation-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for relation classification (default: 0.35)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Maximum relations predicted per entity pair (-1 for all, default: 1)",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable neuro-symbolic post-processing and schema constraint filtering (default: True)",
    )
    parser.add_argument(
        "--match-mode",
        type=str,
        default="exact",
        choices=["exact", "relaxed", "span"],
        help="Evaluation match mode: exact, relaxed, or span (default: exact)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="auto",
        choices=["auto", "standard", "broker_client"],
        help="Pipeline architecture: auto (detected from dataset), standard, or broker_client",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")

    print("=" * 70)
    print("INFORMATION EXTRACTION BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Dataset:            {dataset_path}")
    print(f"Entity Threshold:   {args.entity_threshold}")
    print(f"Relation Threshold: {args.relation_threshold}")
    print(f"Top-K per Pair:     {args.top_k}")

    # 1. Load labeled test samples
    samples = load_dataset_from_json(dataset_path)
    print(f"Loaded {len(samples)} evaluation scenarios.")

    # 2. Extract label universe from gold data
    entity_labels_set = set()
    relation_labels_set = set()
    for s in samples:
        for e in s.expected_entities:
            entity_labels_set.add(e.label)
        for r in s.expected_relations:
            relation_labels_set.add(r.relation)

    entity_labels = sorted(entity_labels_set)
    relation_labels = sorted(relation_labels_set)

    print(f"\nCandidate Entity Labels ({len(entity_labels)}):   {entity_labels}")
    print(f"Candidate Relation Labels ({len(relation_labels)}): {relation_labels}")

    # Resolve target pipeline mode
    pipeline_type = args.pipeline
    if pipeline_type == "auto":
        pipeline_type = (
            "broker_client"
            if ("broker" in entity_labels and "client" in entity_labels)
            else "standard"
        )
    print(f"Selected Pipeline:  {pipeline_type.upper()}\n")

    # 3. Instantiate extraction pipeline
    from structured_ie.ner.gliner_extractor import GLiNERExtractor
    from structured_ie.postprocessing.entity_cleaner import EntityPostprocessor

    ner = GLiNERExtractor()
    postprocessor = EntityPostprocessor() if args.postprocess else None

    if pipeline_type == "broker_client":
        from structured_ie.pipeline.broker_client_pipeline import BrokerClientPipeline
        from structured_ie.postprocessing.broker_client_resolver import BrokerClientResolver

        resolver = BrokerClientResolver()
        pipeline = BrokerClientPipeline(
            ner_extractor=ner,
            postprocessor=postprocessor,
            resolver=resolver,
        )
    else:
        from structured_ie.postprocessing.relation_filter import RelationFilter
        from structured_ie.relations.glirel_extractor import GLiRELExtractor

        re = GLiRELExtractor(top_k=args.top_k)
        rel_filter = RelationFilter() if args.postprocess else None
        pipeline = ExtractionPipeline(
            ner_extractor=ner,
            relation_extractor=re,
            postprocessor=postprocessor,
            relation_filter=rel_filter,
        )

    # Configure an extraction caller adapter that forwards command line thresholds
    class ThresholdPipelineAdapter:
        def __init__(self, inner_pipeline: Any) -> None:
            self.inner = inner_pipeline

        def extract(self, text: str, entity_labels: list[str], relation_labels: list[str]):
            if hasattr(self.inner, "extract"):
                return self.inner.extract(
                    text=text,
                    entity_labels=entity_labels,
                    relation_labels=relation_labels,
                    entity_threshold=args.entity_threshold,
                    relation_threshold=args.relation_threshold,
                )
            raise AttributeError("Pipeline missing extract method.")

    adapter = ThresholdPipelineAdapter(pipeline)

    # 4. Run benchmark evaluation
    evaluator = PipelineEvaluator(
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        match_mode=args.match_mode,
    )

    print("\nRunning evaluation across test cases...")
    result = evaluator.evaluate(pipeline=adapter, samples=samples)

    # 5. Display formatted report
    print("\n" + result.format_summary() + "\n")

    # 6. Save report to JSON file
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "dataset": str(dataset_path),
        "total_samples": result.total_samples,
        "entity_threshold": args.entity_threshold,
        "relation_threshold": args.relation_threshold,
        "metrics": {
            "entity_micro": result.entity_micro.to_dict(),
            "relation_micro": result.relation_micro.to_dict(),
        },
        "per_sample_results": result.per_sample_results,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    print(f"Full evaluation report saved to: {out_path}")


if __name__ == "__main__":
    main()
