# Technical Overview

## Research scope

This project investigates a hybrid information extraction approach for Polish business text, with insurance brokerage authorizations as its main application. It combines pretrained models for recognizing entity mentions with explicit rules for identifier normalization, relation constraints, and business role assignment.

The research contribution is the experimental integration and analysis of these components for a specific domain. The repository does not introduce a new neural architecture or train a new checkpoint.

The central distinction is between an entity's type and its role: both a broker and a client can be companies, while their roles depend on the surrounding passage. The specialized pipeline therefore detects company mentions first and resolves roles afterwards.

## Model interfaces and configuration

| Component | Default | Interface |
| :--- | :--- | :--- |
| Entity extraction | `urchade/gliner_multi-v2.1` | `BaseNERExtractor.extract_entities()` |
| Relation extraction | `jackboyla/glirel-large-v0` | `BaseRelationExtractor.extract_relations()` |
| Runtime | CUDA when available, otherwise CPU | `PipelineConfig` and constructor arguments |

The default model names can also be set through `DEFAULT_GLINER_MODEL` and `DEFAULT_GLIREL_MODEL`. Models are loaded from pretrained weights and run locally. There is no task-specific training step in the repository.

Each wrapper accepts a preloaded model for dependency injection. New model wrappers can implement the same abstract interfaces, allowing pipeline components to be compared without changing the evaluation schema.

## General extraction flow

[`ExtractionPipeline`](../src/structured_ie/pipeline/extraction_pipeline.py) executes:

1. GLiNER entity detection using caller-supplied labels and a confidence threshold.
2. Optional entity cleanup.
3. GLiREL relation prediction if at least two entities and relation labels are available.
4. Optional schema-based relation filtering.
5. Construction of an `ExtractionResult`.

`from_pretrained(enable_postprocessing=True, schema=...)` enables both cleanup and relation filtering. For custom component injection, instantiate `ExtractionPipeline(ner_extractor=..., relation_extractor=..., postprocessor=..., relation_filter=...)` directly.

### Alignment

GLiNER supplies character offsets, whereas GLiREL consumes token-index spans. [`alignment.py`](../src/structured_ie/pipeline/alignment.py) tokenizes with offsets, maps entity spans to token spans, and converts predictions back into the shared relation format. The GLiREL wrapper passes tokens rather than raw text and includes a pretrained-loading compatibility patch.

### Cleanup and constraints

`EntityPostprocessor` removes selected company introductions and identifier prefixes, discards short or digit-free identifier candidates, and reclassifies identifiers by digit count. A context heuristic filters candidates associated with KRS labels.

`RelationFilter` verifies that both endpoints correspond to entities and satisfy the configured source/target types. For predicates with `functional_target=True`, it keeps the highest-scoring relation per target string. Although the schema also exposes `functional_source`, the current filter does not enforce it.

The default schema contains `has NIP`, `has REGON`, `broker for`, and `works for`. Unknown predicates pass the type-constraint check. Endpoint lookup uses surface text, so repeated identical mentions are not fully distinguished by this filter.

## Broker–client extraction flow

[`BrokerClientPipeline`](../src/structured_ie/pipeline/broker_client_pipeline.py) specializes the process:

1. Detect `company` and `NIP` with GLiNER; the default entity threshold is `0.25`.
2. Add pattern-based NIP candidates when nearby model spans do not already cover them.
3. Apply entity cleanup.
4. Bind NIPs to companies and assign broker/client roles through `BrokerClientResolver`.
5. Return either entities and relations or structured parties with validation metadata.

This path does not use GLiREL. Its optional label and relation-threshold arguments exist for compatibility with the shared evaluator; they do not configure neural relation prediction.

### Identifier binding

The resolver attempts three passes:

- Attach a NIP shortly after a company when punctuation or a NIP label suggests a local association.
- Search registration text after a company, with a default maximum distance of 350 characters, or a short preceding-identifier pattern. Intervening companies block these attachments.
- Assign the nearest remaining eligible NIP as a fallback.

The last pass does not apply the 350-character registration limit. Bindings are greedy and keyed by company surface text, so they are not a global optimization over distinct legal entities.

### Role assignment

The resolver scores generic brokerage terms, local role markers, and preceding/following phrases. It can accept a caller-supplied `broker_registry` collection of name fragments; it does not fetch an external registry.

For two or more company candidates, only the first two are compared. The company with the larger broker-minus-client score becomes the broker, and the other becomes the client. For one company, the larger of its two role scores determines the role, with ties assigned to broker.

These are explicit heuristics. Role assignment is not equivalent to resolving legal status or establishing an active authorization, and ambiguous documents can still receive a pair of role labels.

## Normalization and validation

[`TextNormalizer`](../src/structured_ie/normalization/text_normalizer.py) provides NIP candidate detection, digit normalization, and a separate utility for joining words split across hyphenated line breaks. Only candidate detection is integrated into the specialized pipeline; neither pipeline automatically runs full-text line-break cleanup.

[`checksums.py`](../src/structured_ie/normalization/checksums.py) contains NIP and REGON checksum utilities and a KRS digit-format check. KRS is not checksum-validated by this code. Structured broker/client extraction invokes NIP validation; the general pipeline does not automatically apply all available validation utilities.

Normalized identifier text may omit separators that remain in the original document. Offsets are intended to locate the source region, but exact character-slice equality is not a universal invariant. Formatted identifiers require targeted span validation.

## Output contract

| Output | Contents |
| :--- | :--- |
| `Entity` | Text, label, score, start/end character offsets |
| `Relation` | Source text, predicate, target text, score, optional endpoint spans |
| `ExtractionResult` | Input text, entities, relations |
| `BrokerClientExtraction` | Optional broker/client, `confidence`, `needs_review`, `review_reasons` |
| `BrokerParty` / `ClientParty` | Name, NIP, spans, entity score, checksum status, raw NIP text |

`extract()` returns the shared entity/relation representation. `extract_structured()` also validates NIPs and prepares review metadata. It flags failed NIP checksums, missing parties, and a role-score margin below `1.5`. A missing NIP adds a reason but does not, by itself, set `needs_review=True` in the current implementation.

Scores have different meanings: party scores originate from NER, generated relations receive a fixed `1.0`, regex-injected candidates receive `0.99`, and aggregate `confidence` is `1.0` or `0.75` according to the review flag. These values are not calibrated probabilities and should not be compared as if they were.

## Evaluation and extension

`PipelineEvaluator` accepts objects exposing `extract(text, entity_labels, relation_labels)` and aggregates entity and relation TP/FP/FN counts into micro precision, recall, and F1. The CLI adds thresholds and chooses a pipeline explicitly or from the dataset labels.

See [experiments](experiments.md) for matching semantics, committed reports, and reproducibility limits. Future work includes independently held-out documents, repeated-mention resolution, multi-party assignment, calibrated abstention, full-text offset tracking, and document chunking.
