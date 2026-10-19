# Experiments: Zero-Shot Information Extraction in Polish Brokerage Text

## Study objective

The experiments examine whether pretrained generalist extraction models can support a narrow Polish document understanding task without domain-specific fine-tuning, and how deterministic domain knowledge changes their outputs.

The study separates three decisions: detecting entity mentions, assigning identifiers to companies, and assigning companies document-dependent roles. It compares a generic neural pipeline with post-processing variants and a specialized broker–client resolver.

This is an exploratory engineering study using small constructed datasets. The reported scores describe the included examples and provide hypotheses for subsequent evaluation on independently annotated documents.

## Datasets

| Dataset | Size | Target outputs | Purpose |
| :--- | ---: | :--- | :--- |
| [`synthetic_cases.json`](../data/synthetic_cases.json) | 20 | `company`, `NIP`, `REGON`; `has NIP`, `has REGON` | Generic extraction and post-processing comparison |
| [`broker_client_cases.json`](../data/broker_client_cases.json) | 20 | `broker`, `client`, `NIP`; `has NIP`, `broker for` | Specialized role and identifier assignment |
| [`production_stress_cases.json`](../data/production_stress_cases.json) | 6 | Expected parties, identifiers, and selected validation outcomes | Constructed edge scenarios |

The first two datasets contain text with annotated entities and relations. The stress file uses a different schema with fields such as `expected_broker` and `expected_client`; it is not a standard `LabeledSample` dataset. The historical filename does not imply that these six passages form a production-document benchmark.

The examples were used during development. The repository does not establish an independent held-out split, document sampling protocol, annotator agreement, or confidence intervals. Some example identifiers serve as extraction targets even when their checksums are invalid. Entity/relation correctness and identifier validity are distinct evaluation questions.

## Experimental configurations

### EXP-001: Raw generic baseline

GLiNER detects the generic entity labels and GLiREL predicts relations. Entity cleanup and schema filtering are disabled. This configuration measures the behavior of the model combination before domain-oriented correction.

### EXP-002: Cleanup and schema constraints

The same generic task adds entity cleanup and relation filtering. Cleanup removes selected prefixes and digit-free identifier predictions and reclassifies identifiers by digit length. The relation filter rejects invalid endpoint types and resolves competing sources for functional targets.

The comparison with EXP-001 evaluates the combined post-processing stage. It does not isolate the causal contribution of each individual rule.

### EXP-003: Specialized broker–client resolution

GLiNER detects `company` and `NIP`. A deterministic resolver binds identifiers and assigns broker/client roles using local cues. This replaces GLiREL for the specialized task.

This experiment uses a different dataset and output schema from EXP-001/002. Its score should not be interpreted as a direct improvement on the same benchmark.

### EXP-004: Robustness-oriented extensions

The current specialized implementation adds pattern-based identifier candidates, registration-block binding, generic brokerage terms, optional caller-supplied registry fragments, checksum validation, and review metadata.

EXP-003 and EXP-004 are historical development stages. The current CLI runs the current implementation; it does not expose a switch to reconstruct each earlier resolver version. The repository does not contain a separate machine-readable report that isolates the effect of all EXP-004 additions.

## Metrics and matching semantics

The evaluator sums per-sample true positives, false positives, and false negatives before calculating micro precision, recall, and F1:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

The implementation handles empty predictions and empty reference sets explicitly. See [`metrics.py`](../src/structured_ie/evaluation/metrics.py).

| Mode | Entities | Relations |
| :--- | :--- | :--- |
| `exact` | Set equality of `(text, label)` after trimming outer whitespace and lowercasing | Set equality of `(source, predicate, target)` with the same normalization |
| `span` | Set equality of `(start, end, label)` | Same triplet-text matching as `exact`; relation spans are not evaluated |
| `relaxed` | Same label plus matching text, substring containment, or overlapping spans | Same predicate plus matching/contained source and target text |

Exact matching collapses repeated identical strings within a sample. Relaxed matching greedily assigns each prediction to at most one reference and is more permissive about boundaries. These scores should not be described as interchangeable measures of semantic correctness.

The standard evaluator calls `extract()`. It does not measure checksum decisions, `needs_review`, or the correctness of review explanations from `extract_structured()`. Those require separate assertions or a dedicated evaluation protocol.

## Recorded results

### Historical generic baseline and exact comparison

The following values are retained from the earlier experiment log. Standalone JSON reports for these two exact runs are not present in the current repository snapshot.

| Configuration | Task | Precision | Recall | F1 | TP / FP / FN |
| :--- | :--- | ---: | ---: | ---: | :--- |
| EXP-001: raw | Entities | 37.18% | 46.03% | 41.13% | 29 / 49 / 34 |
| EXP-001: raw | Relations | 7.69% | 16.67% | 10.53% | 6 / 72 / 30 |
| EXP-002: cleanup + constraints | Entities | 89.39% | 93.65% | 91.47% | 59 / 7 / 4 |
| EXP-002: cleanup + constraints | Relations | 51.43% | 50.00% | 50.70% | 18 / 17 / 18 |

The recorded entity F1 difference is 50.34 percentage points and the relation F1 difference is 40.17 points. This supports further investigation of the combined correction stage. Full controlled replication requires the original run configuration and dependency/checkpoint revisions.

### Results with committed JSON artifacts

| Dataset and matching | Task | Precision | Recall | F1 | TP / FP / FN |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Generic, relaxed | Entities | 95.45% | 100.00% | 97.67% | 63 / 3 / 0 |
| Generic, relaxed | Relations | 71.43% | 69.44% | 70.42% | 25 / 10 / 11 |
| Broker–client, exact | Entities | 100.00% | 100.00% | 100.00% | 80 / 0 / 0 |
| Broker–client, exact | Relations | 100.00% | 100.00% | 100.00% | 60 / 0 / 0 |

Evidence:

- [`relaxed_evaluation_report.json`](../data/relaxed_evaluation_report.json): generic dataset, entity threshold `0.25`, relation threshold `0.25`.
- [`broker_client_evaluation_report.json`](../data/broker_client_evaluation_report.json): specialized dataset, entity threshold `0.25`, recorded relation threshold `0.35` (unused by the resolver).
- [`broker_client_evaluation_report_relaxed.json`](../data/broker_client_evaluation_report_relaxed.json): records the same aggregate counts for the specialized dataset.
- [`latest_evaluation_report.json`](../data/latest_evaluation_report.json): currently also contains the specialized dataset's aggregate results. The CLI uses this as its default output and can overwrite it.

The JSON artifacts contain per-sample predictions and expected outputs. They do not store a complete run manifest: matching mode, checkpoint revision, dependency versions, code commit, device, and all CLI options are missing. Matching-mode labels above follow the historical log and filenames. The generic relaxed report also records a different relation threshold from the CLI's current default (`0.35`), so boundary tolerance alone cannot be assumed to explain every difference between historical exact and relaxed scores.

The specialized result means that the saved predictions match all annotated entity texts/labels and relation triplets on those 20 passages. It does not establish exact span correctness, reliable abstention, or 100% accuracy on unseen documents.

## Qualitative observations and hypotheses

Earlier experiment notes describe several recurring error types:

| Observation | Implemented response | Remaining question |
| :--- | :--- | :--- |
| Identifier predictions include prefixes such as `NIP:` | Extract digits and adjust the candidate span | Does the correction preserve offsets for all formatted identifiers? |
| Label words are predicted as identifiers without digits | Remove digit-free candidates | What false positives remain in tables or noisy OCR? |
| NIP and REGON labels are confused | Reclassify by digit count | How often are other numeric fields misclassified as identifiers? |
| GLiREL associates an identifier with the wrong company | Apply schema filtering; use structural binding in the specialized path | Which syntax and document layouts defeat the binding rules? |
| Direct broker/client labeling mixes roles or captures role words | Detect companies first and score their roles afterwards | How well does this transfer to unfamiliar phrasing and more than two companies? |

These observations motivate the architecture. They do not establish a measured causal explanation of model internals, pretraining coverage, or tokenization limitations. Such explanations would require additional controlled analysis.

## Constructed stress scenarios

| Scenario | Intended check |
| :--- | :--- |
| Long registration block with address, KRS, and REGON | Associate the company with its NIP without using a nearby KRS candidate |
| Brokerage firm without a brokerage term in its name | Use local role context |
| Spaced and hyphenated NIPs | Recover normalized identifiers |
| OCR-like `N1P:` label | Supplement model detection with pattern candidates |
| Corrupted NIP checksum | Retain the extraction and flag invalidity in the structured result |
| Client-only passage | Return a missing broker and request review |

The earlier log reported six successful scenarios. There is no committed per-case stress evaluation report or dedicated runner that reproduces that aggregate claim. Component tests cover related behaviors, but they are not equivalent to running all six passages through the pretrained model. The demo script contains three illustrative passages and does not evaluate the entire stress file.

Do not pass the stress file directly to `run_evaluation.py`: its schema lacks the standard `expected_entities` and `expected_relations`, whose defaults are empty lists. That would not measure the intended expected outcomes.

## Re-running the current implementation

Install the package as described in the [README](../README.md), then run from the repository root. Use distinct output paths to retain each run's evidence.

```bash
# Generic raw baseline
python examples/run_evaluation.py --dataset data/synthetic_cases.json --pipeline standard --no-postprocess --entity-threshold 0.25 --relation-threshold 0.35 --top-k 1 --match-mode exact --output data/reproduced_baseline.json

# Generic correction stage, same thresholds
python examples/run_evaluation.py --dataset data/synthetic_cases.json --pipeline standard --postprocess --entity-threshold 0.25 --relation-threshold 0.35 --top-k 1 --match-mode exact --output data/reproduced_generic_exact.json

# Generic relaxed configuration using the saved report's thresholds
python examples/run_evaluation.py --dataset data/synthetic_cases.json --pipeline standard --postprocess --entity-threshold 0.25 --relation-threshold 0.25 --top-k 1 --match-mode relaxed --output data/reproduced_generic_relaxed.json

# Current specialized resolver
python examples/run_evaluation.py --dataset data/broker_client_cases.json --pipeline broker_client --entity-threshold 0.25 --match-mode exact --output data/reproduced_broker_client.json
```

These commands define new runs of the current code, not guaranteed reproductions of every historical score. The original generic exact run configuration is incompletely recorded. Dependencies have lower bounds rather than a committed lockfile, and checkpoint revisions are not pinned.

For a controlled comparison, save the code commit, checkpoint revisions, dependency versions, device, complete CLI arguments, matching mode, and dataset hash alongside each report. Keep thresholds fixed when isolating the effect of post-processing or matching criteria.

`--no-postprocess` disables corrections in the generic pipeline. It does not reconstruct a raw specialized baseline: `BrokerClientPipeline` installs its default cleaner when passed `None`, and its resolver remains active.

## Threats to validity and next experiments

1. **Development-set bias:** freeze a held-out test set before modifying rules, with splits by document source or template rather than individual sentences.
2. **Limited task coverage:** add negative examples, repeated mentions, additional companies, unrelated identifiers, and ambiguous or revoked authorizations.
3. **Combined interventions:** disable identifier cleanup, candidate augmentation, binding passes, and role cues separately to measure each component.
4. **Incomplete output evaluation:** report character-span accuracy, party-level correctness, checksum outcomes, and review/abstention precision and recall separately.
5. **Missing comparator families:** compare with regex-only extraction, supervised Polish encoders, and generative structured extraction under a shared protocol.
6. **Document-scale uncertainty:** evaluate OCR-derived text, multi-page contracts, section selection, and cross-chunk entity resolution.
7. **Unmeasured operating cost:** benchmark runtime and memory on specified hardware with cold loading and warm inference reported separately.

These are proposed extensions. No results for supervised encoders, generative baselines, multi-party resolution, or long-document chunking are claimed by the current study.
