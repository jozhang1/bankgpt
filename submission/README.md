# Dupaco Knowledge System

This folder contains the submission implementation for the Knowledge Hackathon. The harness imports `solution.py`, calls `build()`, ingests the five snapshots oldest first, and then calls `answer()` and `conflicts()`.

## Submission contract

`solution.py` exposes `build()`, which returns an object with the required methods:

| Method | Behavior |
| --- | --- |
| `ingest(snapshot)` | Processes one supplied snapshot and adds its source-backed information to the in-memory store. |
| `answer(question, as_of=None)` | Answers from the newest eligible snapshot. With `as_of`, uses the newest snapshot taken on or before that date. |
| `conflicts()` | Returns source-backed disagreements found within and across snapshots. |

The public types and exact method signatures are defined by `harness/contract.py`. Keep the interface unchanged. The implementation uses only the provided snapshots; answer-time code does not fetch the live website.

## Approach

### Ingestion and history

Each snapshot remains a dated version. Pages are cleaned and parsed into structured blocks, grouped into retrieval evidence, and retained with their source URL, title, and snapshot provenance. Deterministic extraction builds normalized facts and observations when the source structure supports them; a rate stated as a whole line such as `6-month certificate APY: 3.50%` is first rewritten into a small table so the same extractor reads it (`ingestion/rate_lines.py`). Semantic extraction can add candidates for supported fact profiles when the OpenAI ingestion model is configured; candidates still pass source and entity validation before they are admitted.

Facts are identified by normalized entity, attribute, and qualifiers. Values, evidence, snapshot dates, and any source-supported effective dates belong to observations, so changes remain traceable without merging different products or conditions.

### Answer selection

The answer selector first considers source-backed Facts and applies product, metric, qualifier, snapshot, and effective-date checks. It renders only observations admitted by the answerability checks. If the Fact route cannot produce an admitted answer, it falls back to the evidence route.

Evidence retrieval combines BM25 with vectors from `sentence-transformers/all-MiniLM-L6-v2` using rank fusion. It sends up to 30 fused candidates to the local `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker; the answer path then gives up to five selected excerpts to the summary model. Both Hugging Face models run locally and can be overridden with `BANKGPT_VECTOR_MODEL` and `BANKGPT_RERANK_MODEL`. On the first index build, the libraries download missing models into the Hugging Face cache. To download them before running the harness (for example, before an offline run), install `requirements.txt` from the repository root, then run:

```bash
python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
PY
```

If a local model cannot load, retrieval falls back to the available ranking signals. The summary path validates its structured output and quote IDs before rendering; on failure, it uses the original retrieved excerpt instead of presenting an unvalidated summary as confirmed.

"Current" means the date of the selected snapshot, never the wall clock: without `as_of` the reference date is the newest snapshot's date, and with `as_of` it is that date. Relative expressions in a question ("last year", "now", "upcoming") are resolved against that reference date, and the evidence-summary prompt is given the same date. If answering raises an unexpected error, `answer()` logs it and returns an unconfident "couldn't find information" answer instead of failing the question.

For `as_of` questions, snapshot selection uses `Snapshot.taken_on`. An article publication date or snapshot date alone does not prove when a rate became effective or that an older offer ended. Unknown applicability remains unresolved instead of being guessed.

### Conflict reporting

Conflicts include differing observations of the same normalized Fact across snapshots. Different Facts are also compared when they share the same entity and metric and have distinct, overlapping, fully bounded numeric scopes defined by the qualifier schema (for example, overlapping model-year ranges). A catalog parent and child product may also be compared when the parent-side observation explicitly names the child in its structured source context; catalog lineage alone is not enough. One-sided caps/floors, disjoint ranges, and incompatible categorical conditions are not enough to merge them. Repeated copies of an assertion are collapsed while source excerpts are preserved. Source-quoted pending statements are compared with Facts and one another; these are reported only as potential conflicts and never receive an automatic winner. For an unscoped report, `resolution` is the final selected value when observations in the latest full-site snapshot agree. If they differ, a uniquely latest source-supported `effective_from` that has started by the snapshot date may resolve the value when no older interval explicitly extends through it (an assertion that starts only after the snapshot date is announced, not current). Source-role precedence is used only when effective dates do not establish a winner: a narrative page (a newsroom, blog, press, article or PDF URL) ranks below a `calculator`, which ranks below an `official_rate_table`. A winner is chosen only when every disagreeing observation has a known, strictly lower rank than the winning one; equal or unknown ranks leave `resolution` as `None`. Table and calculator roles are inferred from page/table structure and calculator UI signals; only the narrative rank is inferred from URL words, because a dated news item or PDF that happens to contain a rate table is still a description of a past rate, not the current schedule. Two dated, non-overlapping versions of one Fact (for example a rate that changed between snapshots) are reported as a conflict whose `resolution` is the later value and whose reason starts with `resolved_version_difference`; set `BANKGPT_REPORT_VERSION_CHANGES=0` to leave such version changes out. The reason explains the selection. If current evidence does not establish one value, `resolution` remains `None`. Snapshot-scoped reports do not use future versions to select a winner. The answer route separately uses source-supported effective dates when selecting a value for a question.

## OpenAI configuration

OpenAI is required for the intended full submission workflow: semantic ingestion and answer-time evidence summarization use the OpenAI API. There are no LLM enable/disable flags. Set `OPENAI_API_KEY` in the environment or repository-root `.env`. Both model settings default to `gpt-6-luna`; set them explicitly to override the defaults:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-6-luna"
export OPENAI_RETRIEVAL_MODEL="gpt-6-luna"
```

`OPENAI_API_KEY` enables the OpenAI calls. `OPENAI_MODEL` selects the ingestion model and defaults to `gpt-6-luna`; `OPENAI_RETRIEVAL_MODEL` selects the answer model and has the same default. Answer-time reasoning effort is set to `none`. The local `.env` is loaded without overriding variables supplied by the runner. Do not commit `.env` or include credentials in the submission archive.

During the configured run, page evidence is sent to OpenAI for semantic extraction, and selected evidence excerpts with the question are sent for answer summarization. These calls may incur API charges.

The OpenAI client permits one automatic retry for retryable transport failures. Schema, claim, or citation validation failures can trigger one application-level repair request that includes the previous response and validation errors. Retries may increase answer latency; the implementation does not claim a particular p95 latency.

## Configuration reference

Every environment variable the submission reads is defined in `foundation/config.py`; no other module reads the environment (a test enforces this). Values are read when used, so changing one takes effect without restarting the process.

| Variable | Default | Effect |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Enables the OpenAI calls. Supply it through the environment or the repository-root `.env`; never commit it. |
| `OPENAI_MODEL` | `gpt-6-luna` | Model for semantic ingestion. |
| `OPENAI_RETRIEVAL_MODEL` | `gpt-6-luna` | Model for answer-time evidence summarization. |
| `BANKGPT_MAX_OUTPUT_TOKENS` | `4096` | Output-token limit per OpenAI call. It is part of the semantic cache key, so changing it re-runs extraction. |
| `BANKGPT_SEMANTIC_CACHE_DIR` | `<repo root>/.cache/semantic` | Where semantic extraction results are cached. The cache is not committed. |
| `BANKGPT_VECTOR_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for vector retrieval. Retrieval falls back to BM25 if it cannot load. |
| `BANKGPT_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used to rerank retrieved evidence. |
| `BANKGPT_REPORT_VERSION_CHANGES` | `1` | `0` leaves dated version changes of one fact out of `conflicts()`; see Conflict reporting. |
| `BANKGPT_VERBOSE_INGEST` | `0` | `1` prints every extracted or skipped candidate and committed fact during ingestion. Progress summaries are always printed. |

## Install and run

Use Python 3.11, as in the assignment environment. From the repository root:

```bash
python -m pip install -r requirements.txt
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

The harness unpacks snapshots on the first run if needed, ingests all five in date order, asks the 40 public questions, and writes the report under `results/` by default. Use `--out <directory>` to choose another report directory. The public set is for practice; the assignment is graded on a larger hidden set, so do not hard-code answers to its question IDs.

## Scoring expectations

The hidden-set score weights accuracy at 55%, conflict reporting at 15%, consistency at 15%, and speed at 15%. A wrong or stale value scores zero for that question even if the correct value is also included, so returning every candidate is not a safe strategy. Source citations can earn additional accuracy credit. The speed score gives full credit when most answers take 2 seconds or less and falls to zero at 20 seconds; speed and consistency are multiplied by accuracy. Hidden questions include paraphrases, `as_of` dates, and changes between snapshots.

The contract requires each `answer()` call to finish within 60 seconds. Snapshot ingestion is not included in answer latency scoring, but the full ingestion run must finish within 30 minutes.

## Files

- `solution.py`: harness entry point and public interface.
- `foundation/`: shared contracts, store, product and entity definitions, and `config.py` (every environment variable).
- `ingestion/`: cleaning, block parsing, evidence chunking, and fact extraction.
- `facts/`: normalization, matching, temporal resolution, and conflict detection.
- `retrieval/`: evidence search, location lookup, query scope, and version selection.
- `answering/`: answerability checks, LLM adapter, and answer rendering.
- Repository root: `requirements.txt` (Python dependencies installed by the evaluator) and `NOTES.md` (system approach and required API configuration).
- `tests/` and `conftest.py`: unit and regression tests. Run `python -m pytest submission` from the repository root, or `python -m pytest` inside `submission/`; they are not needed by the evaluator.
