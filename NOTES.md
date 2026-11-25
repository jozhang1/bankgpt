# Submission Notes

## How it works

The solution lives in `submission/` (`submission/solution.py`, `build()`); `NOTES.md` and `requirements.txt` are at the repository root.

The system ingests the supplied snapshots in order and preserves their dates and source URLs. It cleans and chunks pages into evidence, extracts normalized facts where supported, and links observations to their evidence.

Answers use the newest eligible snapshot by default, or the newest snapshot at or before `as_of`. The system prefers supported Fact selection. Otherwise, BM25 and vector retrieval find evidence, a local cross-encoder reranks candidates, and the answer model summarizes cited excerpts. Conflict reporting compares supported facts across snapshots and retains source-quoted pending statements as potential conflicts without selecting a winner for them.

## How we decide what is current

"Current" is the date of the newest snapshot (28 Sep 2026), never the wall clock. With `as_of`, it is the newest snapshot taken on or before that date, and an `as_of` later than the newest snapshot uses the newest snapshot. Relative wording in a question ("now", "last year", "upcoming") is resolved against that same date.

- **Pages that disappear:** each snapshot is a full site pull, so a page missing from the newest snapshot is treated as gone and is not answered from an older snapshot.
- **Several values for one fact:** values in the selected snapshot are compared first. If they differ, the one with the latest source-stated effective date that has already started by the snapshot date wins. A value that starts after the snapshot date is announced, not current.
- **Ties:** when dates do not decide, source type does: an official rate table beats a calculator, which beats a news, blog or PDF page. If that does not decide either, the conflict is reported without a winner.
- **Conflicts:** `conflicts()` lists facts whose values differ across snapshots or pages. A rate that simply changed between snapshots is reported with the later value as its resolution.

Answers come only from the supplied snapshots; there are no live website calls. Each answer is cited with the source pages it used.

## API configuration

**This solution uses a paid API (OpenAI).** The intended full workflow requires the OpenAI API. Configure these environment variables or add them to the repository-root `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API access for semantic ingestion and answer-time evidence summaries. |
| `OPENAI_MODEL` | Ingestion model; defaults to `gpt-6-luna`. |
| `OPENAI_RETRIEVAL_MODEL` | Answer-summary model; defaults to `gpt-6-luna`. |

Ingestion makes OpenAI calls for semantic fact extraction, and answers that go through retrieval make an OpenAI call to summarize the evidence. The local embedding and reranking models do not require API keys. They are downloaded from Hugging Face on first use if they are not already cached. Packages are listed in `requirements.txt` (Python 3.11).

To download both models before running (for example, before an offline run), install `requirements.txt` and run this from the repository root:

```bash
python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
PY
```

`submission/README.md` has the full configuration table (cache location, model overrides, conflict and ingestion switches); every variable is read in `submission/foundation/config.py`.

Do not include `.env` or API credentials in the submission archive.
