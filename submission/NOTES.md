# Submission Notes

## How it works

The system ingests the supplied snapshots in order and preserves their dates and source URLs. It cleans and chunks pages into evidence, extracts normalized facts where supported, and links observations to their evidence.

Answers use the newest eligible snapshot by default, or the newest snapshot at or before `as_of`. The system prefers supported Fact selection. Otherwise, BM25 and vector retrieval find evidence, a local cross-encoder reranks candidates, and the answer model summarizes cited excerpts. Conflict reporting compares supported facts across snapshots and retains source-quoted pending statements as potential conflicts without selecting a winner for them.

## API configuration

The intended full workflow requires the OpenAI API. Configure these environment variables or add them to the repository-root `.env`:

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | API access for semantic ingestion and answer-time evidence summaries. |
| `OPENAI_MODEL` | Ingestion model; defaults to `gpt-6-luna`. |
| `OPENAI_RETRIEVAL_MODEL` | Answer-summary model; defaults to `gpt-6-luna`. |

The local embedding and reranking models do not require API keys. They are downloaded from Hugging Face on first use if they are not already cached.

To download both models before running (for example, before an offline run), install `submission/requirements.txt` and run this from the repository root:

```bash
python - <<'PY'
from sentence_transformers import CrossEncoder, SentenceTransformer

SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", max_length=512)
PY
```

Do not include `.env` or API credentials in the submission archive.
