"""Every environment variable the submission reads, in one place.

Values are read when the function is called, not at import time, so a process
(or a test) can change the environment between calls. The README lists these
variables with their defaults; keep the two in sync.

Credentials are never defaults and never live in submission files:
OPENAI_API_KEY comes from the runtime environment (or the kit-root .env).
"""

from __future__ import annotations

import os
from pathlib import Path

# Model defaults.
DEFAULT_OPENAI_MODEL = 'gpt-6-luna'
DEFAULT_VECTOR_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
DEFAULT_MAX_OUTPUT_TOKENS = '4096'

# Names of variables that other modules need to report in error messages.
OPENAI_MODEL_ENV = 'OPENAI_MODEL'
OPENAI_RETRIEVAL_MODEL_ENV = 'OPENAI_RETRIEVAL_MODEL'
MAX_OUTPUT_TOKENS_ENV = 'BANKGPT_MAX_OUTPUT_TOKENS'


def _text(name: str, default: str = '') -> str:
    return os.environ.get(name, default).strip()


def openai_api_key_present() -> bool:
    return bool(_text('OPENAI_API_KEY'))


def openai_model_override(name: str) -> str:
    """The model ID set in environment variable ``name`` ('' when unset)."""
    return _text(name)


def max_output_tokens_setting() -> str:
    """Raw BANKGPT_MAX_OUTPUT_TOKENS text; part of the semantic cache key."""
    return os.environ.get(MAX_OUTPUT_TOKENS_ENV, DEFAULT_MAX_OUTPUT_TOKENS)


def max_output_tokens() -> int:
    return int(max_output_tokens_setting())


def vector_model_name() -> str:
    return os.environ.get('BANKGPT_VECTOR_MODEL', DEFAULT_VECTOR_MODEL)


def rerank_model_name() -> str:
    return os.environ.get('BANKGPT_RERANK_MODEL', DEFAULT_RERANK_MODEL)


def semantic_cache_dir() -> str:
    default = Path(__file__).resolve().parents[2] / '.cache' / 'semantic'
    return os.environ.get('BANKGPT_SEMANTIC_CACHE_DIR', str(default))


def report_version_changes() -> bool:
    """Report dated version changes of one Fact as conflicts (on by default)."""
    return _text('BANKGPT_REPORT_VERSION_CHANGES', '1') != '0'


def verbose_ingest() -> bool:
    """Print every extracted/skipped candidate and committed fact while ingesting."""
    return _text('BANKGPT_VERBOSE_INGEST') == '1'
