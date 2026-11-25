"""Optional OpenAI Responses adapter; credentials are supplied by the environment."""
from copy import deepcopy
import json
from pathlib import Path

from foundation import config


def strict_output_schema(schema: dict) -> dict:
    """Require every object property on the wire without mutating local models."""
    result = deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            node.pop('default', None)
            if node.get('type') == 'object':
                node['required'] = list(node.get('properties', {}))
                node['additionalProperties'] = False
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(result)
    return result


class OpenAIResponseNotCompleted(ValueError):
    """Safe response diagnostics; never include raw output or server messages."""
    def __init__(self, response):
        def field(obj, key):
            return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
        status = field(response, 'status')
        reason = field(field(response, 'incomplete_details'), 'reason')
        code = field(field(response, 'error'), 'code')
        self.status = status if status in {'incomplete', 'failed', 'cancelled', 'queued', 'in_progress'} else 'unknown'
        self.reason = reason if reason in {'max_output_tokens', 'max_messages', 'content_filter', 'steered'} else 'unknown'
        safe_codes = {'server_error', 'rate_limit_exceeded', 'invalid_prompt', 'vector_store_timeout'}
        self.error_code = code if code in safe_codes else 'unknown'
        usage = field(response, 'usage')
        tokens = field(usage, 'output_tokens')
        self.output_tokens = tokens if type(tokens) is int else None
        super().__init__(
            f'OpenAI response not completed: status={self.status}, reason={self.reason}, '
            f'error_code={self.error_code}, output_tokens={self.output_tokens}. '
            'No partial JSON was accepted.'
        )


class OpenAICaller:
    def __init__(self, model: str, *, reasoning_effort: str | None = None,
                 temperature: float | None = None, client=None):
        if not model.strip():
            raise ValueError('OPENAI_MODEL must name a Responses structured-output model.')
        if reasoning_effort not in {None, 'none', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('Unsupported OpenAI reasoning effort.')
        if temperature is not None and not 0 <= temperature <= 2:
            raise ValueError('Temperature must be between 0 and 2.')
        self.model = model.strip()
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        if client is None:
            # Lazy import keeps the deterministic submission usable without SDK.
            from openai import OpenAI
            client = OpenAI(max_retries=1, timeout=16)
        self.client = client

    def __call__(self, request: dict, timeout_seconds: float) -> dict:
        max_output_tokens = config.max_output_tokens()
        if max_output_tokens <= 0:
            raise ValueError('BANKGPT_MAX_OUTPUT_TOKENS must be positive.')
        options = {}
        if self.reasoning_effort is not None:
            options['reasoning'] = {'effort': self.reasoning_effort}
        if self.temperature is not None:
            options['temperature'] = self.temperature
        response = self.client.with_options(timeout=timeout_seconds, max_retries=1).responses.create(
            model=self.model,
            instructions=request['instructions'],
            input=json.dumps(request['input'], ensure_ascii=False),
            text={'format': {'type': 'json_schema', 'name': 'evidence_group_selection',
                             'strict': True, 'schema': strict_output_schema(request['output_schema'])}},
            max_output_tokens=max_output_tokens,
            store=False,
            **options,
        )
        if response.status != 'completed':
            raise OpenAIResponseNotCompleted(response)
        if any(getattr(content, 'type', None) == 'refusal'
               for item in response.output for content in getattr(item, 'content', ())):
            raise ValueError('OpenAI declined the evidence request.')
        if not response.output_text:
            raise ValueError('OpenAI response contained no JSON output.')
        result = json.loads(response.output_text)
        if not isinstance(result, dict):
            raise ValueError('OpenAI output must be a JSON object.')
        # Group schema and admission are checked by the existing reasoner/validator.
        return result


def load_local_environment():
    """Load only the kit-root .env; never override evaluator-injected secrets."""
    path = Path(__file__).resolve().parents[2] / '.env'
    if path.is_file():
        from dotenv import load_dotenv
        load_dotenv(path, override=False, interpolate=False)

def _caller_from_environment(*, model_env: str, default_model: str | None = None,
                             reasoning_effort: str | None = None,
                             temperature: float | None = None):
    load_local_environment()
    key_present = config.openai_api_key_present()
    configured_model = config.openai_model_override(model_env)
    model = configured_model or (default_model or '')
    if not key_present and not configured_model:
        return None  # No online model configured: preserve deterministic behavior.
    if not key_present:
        raise ValueError('Set OPENAI_API_KEY in the runtime environment, not in submission files.')
    if not model:
        raise ValueError(f'Set {model_env} to the chosen model ID.')
    return OpenAICaller(
        model, reasoning_effort=reasoning_effort, temperature=temperature,
    )


def caller_from_environment():
    """Build the ingestion caller, defaulting to the configured project model."""
    return _caller_from_environment(
        model_env='OPENAI_MODEL',
        default_model='gpt-6-luna',
    )


def retrieval_caller_from_environment():
    """Enable answer review by default, falling back if setup is unavailable."""
    try:
        return _caller_from_environment(
            model_env='OPENAI_RETRIEVAL_MODEL',
            default_model='gpt-6-luna',
            reasoning_effort='none',
            temperature=0.1,
        )
    except Exception:
        # Missing credentials, SDK, or invalid client configuration must not
        # prevent the deterministic answer path from starting.
        return None
