"""Versioned model-result cache, never a cache of Evidence/Fact/Observation IDs."""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path

from foundation import config
from tempfile import NamedTemporaryFile


class SemanticModelCache:
    VERSION = 'semantic-model-cache-v1'

    def __init__(self, caller, *, directory=None, namespace=None):
        proposed_namespace = namespace or getattr(caller, 'model', None)
        self.namespace = proposed_namespace if isinstance(proposed_namespace, str) and proposed_namespace.strip() else None
        # Unnamed injected callers only share within this cache instance.
        self.directory = Path(directory) if directory is not None and self.namespace else None
        self.memory = {}
        self.hits = 0
        self.misses = 0

    def key(self, request):
        payload = {'version': self.VERSION, 'model': self.namespace,
                   'request': request,
                   'max_output_tokens': config.max_output_tokens_setting()}
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':')).encode()).hexdigest()

    def get(self, key):
        if key in self.memory:
            self.hits += 1
            return deepcopy(self.memory[key])
        if self.directory is not None:
            try:
                record = json.loads((self.directory / (key + '.json')).read_text())
                if record.get('key') == key and record.get('version') == self.VERSION:
                    response = record['response']
                    if isinstance(response, dict):
                        self.memory[key] = response
                        self.hits += 1
                        return deepcopy(response)
            except (OSError, ValueError, KeyError, TypeError):
                pass  # Corrupt/unavailable cache is a miss, never source approval.
        self.misses += 1
        return None

    def discard(self, key):
        self.memory.pop(key, None)
        if self.directory is not None:
            try:
                (self.directory / (key + '.json')).unlink(missing_ok=True)
            except OSError:
                pass

    def put(self, key, response):
        # Only callers that completed schema/scope validation may write here.
        self.memory[key] = deepcopy(response)
        if self.directory is None:
            return
        temporary = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            with NamedTemporaryFile(mode='w', dir=self.directory, suffix='.tmp',
                                    encoding='utf-8', delete=False) as output:
                temporary = Path(output.name)
                json.dump({'version': self.VERSION, 'key': key, 'response': response},
                          output, ensure_ascii=False)
            os.replace(temporary, self.directory / (key + '.json'))
        except OSError:
            pass  # Cache writes cannot invalidate otherwise valid ingestion.
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
