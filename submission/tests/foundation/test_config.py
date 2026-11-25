import os
import unittest
from pathlib import Path
from unittest.mock import patch

from foundation import config


class ConfigTests(unittest.TestCase):

    def test_defaults_when_nothing_is_set(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(config.report_version_changes())
            self.assertFalse(config.verbose_ingest())
            self.assertFalse(config.openai_api_key_present())
            self.assertEqual(config.openai_model_override(config.OPENAI_MODEL_ENV), '')
            self.assertEqual(config.max_output_tokens(), 4096)
            self.assertEqual(config.vector_model_name(), config.DEFAULT_VECTOR_MODEL)
            self.assertEqual(config.rerank_model_name(), config.DEFAULT_RERANK_MODEL)
            cache = Path(config.semantic_cache_dir())
            self.assertEqual(cache.parts[-2:], ('.cache', 'semantic'))

    def test_values_are_read_when_called_not_at_import(self):
        with patch.dict(os.environ, {'BANKGPT_REPORT_VERSION_CHANGES': '0',
                                     'BANKGPT_VERBOSE_INGEST': '1',
                                     'BANKGPT_MAX_OUTPUT_TOKENS': '1000',
                                     'BANKGPT_SEMANTIC_CACHE_DIR': '/tmp/x',
                                     'OPENAI_API_KEY': ' k ',
                                     'OPENAI_MODEL': ' m '}, clear=True):
            self.assertFalse(config.report_version_changes())
            self.assertTrue(config.verbose_ingest())
            self.assertEqual(config.max_output_tokens(), 1000)
            self.assertEqual(config.semantic_cache_dir(), '/tmp/x')
            self.assertTrue(config.openai_api_key_present())
            self.assertEqual(config.openai_model_override('OPENAI_MODEL'), 'm')

    def test_only_the_config_module_reads_the_environment(self):
        root = Path(__file__).resolve().parents[2]
        offenders = []
        for path in root.rglob('*.py'):
            if 'tests' in path.parts or path.name in {'config.py', 'conftest.py'}:
                continue
            text = path.read_text()
            if 'os.environ' in text or 'os.getenv' in text:
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
