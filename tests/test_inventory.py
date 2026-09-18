"""Проверки возобновления и ошибок без сетевых запросов и расхода OpenRouter."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, ExitStack
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inventory import core


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pairs = self.root / 'ru_language_pairs'
        self.udm = self.root / 'hf_checkpoints'
        self.language = {'language_ru': 'удмуртский', 'opus_code': 'udm', 'iso639_3': 'udm'}
        self.packet = {'description': 'parallel ru udm text', 'files': [], 'folders': [], 'errors': []}
        self.decision = {'suitable': True, 'verdict': 'keep', 'reason': 'Есть пара.',
                         'evidence': [{'source': 'description', 'quote': 'parallel ru udm'}]}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {'OPENROUTER_API_KEY': 'test-key'}))
        self.stack.enter_context(patch.object(core.time, 'sleep'))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        core.CHECKPOINT_UPLOAD = None

    def prepare(self, ids=('example/a',), refresh=False):
        with patch.object(core, 'hf_pair_groups', return_value=([], [{'id': x} for x in ids], [])), \
             patch.object(core, 'query_opus_for_language', return_value={'opus_checked': True, 'opus_ru_parallel_pairs': 4}):
            return core.prepare_pair_inventory([self.language], self.pairs, self.udm, refresh=refresh)

    def response(self, content=None, finish='stop', status=200):
        body = {'model': 'test-model', 'choices': [{'finish_reason': finish,
                 'message': {'content': json.dumps(self.decision) if content is None else content}}]}
        response = Mock(status_code=status, text=json.dumps(body), headers={})
        response.json.return_value = body
        if status >= 400:
            response.raise_for_status.side_effect = core.requests.HTTPError('quota', response=response)
        return response

    def test_completed_skipped_after_catalog_refresh_and_relocation(self):
        self.prepare()
        with patch.object(core, 'hf_collect_evidence', new=lambda dataset_id: self.packet), \
             patch.object(core.requests, 'post', return_value=self.response()) as post:
            core.run_pair_inventory(self.pairs, max_requests=1)
            self.assertEqual(post.call_count, 1)
            self.prepare(('example/a', 'example/b'), refresh=True)
            summary = core.summarize_pair_inventory(self.pairs)['summary']
            self.assertEqual(summary['pending'], 1)
            core.run_pair_inventory(self.pairs, max_requests=1)
            self.assertEqual(post.call_count, 2)
            core.run_pair_inventory(self.pairs, max_requests=20)
            self.assertEqual(post.call_count, 2)

    def test_429_saved_pending_and_resumed(self):
        self.prepare()
        with patch.object(core, 'hf_collect_evidence', new=lambda dataset_id: self.packet), \
             patch.object(core.requests, 'post', return_value=self.response(status=429)) as post:
            summary = core.run_pair_inventory(self.pairs, max_requests=20)['summary']
            self.assertEqual(post.call_count, 1)
            self.assertEqual(summary['pending'], 1)
        with patch.object(core, 'hf_collect_evidence', new=lambda dataset_id: self.packet), \
             patch.object(core.requests, 'post', return_value=self.response()):
            self.assertEqual(core.run_pair_inventory(self.pairs, max_requests=1)['summary']['pending'], 0)

    def test_actual_http_retries_count_toward_budget(self):
        trace = {}
        budget = {'remaining': 2}
        with patch.object(core.requests, 'post', return_value=self.response(status=503)) as post:
            with self.assertRaisesRegex(RuntimeError, 'Лимит запросов'):
                core.hf_llm_decision(self.packet, 'test-key', trace=trace, request_budget=budget)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(budget['remaining'], 0)

    def test_truncated_response_saved_and_retryable(self):
        self.prepare()
        with patch.object(core, 'hf_collect_evidence', new=lambda dataset_id: self.packet), \
             patch.object(core.requests, 'post', return_value=self.response('{"reason": "unfinished', 'length')):
            result = core.run_pair_inventory(self.pairs, max_requests=1)
        self.assertEqual(result['summary']['errors'], 1)
        run_id = json.loads((self.udm / 'latest_run.json').read_text())['run_id']
        record = json.loads((self.udm / run_id / (hashlib.sha256(b'example/a').hexdigest() + '.json')).read_text())
        self.assertEqual(record['result']['error_stage'], 'response_length')
        self.assertIn('unfinished', record['trace']['response_body'])

    def test_storage_failure_stops_before_model(self):
        self.prepare()
        with patch.object(core, 'CHECKPOINT_UPLOAD', side_effect=OSError('Drive unavailable')), \
             patch.object(core.requests, 'post') as post:
            with self.assertRaises(OSError):
                core.run_pair_inventory(self.pairs, max_requests=20)
            post.assert_not_called()

    def test_failed_opus_refresh_preserves_counts(self):
        self.prepare()
        with patch.object(core, 'hf_pair_groups', return_value=([], [], [])), \
             patch.object(core, 'query_opus_for_language', return_value={'opus_checked': False, 'opus_error': 'timeout'}):
            core.prepare_pair_inventory([self.language], self.pairs, self.udm, refresh=True)
        opus = json.loads((self.pairs / 'ru-udm/opus.json').read_text())
        self.assertEqual(opus['opus_ru_parallel_pairs'], 4)
        self.assertEqual(opus['refresh_error'], 'timeout')

    def test_zero_budget_does_not_collect_or_request(self):
        self.prepare()
        with patch.object(core, 'hf_collect_evidence') as collect, patch.object(core.requests, 'post') as post:
            core.run_pair_inventory(self.pairs, max_requests=0)
            collect.assert_not_called()
            post.assert_not_called()


class DriveStoreTests(unittest.TestCase):
    def setUp(self):
        from inventory.drive_store import DriveStore
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = DriveStore.__new__(DriveStore)
        self.store.root = Path(self.temp.name).resolve()
        self.relative = Path('hf_checkpoints/run/record.json')
        self.path = self.store.root / self.relative
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"result": "new"}')
        self.store.ids = {Path('.'): 'root', Path('hf_checkpoints'): 'hf',
                          Path('hf_checkpoints/run'): 'run', self.relative: 'record'}
        self.store.versions = {self.relative: 'old-time'}
        self.store.digests = {}
        self.store.service = Mock()

    def test_remote_conflict_never_overwritten(self):
        self.store.service.files().get().execute.return_value = {'modifiedTime': 'changed-by-colab'}
        with self.assertRaisesRegex(RuntimeError, 'Конфликт'):
            self.store.upload(self.path)
        self.store.service.files().update.assert_not_called()

    def test_unchanged_file_not_reuploaded(self):
        self.store.service.files().get().execute.return_value = {'modifiedTime': 'old-time'}
        self.store.service.files().update().execute.return_value = {'id': 'record', 'modifiedTime': 'new-time'}
        self.store.upload(self.path)
        self.store.service.reset_mock()
        self.store.upload(self.path)
        self.store.service.files.assert_not_called()

    def test_state_cannot_escape_storage_root(self):
        with self.assertRaises(ValueError):
            self.store.upload(self.store.root.parent / 'other.json')


if __name__ == '__main__':
    unittest.main()
