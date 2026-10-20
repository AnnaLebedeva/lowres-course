"""Проверки возобновления и ошибок без сетевых запросов и расхода OpenRouter."""
import hashlib
import io
import json
import os
import ssl
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, ExitStack
from unittest.mock import Mock, patch

import httplib2
from google.auth.exceptions import RefreshError
from googleapiclient.errors import HttpError

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

    def test_reasoning_fallback_and_content_priority(self):
        answer = json.dumps(self.decision, ensure_ascii=False)
        cases = [
            (None, answer, 'reasoning'),
            ('  ', answer, 'reasoning'),
            (None, json.dumps(answer, ensure_ascii=False)[1:-1], 'reasoning'),
            (answer, 'not a decision', 'content'),
        ]
        for content, reasoning, source in cases:
            with self.subTest(source=source, content=content):
                response = self.response()
                response.json.return_value['choices'][0]['message'] = {
                    'content': content, 'reasoning': reasoning}
                trace = {}
                with patch.object(core.requests, 'post', return_value=response) as post:
                    result = core.hf_llm_decision(self.packet, 'test-key', trace=trace)
                self.assertEqual(result['verdict'], 'keep')
                self.assertEqual(trace['answer_source'], source)
                self.assertEqual(post.call_count, 1)

    def test_reasoning_must_be_complete_valid_decision(self):
        invalid = ['thinking ' + json.dumps(self.decision), '{}',
                   json.dumps({**self.decision, 'suitable': False}),
                   json.dumps({**self.decision, 'evidence': []})]
        for reasoning in invalid:
            with self.subTest(reasoning=reasoning):
                response = self.response()
                response.json.return_value['choices'][0]['message'] = {
                    'content': None, 'reasoning': reasoning}
                with patch.object(core.requests, 'post', return_value=response):
                    with self.assertRaises(ValueError):
                        core.hf_llm_decision(self.packet, 'test-key')

    def test_length_is_not_rescued_by_reasoning(self):
        response = self.response(finish='length')
        response.json.return_value['choices'][0]['message'] = {
            'content': None, 'reasoning': json.dumps(self.decision)}
        with patch.object(core.requests, 'post', return_value=response):
            with self.assertRaisesRegex(ValueError, 'обрезан'):
                core.hf_llm_decision(self.packet, 'test-key')

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
        self.store.pending_creates = {}
        self.store.pending_updates = {}
        self.store.http = Mock()
        self.store.service = Mock()
        sleep_patch = patch('inventory.drive_store.time.sleep')
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self.store.service.files().list().execute.return_value = {'files': []}

    @staticmethod
    def http_error(status, reason='backendError'):
        return HttpError(httplib2.Response({'status': str(status)}),
                         json.dumps({'error': {'errors': [{'reason': reason}]}}).encode())

    def created_metadata(self, file_id='new-record'):
        return {'id': file_id, 'name': 'record.json', 'mimeType': 'application/json',
                'parents': ['run'], 'trashed': False, 'modifiedTime': 'new-time',
                'md5Checksum': hashlib.md5(self.path.read_bytes()).hexdigest()}

    def prepare_create(self):
        self.store.versions.clear()
        self.store.ids.pop(self.relative)
        self.store.service.files().generateIds().execute.return_value = {'ids': ['new-record']}

    def test_transient_errors_retry_with_bounded_backoff(self):
        for error in (self.http_error(500), self.http_error(503), self.http_error(429),
                      self.http_error(403, 'rateLimitExceeded'), ssl.SSLEOFError('EOF'),
                      TimeoutError('timeout'), ConnectionResetError('reset')):
            with self.subTest(error=type(error).__name__):
                operation = Mock(side_effect=[error, 'ok'])
                self.sleep.reset_mock()
                self.assertEqual(self.store._retry(operation), 'ok')
                self.assertEqual(operation.call_count, 2)
                self.sleep.assert_called_once_with(2)
        operation = Mock(side_effect=self.http_error(500))
        self.sleep.reset_mock()
        with self.assertRaises(HttpError):
            self.store._retry(operation)
        self.assertEqual(operation.call_count, 4)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [2, 4, 8])

    def test_auth_permission_and_conflict_errors_are_not_retried(self):
        for error in (self.http_error(400), self.http_error(401),
                      self.http_error(403, 'insufficientPermissions'), self.http_error(404),
                      self.http_error(409), RefreshError('invalid_grant'),
                      ssl.SSLCertVerificationError('certificate'), RuntimeError('conflict')):
            with self.subTest(error=type(error).__name__):
                operation = Mock(side_effect=error)
                self.sleep.reset_mock()
                with self.assertRaises(type(error)):
                    self.store._retry(operation)
                operation.assert_called_once()
                self.sleep.assert_not_called()

    def test_list_recovers_from_ssl_eof_and_discards_connection(self):
        self.store.service.files().list().execute.side_effect = [ssl.SSLEOFError('EOF'), {'files': []}]
        self.assertEqual(self.store.children('root'), {})
        self.store.http.close.assert_called_once()

    def test_update_retries_500_with_fresh_media(self):
        files = self.store.service.files()
        files.get().execute.return_value = {'id': 'record', 'modifiedTime': 'old-time'}
        files.update().execute.side_effect = [self.http_error(500), {'id': 'record', 'modifiedTime': 'new-time'}]
        files.update.reset_mock()
        self.store.upload(self.path)
        self.assertEqual(files.update.call_count, 2)
        for call in files.update.call_args_list:
            media = call.kwargs['media_body']
            self.assertEqual(media.getbytes(0, media.size()), self.path.read_bytes())
        self.assertEqual(self.store.versions[self.relative], 'new-time')

    def test_lost_update_response_is_recognized_without_second_write(self):
        files = self.store.service.files()
        files.get().execute.side_effect = [
            {'id': 'record', 'modifiedTime': 'old-time'}, self.created_metadata('record')]
        files.update().execute.side_effect = ssl.SSLEOFError('response lost after commit')
        files.update.reset_mock()
        self.store.upload(self.path)
        files.update.assert_called_once()
        self.assertEqual(self.store.versions[self.relative], 'new-time')
        self.assertEqual(self.store.pending_updates, {})

    def test_concurrent_edit_between_update_attempts_is_not_overwritten(self):
        files = self.store.service.files()
        files.get().execute.side_effect = [
            {'modifiedTime': 'old-time'}, {'modifiedTime': 'colab-time', 'md5Checksum': 'other'}]
        files.update().execute.side_effect = ssl.SSLEOFError('EOF')
        files.update.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'Конфликт'):
            self.store.upload(self.path)
        files.update.assert_called_once()

    def test_lost_create_response_does_not_duplicate_file(self):
        self.prepare_create()
        files = self.store.service.files()
        files.get().execute.side_effect = [self.http_error(404), self.created_metadata()]
        files.create().execute.side_effect = ssl.SSLEOFError('response lost after commit')
        files.create.reset_mock()
        self.store.upload(self.path)
        files.create.assert_called_once()
        self.assertEqual(files.create.call_args.kwargs['body']['id'], 'new-record')
        self.assertEqual(self.store.ids[self.relative], 'new-record')

    def test_create_reuses_id_when_first_request_did_not_commit(self):
        self.prepare_create()
        files = self.store.service.files()
        files.get().execute.side_effect = self.http_error(404)
        files.create().execute.side_effect = [self.http_error(503), self.created_metadata()]
        files.create.reset_mock()
        self.store.upload(self.path)
        self.assertEqual([c.kwargs['body']['id'] for c in files.create.call_args_list],
                         ['new-record', 'new-record'])

    def test_create_409_reconciles_only_matching_content(self):
        for checksum in (hashlib.md5(self.path.read_bytes()).hexdigest(), 'someone-elses-content'):
            with self.subTest(checksum=checksum):
                self.store.versions.clear()
                self.store.digests.clear()
                self.store.service.files().generateIds().execute.return_value = {'ids': ['new-record']}
                files = self.store.service.files()
                files.get().execute.side_effect = [self.http_error(404),
                    {**self.created_metadata(), 'md5Checksum': checksum}]
                files.create().execute.side_effect = self.http_error(409)
                if checksum == 'someone-elses-content':
                    with self.assertRaisesRegex(RuntimeError, 'Конфликт'):
                        self.store.upload(self.path)
                else:
                    self.store.upload(self.path)

    def test_flush_preserves_create_id_after_exhausted_retries(self):
        self.prepare_create()
        files = self.store.service.files()
        files.get().execute.side_effect = self.http_error(404)
        files.create().execute.side_effect = ssl.SSLEOFError('EOF')
        files.create.reset_mock()
        with self.assertRaises(ssl.SSLEOFError):
            self.store.upload(self.path)
        self.assertEqual(files.create.call_count, 4)
        files.get().execute.side_effect = None
        files.get().execute.return_value = self.created_metadata()
        self.store.flush()
        self.assertEqual(files.create.call_count, 4)
        self.assertEqual(self.store.pending_creates, {})
        self.assertEqual(self.store.ids[self.relative], 'new-record')

    def test_folder_create_recovers_after_lost_response(self):
        files = self.store.service.files()
        files.generateIds().execute.return_value = {'ids': ['new-folder']}
        files.get().execute.side_effect = [self.http_error(404),
            {'id': 'new-folder', 'name': 'run', 'mimeType': 'application/vnd.google-apps.folder',
             'parents': ['hf'], 'modifiedTime': 'new-time'}]
        files.create().execute.side_effect = ssl.SSLEOFError('EOF')
        files.create.reset_mock()
        item = self.store._create(Path('hf_checkpoints/run'), 'hf', 'application/vnd.google-apps.folder')
        self.assertEqual(item['id'], 'new-folder')
        files.create.assert_called_once()

    def test_other_file_with_same_name_is_not_adopted(self):
        self.prepare_create()
        files = self.store.service.files()
        files.get().execute.side_effect = self.http_error(404)
        files.list().execute.return_value = {'files': [{'id': 'colab-record', 'name': 'record.json'}]}
        with self.assertRaisesRegex(RuntimeError, 'Другой запуск'):
            self.store.upload(self.path)
        files.create.assert_not_called()

    def test_create_does_not_multiply_list_retries(self):
        self.prepare_create()
        files = self.store.service.files()
        files.get().execute.side_effect = self.http_error(404)
        files.list().execute.side_effect = self.http_error(500)
        with self.assertRaises(HttpError):
            self.store.upload(self.path)
        self.assertEqual(files.list().execute.call_count, 4)
        self.assertEqual(self.sleep.call_count, 3)
        files.create.assert_not_called()

    def test_pull_retries_download_without_replaying_model(self):
        from inventory import drive_store
        self.store.children = Mock(side_effect=[
            {'hf_checkpoints': {'id': 'hf', 'mimeType': 'application/vnd.google-apps.folder'}},
            {'record.json': {'id': 'record', 'mimeType': 'application/json',
                             'modifiedTime': '2026-10-20T12:00:00Z'}}])
        def downloader(buffer, request):
            def download(**kwargs):
                if not getattr(download, 'retried', False):
                    download.retried = True
                    raise ssl.SSLEOFError('EOF')
                buffer.write(b'{}')
                return None, True
            return Mock(next_chunk=Mock(side_effect=download))
        with patch.object(drive_store, 'MediaIoBaseDownload', side_effect=downloader), \
             patch.object(core.requests, 'post') as model_request:
            self.store.pull()
            model_request.assert_not_called()
        self.assertEqual((self.store.root / 'hf_checkpoints/record.json').read_bytes(), b'{}')
        self.sleep.assert_called_once_with(2)

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
