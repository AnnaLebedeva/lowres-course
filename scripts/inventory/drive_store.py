"""Синхронизация двух папок чекпоинтов с личным Google Диском через OAuth."""
import errno
import hashlib
import http.client
import io
import json
import os
import socket
import ssl
import time
from datetime import datetime
from pathlib import Path

import httplib2
from google.oauth2.credentials import Credentials
from google_auth_httplib2 import AuthorizedHttp
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


def transient_error(exc):
    """Повторяем только временные ошибки, но не отозванный доступ или конфликты."""
    if isinstance(exc, HttpError):
        if exc.resp.status in (408, 429) or 500 <= exc.resp.status < 600:
            return True
        if exc.resp.status == 403:
            try:
                errors = json.loads(exc.content)['error'].get('errors', [])
                return any(e.get('reason') in {'rateLimitExceeded', 'userRateLimitExceeded'} for e in errors)
            except (ValueError, KeyError, TypeError):
                return False
        return False
    if isinstance(exc, (TimeoutError, socket.timeout, socket.gaierror, ConnectionError,
                        ssl.SSLEOFError, ssl.SSLZeroReturnError,
                        httplib2.ServerNotFoundError, http.client.IncompleteRead,
                        http.client.BadStatusLine)):
        return True
    return isinstance(exc, OSError) and exc.errno in {
        errno.ETIMEDOUT, errno.ECONNRESET, errno.ECONNABORTED,
        errno.EPIPE, errno.ENETUNREACH, errno.EHOSTUNREACH,
    }


class DriveStore:
    """Читает существующие чекпоинты и загружает изменения после каждого результата.

    Не удаляет файлы. При изменении файла другим процессом останавливается,
    чтобы не затереть результаты Colab. Одновременный запуск Colab и Actions запрещён.
    """

    def __init__(self, root_id, local_root, credentials_json):
        """Подключает OAuth пользователя; root_id должен указывать на lowres_lab."""
        info = json.loads(credentials_json)
        if not all(info.get(k) for k in ('client_id', 'client_secret', 'refresh_token')):
            raise ValueError('GOOGLE_DRIVE_TOKEN_JSON не содержит OAuth refresh token')
        credentials = Credentials.from_authorized_user_info(info)
        self.http = httplib2.Http(timeout=60)
        self.service = build('drive', 'v3', http=AuthorizedHttp(credentials, http=self.http),
                             cache_discovery=False)
        self.root = Path(local_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ids = {Path('.'): root_id}
        self.versions = {}
        self.digests = {}
        self.pending_creates = {}
        self.pending_updates = {}
        root = self._retry(lambda: self.service.files().get(
            fileId=root_id, fields='id,name,mimeType').execute())
        if root['mimeType'] != 'application/vnd.google-apps.folder' or root['name'] != 'lowres_lab':
            raise ValueError('GOOGLE_DRIVE_FOLDER_ID должен указывать на папку lowres_lab')

    def _retry(self, operation):
        """Одна попытка и до трёх повторов; не повторяет запросы к модели."""
        for attempt in range(4):
            try:
                return operation()
            except Exception as exc:
                if not transient_error(exc) or attempt == 3:
                    raise
                # После SSL EOF нельзя продолжать использовать повреждённый сокет.
                if not isinstance(exc, HttpError):
                    self.http.close()
                delay = 2 ** (attempt + 1)
                print(f'Drive: временный сбой {type(exc).__name__}; повтор {attempt + 1}/3 через {delay} с', flush=True)
                time.sleep(delay)

    def _create(self, relative, parent_id, mime_type, content=None):
        """Повтор использует тот же ID, даже если успешный ответ Google потерялся."""
        if relative not in self.pending_creates:
            generated = self._retry(lambda: self.service.files().generateIds(
                count=1, space='drive').execute())
            self.pending_creates[relative] = generated['ids'][0]
        file_id = self.pending_creates[relative]
        fields = 'id,name,mimeType,parents,trashed,modifiedTime,md5Checksum'

        def existing():
            try:
                item = self.service.files().get(fileId=file_id, fields=fields).execute()
            except HttpError as exc:
                if exc.resp.status == 404:
                    return None
                raise
            if (item.get('trashed') or item.get('name') != relative.name or
                    item.get('mimeType') != mime_type or parent_id not in item.get('parents', []) or
                    (content is not None and item.get('md5Checksum') != hashlib.md5(content).hexdigest())):
                raise RuntimeError(f'Конфликт после создания {relative}; перечитайте состояние')
            return item

        def create():
            item = existing()
            if item is not None:
                return item
            if relative.name in self.children(parent_id, retry=False):
                raise RuntimeError(f'Другой запуск уже создал {relative}; перечитайте состояние')
            kwargs = {'body': {'id': file_id, 'name': relative.name, 'mimeType': mime_type,
                               'parents': [parent_id]}, 'fields': fields}
            if content is not None:
                kwargs['media_body'] = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)
            try:
                return self.service.files().create(**kwargs).execute()
            except HttpError as exc:
                if exc.resp.status == 409:
                    item = existing()
                    if item is not None:
                        return item
                raise

        saved = self._retry(create)
        del self.pending_creates[relative]
        return saved

    def children(self, folder_id, *, retry=True):
        """Читает все страницы списка прямых потомков, обнаруживает дубли имён."""
        token, found = None, {}
        while True:
            def fetch_page():
                return self.service.files().list(
                    q=f"'{folder_id}' in parents and trashed = false", pageSize=1000,
                    pageToken=token, fields='nextPageToken,files(id,name,mimeType,modifiedTime)',
                ).execute()
            # Создание уже обёрнуто в _retry: не умножаем вложенные повторы.
            data = self._retry(fetch_page) if retry else fetch_page()
            for item in data.get('files', []):
                name = item['name']
                if name in found:
                    raise RuntimeError(f'В папке Drive несколько файлов с именем {name}; устраните дубль')
                if name in {'.', '..'} or '/' in name or '\\' in name:
                    raise ValueError('Недопустимое имя в папке чекпоинтов')
                found[name] = item
            token = data.get('nextPageToken')
            if not token:
                return found

    def pull(self):
        """Скачивает только JSON/CSV состояния, не корпуса и не другие папки Drive."""
        root_items = self.children(self.ids[Path('.')])
        queue = []
        for name in ('hf_checkpoints', 'ru_language_pairs'):
            item = root_items.get(name)
            if item:
                if item['mimeType'] != 'application/vnd.google-apps.folder':
                    raise ValueError(f'{name} должен быть папкой')
                queue.append((Path(name), item['id']))
        while queue:
            relative, folder_id = queue.pop(0)
            self.ids[relative] = folder_id
            (self.root / relative).mkdir(parents=True, exist_ok=True)
            for name, item in self.children(folder_id).items():
                path = relative / name
                if item['mimeType'] == 'application/vnd.google-apps.folder':
                    queue.append((path, item['id']))
                elif path.suffix in {'.json', '.csv'}:
                    buffer = io.BytesIO()
                    download = MediaIoBaseDownload(buffer, self.service.files().get_media(fileId=item['id']))
                    done = False
                    while not done:
                        _, done = self._retry(lambda: download.next_chunk(num_retries=0))
                    content = buffer.getvalue()
                    target = self.root / path
                    target.write_bytes(content)
                    stamp = datetime.fromisoformat(item['modifiedTime'].replace('Z', '+00:00')).timestamp()
                    os.utime(target, (stamp, stamp))
                    self.ids[path] = item['id']
                    self.versions[path] = item['modifiedTime']
                    self.digests[path] = hashlib.sha256(content).hexdigest()

    def upload(self, path):
        """Сохраняет изменённый файл; проверяет конфликты и создаёт нужные подпапки."""
        path = Path(path).resolve()
        relative = path.relative_to(self.root)
        if relative.parts[0] not in {'hf_checkpoints', 'ru_language_pairs'} or path.suffix not in {'.json', '.csv'}:
            raise ValueError('Запись разрешена только для JSON/CSV чекпоинтов')
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if self.digests.get(relative) == digest:
            return
        parent = Path('.')
        for part in relative.parent.parts:
            child = parent / part
            if child not in self.ids:
                existing = self.children(self.ids[parent]).get(part)
                if existing:
                    if existing['mimeType'] != 'application/vnd.google-apps.folder':
                        raise RuntimeError(f'Ожидалась папка {child}')
                    self.ids[child] = existing['id']
                else:
                    created = self._create(child, self.ids[parent], 'application/vnd.google-apps.folder')
                    self.ids[child] = created['id']
            parent = child
        mime_type = 'application/json' if path.suffix == '.json' else 'text/csv'
        if relative in self.versions:
            def update():
                current = self.service.files().get(fileId=self.ids[relative],
                                                   fields='id,modifiedTime,md5Checksum').execute()
                # Признаём потерянный успешный ответ только для наших отправленных байтов.
                if (self.pending_updates.get(relative) == digest and
                        current.get('md5Checksum') == hashlib.md5(content).hexdigest()):
                    return current
                if current['modifiedTime'] != self.versions[relative]:
                    raise RuntimeError(f'Конфликт с другим запуском: {relative}. Автопрогон остановлен.')
                self.pending_updates[relative] = digest
                media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)
                return self.service.files().update(fileId=self.ids[relative], media_body=media,
                                                  fields='id,modifiedTime').execute()
            saved = self._retry(update)
        else:
            saved = self._create(relative, self.ids[parent], mime_type, content)
        self.ids[relative] = saved['id']
        self.versions[relative] = saved['modifiedTime']
        self.digests[relative] = digest
        self.pending_updates.pop(relative, None)

    def flush(self):
        """Досохраняет производные CSV и файлы, которые ещё не отправлены на Drive."""
        for name in ('hf_checkpoints', 'ru_language_pairs'):
            for path in sorted((self.root / name).rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.csv'}:
                    self.upload(path)
