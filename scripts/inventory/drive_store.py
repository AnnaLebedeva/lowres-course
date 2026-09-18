"""Синхронизация двух папок чекпоинтов с личным Google Диском через OAuth."""
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload


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
        self.service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
        self.root = Path(local_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ids = {Path('.'): root_id}
        self.versions = {}
        self.digests = {}
        root = self.service.files().get(fileId=root_id, fields='id,name,mimeType').execute()
        if root['mimeType'] != 'application/vnd.google-apps.folder' or root['name'] != 'lowres_lab':
            raise ValueError('GOOGLE_DRIVE_FOLDER_ID должен указывать на папку lowres_lab')

    def children(self, folder_id):
        """Читает все страницы списка прямых потомков, обнаруживает дубли имён."""
        token, found = None, {}
        while True:
            data = self.service.files().list(
                q=f"'{folder_id}' in parents and trashed = false", pageSize=1000,
                pageToken=token, fields='nextPageToken,files(id,name,mimeType,modifiedTime)',
            ).execute()
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
                        _, done = download.next_chunk(num_retries=3)
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
                    created = self.service.files().create(body={
                        'name': part, 'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [self.ids[parent]],
                    }, fields='id').execute()
                    self.ids[child] = created['id']
            parent = child
        media = MediaIoBaseUpload(io.BytesIO(content), mimetype='application/json' if path.suffix == '.json' else 'text/csv')
        if relative in self.versions:
            current = self.service.files().get(fileId=self.ids[relative], fields='modifiedTime').execute()
            if current['modifiedTime'] != self.versions[relative]:
                raise RuntimeError(f'Конфликт с другим запуском: {relative}. Автопрогон остановлен.')
            saved = self.service.files().update(fileId=self.ids[relative], media_body=media,
                                                fields='id,modifiedTime').execute()
        else:
            if path.name in self.children(self.ids[parent]):
                raise RuntimeError(f'Другой запуск уже создал {relative}; перечитайте состояние')
            saved = self.service.files().create(body={'name': path.name, 'parents': [self.ids[parent]]},
                                                media_body=media, fields='id,modifiedTime').execute()
        self.ids[relative] = saved['id']
        self.versions[relative] = saved['modifiedTime']
        self.digests[relative] = digest

    def flush(self):
        """Досохраняет производные CSV и файлы, которые ещё не отправлены на Drive."""
        for name in ('hf_checkpoints', 'ru_language_pairs'):
            for path in sorted((self.root / name).rglob('*')):
                if path.is_file() and path.suffix in {'.json', '.csv'}:
                    self.upload(path)
