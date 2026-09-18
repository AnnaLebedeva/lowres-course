"""Командный запуск инвентаризации: discover, review или summary; без GPU и Colab."""
import argparse
import json
import os
from pathlib import Path

from inventory import core


def main():
    """Загружает состояние, выполняет выбранный этап и сохраняет сводку.

    --drive подключает Google Drive; без него работает с локальной копией состояния.
    review не обновляет каталог, discover не расходует OpenRouter.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['discover', 'review', 'summary'])
    parser.add_argument('--state-dir', type=Path, default=Path('.inventory-state'))
    parser.add_argument('--drive', action='store_true')
    parser.add_argument('--max-requests', type=int, default=20)
    args = parser.parse_args()
    if not 0 <= args.max_requests <= 20:
        parser.error('--max-requests должен быть от 0 до 20')
    store = None
    if args.drive:
        from inventory.drive_store import DriveStore
        if not os.environ.get('GOOGLE_DRIVE_FOLDER_ID') or not os.environ.get('GOOGLE_DRIVE_TOKEN_JSON'):
            parser.error('Задайте GOOGLE_DRIVE_FOLDER_ID и GOOGLE_DRIVE_TOKEN_JSON')
        store = DriveStore(os.environ['GOOGLE_DRIVE_FOLDER_ID'], args.state_dir,
                           os.environ['GOOGLE_DRIVE_TOKEN_JSON'])
        store.pull()
        core.CHECKPOINT_UPLOAD = store.upload
    root = args.state_dir / 'ru_language_pairs'
    try:
        if args.mode == 'discover':
            languages = json.loads(Path(core.__file__).with_name('languages.json').read_text())
            core.prepare_pair_inventory(languages, root, args.state_dir / 'hf_checkpoints', refresh=True)
            result = core.summarize_pair_inventory(root)
        else:
            if not (root / 'pairs.json').exists():
                parser.error('Каталог ещё не создан. Сначала запустите режим discover.')
            if args.mode == 'review':
                result = core.run_pair_inventory(root, max_requests=args.max_requests, retry_errors=True)
            else:
                result = core.summarize_pair_inventory(root)
        print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as report:
                report.write('## Инвентаризация датасетов\n\n')
                report.write('Результаты и полные ответы модели сохранены в Google Drive → lowres_lab.\n\n')
                for name, value in result['summary'].items():
                    report.write(f'- {name}: {value}\n')
    finally:
        # Основные JSON пишутся сразу. CSV досохраняем даже при ошибке следующего шага.
        if store:
            store.flush()
        core.CHECKPOINT_UPLOAD = None


if __name__ == '__main__':
    main()
