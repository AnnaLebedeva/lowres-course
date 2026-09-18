"""Однократная локальная авторизация Drive; токен записывается в файл, не в вывод."""
import argparse
import os
from pathlib import Path
from google_auth_oauthlib.flow import InstalledAppFlow


def main():
    """Открывает согласие Google для Desktop OAuth client и сохраняет refresh token."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('.inventory-secrets/drive-token.json'))
    args = parser.parse_args()
    flow = InstalledAppFlow.from_client_secrets_file(str(args.client), scopes=['https://www.googleapis.com/auth/drive'])
    credentials = flow.run_local_server(port=0, access_type='offline', prompt='consent')
    if not credentials.refresh_token:
        raise RuntimeError('Google не выдал refresh token; повторите авторизацию')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(credentials.to_json())
    os.chmod(args.output, 0o600)
    print(f'Токен сохранён в {args.output}. Добавьте содержимое как GOOGLE_DRIVE_TOKEN_JSON в GitHub Secrets.')


if __name__ == '__main__':
    main()
