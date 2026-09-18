# Автопрогон инвентаризации в GitHub Actions

Пайплайн работает на обычном CPU: скачивает метаданные и HTML, извлекает описание
и пути файлов, обращается к модели через OpenRouter. Файлы корпусов не скачивает.
Colab остаётся учебной тетрадкой; Actions выполняет только рабочую часть разделов 6–7.

## Что уже добавлено

- `.github/workflows/dataset-inventory.yml`: ежедневный `review` в 09:17 МСК,
  еженедельный `discover` по понедельникам в 07:17 МСК и кнопка ручного запуска.
- `scripts/run_dataset_inventory.py`: команды `discover`, `review`, `summary`.
- `scripts/inventory/core.py`: логика из актуальной тетрадки, включая короткий ответ
  судьи, проверку цитат, полные ответы API, ошибки и возобновление.
- `scripts/inventory/drive_store.py`: чтение существующих чекпоинтов и запись на Drive.
- `Inventory tests`: проверки на искусственных ответах, без запросов OpenRouter.

Расписание задано в UTC; запуск GitHub может задержаться. В публичном репозитории
расписание отключается после 60 дней отсутствия активности. Стандартный Linux runner
в публичном репозитории бесплатный; бесплатная квота OpenRouter от этого не увеличивается.
[Расписание GitHub](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule),
[стоимость Actions](https://docs.github.com/en/billing/concepts/product-billing/github-actions).

Автопрогон начнёт работать после настройки ниже. До этого он пишет в Summary,
что выключен, и не обращается к источникам или модели.

## 1. Один раз разрешить доступ к Google Диску

Нужен OAuth от владельца личного Google Диска. Подключение Drive в Colab или Codex
не предоставляет GitHub доступ автоматически.

1. В [Google Cloud Console](https://console.cloud.google.com/) создайте проект для курса
   и включите **Google Drive API** через APIs & Services → Library.
2. Настройте Google Auth Platform / OAuth consent screen для личного использования.
   Если приложение в режиме Testing, добавьте свой Google-аккаунт как test user.
3. Создайте OAuth client типа **Desktop app**. Скачайте JSON в
   `.inventory-secrets/google-client.json` внутри локального репозитория.
4. На своём компьютере, из корня репозитория выполните:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-inventory.txt
   python scripts/authorize_inventory_drive.py --client .inventory-secrets/google-client.json
   ```

5. В открывшемся браузере выберите аккаунт с папкой `lowres_lab` и разрешите доступ.
   Получится `.inventory-secrets/drive-token.json`. Это файл для GitHub Secret,
   не для коммита. Каталог уже исключён через `.gitignore`.

Запрашивается scope `drive`, поскольку нужно читать и обновлять существующие файлы,
созданные Colab; scope `drive.file` сам по себе не открывает их этому OAuth-клиенту.
Сам скрипт обходит только `hf_checkpoints` и `ru_language_pairs` внутри указанной папки.

Для External-приложения в статусе Testing refresh token для такого доступа истекает
через 7 дней. Для длительного расписания нужно перевести приложение в Production
с учётом требований Google к вашему приложению и заново авторизоваться.
[Документация Google OAuth](https://developers.google.com/identity/protocols/oauth2#expiration).

## 2. Настроить Secrets и Variables в репозитории

Откройте [Settings → Secrets and variables → Actions](https://github.com/AnnaLebedeva/lowres-course/settings/secrets/actions).

Во вкладке **Secrets** создайте:

| Имя | Значение |
| --- | --- |
| `OPENROUTER_API_KEY` | Ключ вашего аккаунта OpenRouter |
| `GOOGLE_DRIVE_TOKEN_JSON` | Всё содержимое `.inventory-secrets/drive-token.json` |

Во вкладке **Variables** создайте:

| Имя | Значение |
| --- | --- |
| `GOOGLE_DRIVE_FOLDER_ID` | ID существующей папки **lowres_lab**, не ID тетрадки |
| `INVENTORY_ENABLED` | `true` для включения; `false` для паузы |

ID папки — часть URL Google Drive после `/folders/`. Имя папки проверяется скриптом.
Данные и полные ответы модели остаются на вашем Google Диске; в публичный репозиторий
и Actions artifacts они не загружаются.

## 3. Первый ручной запуск

Откройте **Actions → Dataset inventory → Run workflow** на ветке `main`.

1. Выберите `summary`: если раздел 6 тетрадки уже подготовил `pairs.json`,
   появится сводка существующих проверок без расхода OpenRouter.
   Если каталога нет, выполните `discover`.
2. Выполните `discover`, чтобы обновить списки кандидатов всех языковых пар.
   Этот режим не вызывает модель. Старые результаты сохраняются.
3. Выполните `review` с `max_requests=1`. Проверьте новый JSON на Drive и Summary запуска.
4. Следующий `review` продолжит очередь; завершённую запись заново не проверяет.
   Расписание использует максимум 20 HTTP-запросов на запуск, включая повторы 502/503.

Бюджет 20 — ограничение одного запуска, а не учёт всей суточной квоты аккаунта.
Ручные запуски и Colab расходуют ту же квоту OpenRouter. При 429/402/401/403
скрипт сохраняет pending и прекращает все проверки до следующего запуска.
Один проход обрабатывает не более одного датасета каждой пары, начиная с сохранённого
указателя, поэтому может использовать меньше 20 запросов.

Не запускайте записывающие ячейки Colab одновременно с Actions. Два Actions-запуска
сериализованы через concurrency; конфликт с изменённым файлом Drive останавливает
запись. Проверка времени изменения не является распределённой блокировкой.

## Как сохраняется прогресс

Используются прежние папки тетрадки:

```text
lowres_lab/
  hf_checkpoints/                    # существующие проверки ru–udm
  ru_language_pairs/
    pairs.json
    cursor.json
    all_pairs_summary.csv
    all_pair_decisions.csv
    manual_review.csv
    summary.json
    ru-.../
      catalog.json
      opus.json
      reviews/
```

Каждый JSON отправляется на Drive сразу после записи. При недоступности хранилища
процесс останавливается, чтобы не продолжать расходовать запросы без сохранения.
Если процесс прервали между ответом модели и сохранением, последний запрос может
повториться. CSV — производные таблицы: их можно восстановить командой `summary`.

`keep`, `reject` и `insufficient_evidence` повторно не отправляются модели;
`insufficient_evidence` остаётся для ручного разбора. `pending` и `error` продолжаются.
Проверка относится к конкретной паре и цели: результат ru–udm нельзя автоматически
перенести на ru–tt. Изменение кода диагностики не аннулирует готовые результаты.
Изменение цели/модели/лимитов материалов требует явного решения о новой проверке.

Обновление каталогов объединяет старых и новых кандидатов по полному HF ID.
Старые записи не удаляются при временном исчезновении из выдачи. OPUS обновляет
метаданные; при ошибке сохраняются последние успешные показатели с `refresh_error`.
Автоматического объединения копий OPUS и HF по названию нет: такая связь требует
подтверждения. Числа OPUS — суммы выравниваний, не число уникальных предложений.

## Где смотреть ошибки

Actions → Dataset inventory → нужный запуск: **Summary** и журнал шага выполнения.
На Drive в JSON датасета: `result`, `trace.response_body`, `trace.raw_answer`,
`trace.finish_reason`, `materials.errors` и `history`. `response_length` означает,
что модель не завершила ответ в токенный лимит. Это техническая ошибка, не отказ датасету.

Старый `scripts/build_language_dataset_inventory.py` оставлен как прежний пример;
workflow его не запускает. Учебная тетрадка в Google Drive не меняется этим скриптом.

## Локальная проверка

```bash
python -m unittest discover -s tests -v
python scripts/run_dataset_inventory.py summary --drive
```

Для локальной копии папки `lowres_lab` можно убрать `--drive` и указать
`--state-dir /путь/к/lowres_lab`. Тогда изменения останутся только локально.
