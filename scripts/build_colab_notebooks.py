import json
from pathlib import Path


ROOT = Path("/Users/ann.lebedeva/Documents/lowres")
OUT = ROOT / "colab_notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
import os, re, json, textwrap, math, statistics, random, io, time
from pathlib import Path
import pandas as pd
import numpy as np
import requests

DATA_DIR = Path('/content/lowres_lab')
DATA_DIR.mkdir(exist_ok=True)

def show_df(df, n=10):
    display(df.head(n))

def save_artifact(name, obj):
    path = DATA_DIR / name
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(path, index=False)
    else:
        path.write_text(str(obj), encoding='utf-8')
    print('saved:', path)
"""


def lesson01_real_agent_cells():
    return [
        md("""
# Занятие 1. Реальный мини-агент языковой лаборатории

**Цель практики:** собрать не теоретический пример, а маленького агента, который делает полезную работу для проекта сохранения малоресурсного языка.

Сценарий: у нас есть задача первичной разведки по удмуртскому языку. Агент должен:

1. найти реальные открытые материалы в Wikimedia Commons;
2. выбрать изображение, которое можно скачать;
3. прогнать OCR baseline в бесплатном Colab;
4. взять небольшой контрольный корпус из Удмуртской Википедии;
5. собрать отчет: что найдено, насколько читаемый OCR, что нужно проверить человеку.

Сначала соберем агента без фреймворка: обычные функции, словарь `state`, явные вызовы инструментов и решения. Потом завернем те же шаги в `LangGraph`, чтобы увидеть, зачем вообще нужен фреймворк для агентских workflow.

LLM/API-ключи не нужны: на первом занятии важнее понять архитектуру агента, чем подключать внешнюю модель.
"""),
        code("""
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr tesseract-ocr-rus
!pip -q install langgraph pytesseract pillow pandas matplotlib requests
"""),
        code(COMMON_SETUP + """
from typing import Any, Dict, List, TypedDict
from urllib.parse import quote
from PIL import Image, UnidentifiedImageError
from IPython.display import display
import matplotlib.pyplot as plt
import pytesseract
from langgraph.graph import StateGraph, END
"""),
        md("""
## 1. Настройка реальных источников

Берем два открытых источника:

- Wikimedia Commons: категория `Udmurt Dunne`, где лежат реальные файлы, связанные с удмуртской газетой.
- Удмуртская Википедия: небольшой корпус страниц для сравнения с OCR-выводом.

Это не учебная таблица: агент будет обращаться к API и сохранять полученные данные.

Если Wikimedia временно отвечает `429 Too Many Requests`, подождите минуту и перезапустите ячейку. В тетрадке запросы сгруппированы, но публичные API все равно иногда ограничивают Colab-адреса.
"""),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
UDM_WIKI_API = 'https://udm.wikipedia.org/w/api.php'

COMMONS_CATEGORY = 'Category:Udmurt Dunne'
LANGUAGE = 'удмуртский'
FALLBACK_FILE_TITLE = 'File:Удномер.jpg'
RASTER_MIME_TYPES = {'image/jpeg', 'image/png', 'image/tiff', 'image/webp'}

def api_get(url, params, timeout=30, attempts=3):
    last_error = None
    for attempt in range(attempts):
        r = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={'User-Agent': 'lowres-course-colab/1.0 (contact: teaching-notebook)'},
        )
        if r.status_code == 429 and attempt < attempts - 1:
            wait = int(r.headers.get('Retry-After', 2 + attempt * 2))
            print(f'Wikimedia API rate limit: ждем {wait} сек. и пробуем еще раз')
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 + attempt * 2)
    raise last_error

def commons_category_files(category, limit=20):
    data = api_get(COMMONS_API, {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmtype': 'file',
        'cmlimit': limit,
        'format': 'json',
    })
    return data.get('query', {}).get('categorymembers', [])

def commons_imageinfo(title):
    data = api_get(COMMONS_API, {
        'action': 'query',
        'titles': title,
        'prop': 'imageinfo',
        'iiprop': 'url|mime|size|extmetadata',
        'format': 'json',
    })
    pages = data.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    info = page.get('imageinfo', [{}])[0]
    meta = info.get('extmetadata', {})
    return {
        'title': title,
        'url': info.get('url'),
        'mime': info.get('mime'),
        'width': info.get('width'),
        'height': info.get('height'),
        'license': meta.get('LicenseShortName', {}).get('value'),
        'artist': meta.get('Artist', {}).get('value'),
        'description': meta.get('ImageDescription', {}).get('value'),
    }

def commons_imageinfo_batch(titles):
    if not titles:
        return []
    data = api_get(COMMONS_API, {
        'action': 'query',
        'titles': '|'.join(titles),
        'prop': 'imageinfo',
        'iiprop': 'url|mime|size|extmetadata',
        'format': 'json',
    })
    by_title = {}
    for page in data.get('query', {}).get('pages', {}).values():
        title = page.get('title')
        info = page.get('imageinfo', [{}])[0]
        meta = info.get('extmetadata', {})
        by_title[title] = {
            'title': title,
            'url': info.get('url'),
            'mime': info.get('mime'),
            'width': info.get('width'),
            'height': info.get('height'),
            'license': meta.get('LicenseShortName', {}).get('value'),
            'artist': meta.get('Artist', {}).get('value'),
            'description': meta.get('ImageDescription', {}).get('value'),
        }
    return [by_title[t] for t in titles if t in by_title]

def is_raster_image(info):
    return bool(info.get('url')) and info.get('mime') in RASTER_MIME_TYPES

def fetch_udm_wiki_pages(limit=5):
    random_pages = api_get(UDM_WIKI_API, {
        'action': 'query',
        'generator': 'random',
        'grnnamespace': 0,
        'grnlimit': limit,
        'prop': 'extracts',
        'explaintext': 1,
        'exintro': 1,
        'format': 'json',
    })
    pages = []
    for page in random_pages.get('query', {}).get('pages', {}).values():
        title = page.get('title', '')
        extract = page.get('extract', '') or ''
        pages.append({
            'title': title,
            'chars': len(extract),
            'tokens': len(re.findall(r'\\w+', extract.lower())),
            'cyrillic_share': cyrillic_share(extract),
            'url': 'https://udm.wikipedia.org/wiki/' + quote(title.replace(' ', '_')),
            'extract_preview': extract[:400],
        })
    return pages

def cyrillic_share(text):
    letters = re.findall(r'[A-Za-zА-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', text)
    if not letters:
        return 0.0
    cyr = [x for x in letters if re.match(r'[А-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', x)]
    return round(len(cyr) / len(letters), 3)

def text_diagnostics(text):
    tokens = re.findall(r'\\w+', text.lower())
    short_tokens = [t for t in tokens if len(t) <= 2]
    weird = re.findall(r'[^\\w\\s.,:;!?()\\-«»\"\\'/А-Яа-яЁёӜӝӞӟӤӥӦӧӴӵӸӹІіЇїЄєЎўҐґ]', text)
    return {
        'chars': len(text),
        'tokens': len(tokens),
        'cyrillic_share': cyrillic_share(text),
        'short_token_share': round(len(short_tokens) / max(1, len(tokens)), 3),
        'weird_char_count': len(weird),
        'sample': text[:700],
    }
"""),
        md("""
## 2. Агент без фреймворка: что под капотом

Минимальный агент состоит из четырех вещей:

- `state`: явная модель текущего контекста задачи;
- tools: функции, которые ходят во внешний мир или обрабатывают данные;
- observations: результаты вызова tools;
- policy: простые правила, которые решают, что делать дальше.

Ниже это обычный Python. Никакого LangGraph пока нет.
"""),
        code("""
def scout_sources(state):
    files = commons_category_files(state['commons_category'])
    if not files:
        files = [{'title': FALLBACK_FILE_TITLE, 'pageid': None}]
    state['commons_files'] = files
    save_artifact('lesson01_commons_candidates.csv', pd.DataFrame(files))
    return state

def choose_downloadable_image(state):
    titles = [item['title'] for item in state['commons_files']]
    inspected = commons_imageinfo_batch(titles)
    selected = None
    for info in inspected:
        if is_raster_image(info) and selected is None:
            selected = info
    if selected is None:
        fallback = commons_imageinfo(FALLBACK_FILE_TITLE)
        inspected.append(fallback)
        selected = fallback
    state['selected_file'] = selected
    state['inspected_files'] = inspected
    save_artifact('lesson01_commons_sources.csv', pd.DataFrame(inspected))
    return state

def download_openable_image(info, candidate_index=0):
    response = requests.get(
        info['url'],
        timeout=60,
        headers={'User-Agent': 'lowres-course-colab/1.0 (teaching notebook)'},
    )
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
    if content_type and content_type not in RASTER_MIME_TYPES and content_type != 'application/octet-stream':
        raise ValueError(f'expected raster image, got Content-Type={content_type}')

    raw = response.content
    try:
        img = Image.open(io.BytesIO(raw)).convert('RGB')
    except UnidentifiedImageError as exc:
        raise ValueError('downloaded bytes are not an openable raster image') from exc

    if max(img.size) > 2200:
        img.thumbnail((2200, 2200))

    image_path = DATA_DIR / f'lesson01_ocr_source_{candidate_index}.jpg'
    img.save(image_path, format='JPEG', quality=92)
    return image_path

def run_ocr_baseline(state):
    selected_title = state['selected_file'].get('title')
    candidates = [state['selected_file']]
    for candidate in state.get('inspected_files', []):
        if candidate.get('title') != selected_title:
            candidates.append(candidate)

    errors = []
    image_path = None
    selected = None
    for i, candidate in enumerate(candidates):
        if not is_raster_image(candidate):
            errors.append({'title': candidate.get('title'), 'error': f"unsupported mime: {candidate.get('mime')}"})
            continue
        try:
            image_path = download_openable_image(candidate, i)
            selected = candidate
            break
        except Exception as exc:
            errors.append({'title': candidate.get('title'), 'error': str(exc)})

    if selected is None or image_path is None:
        save_artifact('lesson01_download_errors.csv', pd.DataFrame(errors))
        raise RuntimeError('Не удалось скачать ни одного открываемого raster-изображения из Commons candidates.')

    if errors:
        save_artifact('lesson01_download_errors.csv', pd.DataFrame(errors))

    text = pytesseract.image_to_string(Image.open(image_path), lang='rus')
    save_artifact('lesson01_ocr_raw.txt', text)

    state['selected_file'] = selected
    state['image_path'] = str(image_path)
    state['ocr_text'] = text
    return state

def decide_human_tasks(state):
    diag = text_diagnostics(state.get('ocr_text', ''))
    tasks = []
    if diag['chars'] < 100:
        tasks.append('OCR почти ничего не извлек: выбрать другой скан или улучшить предобработку.')
    if diag['cyrillic_share'] < 0.7:
        tasks.append('В тексте много некириллического шума: проверить язык OCR и качество изображения.')
    if diag['short_token_share'] > 0.45:
        tasks.append('Много коротких фрагментов: нужна ручная проверка строк и сегментации.')
    if not tasks:
        tasks.append('Выбрать 20-30 строк и вручную оценить ошибки OCR.')
    state['ocr_diagnostics'] = diag
    state['human_tasks'] = tasks
    return state

def add_wiki_probe(state):
    pages = fetch_udm_wiki_pages(limit=5)
    save_artifact('lesson01_udm_wiki_probe.csv', pd.DataFrame(pages))
    all_text = ' '.join(p['extract_preview'] for p in pages)
    state['wiki_pages'] = pages
    state['wiki_stats'] = text_diagnostics(all_text)
    return state

def build_agent_report(state):
    report = {
        'language': state['language'],
        'agent_type': state.get('agent_type', 'plain Python tool-using agent'),
        'real_sources': {
            'commons_category': state['commons_category'],
            'selected_file_title': state['selected_file'].get('title'),
            'selected_file_url': state['selected_file'].get('url'),
            'license': state['selected_file'].get('license'),
            'wiki_pages': [p['url'] for p in state['wiki_pages']],
        },
        'ocr_diagnostics': state['ocr_diagnostics'],
        'wiki_probe_stats': state['wiki_stats'],
        'next_human_tasks': state['human_tasks'],
        'state_fields_used': sorted(state.keys()),
    }
    state['report'] = report
    save_artifact('lesson01_real_agent_report.json', json.dumps(report, ensure_ascii=False, indent=2))
    return state

def run_plain_agent(language, commons_category):
    state = {
        'language': language,
        'commons_category': commons_category,
        'goal': 'оценить, можно ли начать OCR-разведку по открытым удмуртским материалам',
        'plan': [
            'найти файлы',
            'выбрать изображение',
            'запустить OCR',
            'оценить текст',
            'сравнить с wiki-корпусом',
            'собрать отчет',
        ],
    }
    for step in [
        scout_sources,
        choose_downloadable_image,
        run_ocr_baseline,
        decide_human_tasks,
        add_wiki_probe,
        build_agent_report,
    ]:
        state = step(state)
        print('done:', step.__name__, '| state keys:', sorted(state.keys()))
    return state

plain_state = run_plain_agent(LANGUAGE, COMMONS_CATEGORY)
print(json.dumps(plain_state['report'], ensure_ascii=False, indent=2))
"""),
        md("""
## 3. Та же логика в LangGraph

Теперь берем те же функции и раскладываем их в граф. Смысл LangGraph не в том, что он “умнее”, а в том, что он делает архитектуру явной:

- у каждого узла есть входной и выходной `state`;
- переходы между шагами видны отдельно от кода инструментов;
- позже можно добавлять условия, повторы, LLM-узлы, human review и сохранение состояния между запусками.
"""),
        code("""
class LabAgentState(TypedDict, total=False):
    language: str
    commons_category: str
    commons_files: List[Dict[str, Any]]
    inspected_files: List[Dict[str, Any]]
    selected_file: Dict[str, Any]
    image_path: str
    ocr_text: str
    ocr_diagnostics: Dict[str, Any]
    wiki_pages: List[Dict[str, Any]]
    wiki_stats: Dict[str, Any]
    report: Dict[str, Any]
    human_tasks: List[str]
    agent_type: str

def source_scout_node(state: LabAgentState):
    next_state = scout_sources(dict(state))
    return {'commons_files': next_state['commons_files']}

def metadata_node(state: LabAgentState):
    next_state = choose_downloadable_image(dict(state))
    return {'selected_file': next_state['selected_file'], 'inspected_files': next_state['inspected_files']}

def ocr_node(state: LabAgentState):
    next_state = run_ocr_baseline(dict(state))
    return {'image_path': next_state['image_path'], 'ocr_text': next_state['ocr_text']}

def ocr_diagnostics_node(state: LabAgentState):
    next_state = decide_human_tasks(dict(state))
    return {'ocr_diagnostics': next_state['ocr_diagnostics'], 'human_tasks': next_state['human_tasks']}

def wiki_probe_node(state: LabAgentState):
    next_state = add_wiki_probe(dict(state))
    return {'wiki_pages': next_state['wiki_pages'], 'wiki_stats': next_state['wiki_stats']}

def report_node(state: LabAgentState):
    next_state = build_agent_report(dict(state))
    return {'report': next_state['report']}

workflow = StateGraph(LabAgentState)
workflow.add_node('source_scout', source_scout_node)
workflow.add_node('metadata', metadata_node)
workflow.add_node('ocr', ocr_node)
workflow.add_node('ocr_diagnostics', ocr_diagnostics_node)
workflow.add_node('wiki_probe', wiki_probe_node)
workflow.add_node('report', report_node)

workflow.set_entry_point('source_scout')
workflow.add_edge('source_scout', 'metadata')
workflow.add_edge('metadata', 'ocr')
workflow.add_edge('ocr', 'ocr_diagnostics')
workflow.add_edge('ocr_diagnostics', 'wiki_probe')
workflow.add_edge('wiki_probe', 'report')
workflow.add_edge('report', END)

agent = workflow.compile()
"""),
        md("## 4. Запускаем LangGraph-версию"),
        code("""
state = agent.invoke({
    'language': LANGUAGE,
    'commons_category': COMMONS_CATEGORY,
    'agent_type': 'LangGraph tool-using workflow agent',
})

print(json.dumps(state['report'], ensure_ascii=False, indent=2))
"""),
        md("## 5. Смотрим реальные артефакты"),
        code("""
print('Выбранный файл:', state['selected_file']['title'])
print('Лицензия:', state['selected_file'].get('license'))
print('URL:', state['selected_file'].get('url'))

img = Image.open(state['image_path'])
print('Размер изображения для OCR:', img.size)
display(img)
"""),
        code("""
print(state['ocr_text'][:2000])
"""),
        code("""
display(pd.DataFrame(state['wiki_pages'])[['title', 'chars', 'tokens', 'cyrillic_share', 'url']])
display(pd.DataFrame([state['ocr_diagnostics'], state['wiki_stats']], index=['ocr', 'wiki_probe']))
"""),
        md("""
## 6. Что здесь агентского

В этой вводной тетрадке важны не “виды агентов”, а архитектура на конкретной задаче:

- `state` хранит текущую картину задачи: цель, найденные источники, выбранный файл, OCR-текст, диагностику, контрольный корпус и задачи для человека.
- каждый узел делает один проверяемый шаг;
- следующий шаг выбирается на основании уже собранного состояния;
- человек остается в контуре там, где нельзя автоматически решать про права, качество и публикацию.

В следующих тетрадках мы уже не будем каждый раз подробно разбирать архитектуру. Будем использовать этот принцип как рабочий шаблон: источник, tool, state, диагностика, human boundary, отчет.
"""),
        md("""
## Вопросы для отчета

1. Какой реальный материал нашел агент и можно ли понять его лицензию?
2. Какие части plain Python-агента соответствуют `state`, tools, observations и policy?
3. Что стало понятнее или надежнее после переноса той же логики в LangGraph?
4. Получился ли OCR-текст вменяемым? Покажите 3-5 характерных ошибок.
5. Что в этой задаче нельзя отдавать агенту полностью автоматически?
"""),
    ]


def lesson01_dataset_scout_cells():
    languages_code = repr([
        {"language_ru": "татарский", "language_en": "Tatar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "tat", "opus_code": "tt"},
        {"language_ru": "башкирский", "language_en": "Bashkir", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "bak", "opus_code": "ba"},
        {"language_ru": "чувашский", "language_en": "Chuvash", "family": "Тюркская", "branch": "огурская", "iso639_3": "chv", "opus_code": "chv"},
        {"language_ru": "якутский / саха", "language_en": "Sakha / Yakut", "family": "Тюркская", "branch": "сибирская", "iso639_3": "sah", "opus_code": "sah"},
        {"language_ru": "тувинский", "language_en": "Tuvan", "family": "Тюркская", "branch": "сибирская", "iso639_3": "tyv", "opus_code": "tyv"},
        {"language_ru": "хакасский", "language_en": "Khakas", "family": "Тюркская", "branch": "сибирская", "iso639_3": "kjh", "opus_code": "kjh"},
        {"language_ru": "алтайский", "language_en": "Altai", "family": "Тюркская", "branch": "сибирская", "iso639_3": "alt", "opus_code": "alt"},
        {"language_ru": "кумыкский", "language_en": "Kumyk", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "kum", "opus_code": "kum"},
        {"language_ru": "карачаево-балкарский", "language_en": "Karachay-Balkar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "krc", "opus_code": "krc"},
        {"language_ru": "ногайский", "language_en": "Nogai", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "nog", "opus_code": "nog"},
        {"language_ru": "крымскотатарский", "language_en": "Crimean Tatar", "family": "Тюркская", "branch": "кыпчакско-огузская", "iso639_3": "crh", "opus_code": "crh"},
        {"language_ru": "удмуртский", "language_en": "Udmurt", "family": "Уральская", "branch": "пермская", "iso639_3": "udm", "opus_code": "udm"},
        {"language_ru": "коми-зырянский", "language_en": "Komi-Zyrian", "family": "Уральская", "branch": "пермская", "iso639_3": "kpv", "opus_code": "kpv"},
        {"language_ru": "коми-пермяцкий", "language_en": "Komi-Permyak", "family": "Уральская", "branch": "пермская", "iso639_3": "koi", "opus_code": "koi"},
        {"language_ru": "эрзянский", "language_en": "Erzya", "family": "Уральская", "branch": "мордовская", "iso639_3": "myv", "opus_code": "myv"},
        {"language_ru": "мокшанский", "language_en": "Moksha", "family": "Уральская", "branch": "мордовская", "iso639_3": "mdf", "opus_code": "mdf"},
        {"language_ru": "марийский луговой", "language_en": "Meadow Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mhr", "opus_code": "mhr"},
        {"language_ru": "марийский горный", "language_en": "Hill Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mrj", "opus_code": "mrj"},
        {"language_ru": "карельский", "language_en": "Karelian", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "krl", "opus_code": "krl"},
        {"language_ru": "вепсский", "language_en": "Veps", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "vep", "opus_code": "vep"},
        {"language_ru": "хантыйский", "language_en": "Khanty", "family": "Уральская", "branch": "угорская", "iso639_3": "kca", "opus_code": None},
        {"language_ru": "мансийский", "language_en": "Mansi", "family": "Уральская", "branch": "угорская", "iso639_3": "mns", "opus_code": "mns"},
        {"language_ru": "ненецкий", "language_en": "Nenets", "family": "Уральская", "branch": "самодийская", "iso639_3": "yrk", "opus_code": "yrk"},
        {"language_ru": "чеченский", "language_en": "Chechen", "family": "Северокавказская", "branch": "нахская", "iso639_3": "che", "opus_code": "ce"},
        {"language_ru": "ингушский", "language_en": "Ingush", "family": "Северокавказская", "branch": "нахская", "iso639_3": "inh", "opus_code": "inh"},
        {"language_ru": "аварский", "language_en": "Avar", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "ava", "opus_code": "av"},
        {"language_ru": "даргинский", "language_en": "Dargwa", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "dar", "opus_code": "dar"},
        {"language_ru": "лезгинский", "language_en": "Lezgian", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lez", "opus_code": "lez"},
        {"language_ru": "лакский", "language_en": "Lak", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lbe", "opus_code": "lbe"},
        {"language_ru": "рутульский", "language_en": "Rutul", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "rut", "opus_code": "rut"},
        {"language_ru": "адыгейский", "language_en": "Adyghe", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "ady", "opus_code": "ady"},
        {"language_ru": "кабардино-черкесский", "language_en": "Kabardian", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "kbd", "opus_code": "kbd"},
        {"language_ru": "абазинский", "language_en": "Abaza", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "abq", "opus_code": None},
        {"language_ru": "бурятский", "language_en": "Buryat", "family": "Монгольская", "branch": "монгольская", "iso639_3": "bxr", "opus_code": "bxr"},
        {"language_ru": "калмыцкий", "language_en": "Kalmyk", "family": "Монгольская", "branch": "ойратская", "iso639_3": "xal", "opus_code": "xal"},
        {"language_ru": "эвенкийский", "language_en": "Evenki", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "evn", "opus_code": "evn"},
        {"language_ru": "нанайский", "language_en": "Nanai", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "gld", "opus_code": "gld"},
        {"language_ru": "нивхский", "language_en": "Nivkh", "family": "изолят / палеоазиатская группа", "branch": "нивхская", "iso639_3": "niv", "opus_code": None},
        {"language_ru": "чукотский", "language_en": "Chukchi", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "ckt", "opus_code": None},
        {"language_ru": "корякский", "language_en": "Koryak", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "kpy", "opus_code": None},
        {"language_ru": "алеутский", "language_en": "Aleut", "family": "эскимосско-алеутская", "branch": "алеутская", "iso639_3": "ale", "opus_code": "ale"},
        {"language_ru": "эскимосский / юпик", "language_en": "Yupik", "family": "эскимосско-алеутская", "branch": "эскимосская", "iso639_3": "ess", "opus_code": None},
    ])
    return [
        md("""
# Занятие 1. Пайплайн инвентаризации языковых датасетов

**Цель практики:** не OCR и не ASR, а стартовая карта проекта: какие языки берем в поле зрения и какие открытые данные уже можно найти.

В этой тетрадке мы собираем первоначальную информацию обычным воспроизводимым пайплайном:

1. берет редактируемый seed list основных живых языков народов России без диалектального уровня;
2. показывает, как выглядит ответ OPUS и какие поля из него достаем;
3. показывает, как выглядит карточка/ответ Hugging Face Datasets и какие поля из него достаем;
4. проверяет OPUS API на параллельные данные с русским;
5. проверяет Hugging Face Datasets как каталог опубликованных корпусов;
6. собирает таблицу, которую можно открыть в Google Sheets и дальше править руками.

Готовый снапшот этой таблицы уже создан в Google Sheets: https://docs.google.com/spreadsheets/d/1Qfr6JCB5CF-NLwQBODStqfhesrYw9tIVh2s_A6cg0d8
"""),
        code("""
!pip -q install pandas requests openpyxl beautifulsoup4 langgraph
"""),
        code(COMMON_SETUP + """
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, TypedDict
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END
"""),
        md("""
## 1. Seed list языков

Это не “истина навсегда”, а стартовая рабочая рамка для курса. Ее надо обсуждать и уточнять: какие языки добавить, где объединять варианты, где наоборот нельзя смешивать разные языковые сообщества.
"""),
        code(f"""
LANGUAGES = {languages_code}

seed_df = pd.DataFrame(LANGUAGES)
display(seed_df.groupby(['family', 'branch']).size().reset_index(name='languages'))
display(seed_df.head(12))
"""),
        md("""
## 2. Источники пайплайна: OPUS API и Hugging Face API

Здесь мы работаем со структурированными источниками готовых датасетов. Это важное ограничение: API дают воспроизводимые поля и ссылки, а веб-поиск дает только кандидатов, которые потом нужно проверять человеком.

В этом занятии мы не используем Wikipedia: это хороший источник текстовых данных, но не каталог готовых датасетов. Сейчас нас интересует именно инвентаризация уже опубликованных датасетов и корпусов.
"""),
        code("""
OPUS_API = 'https://opus.nlpl.eu/opusapi'
HF_DATASETS_API = 'https://huggingface.co/api/datasets'

def api_get(url, params, attempts=3, timeout=30):
    \"\"\"Загружает JSON из публичного API с повторами и паузами при rate limit.\"\"\"
    last_error = None
    for attempt in range(attempts):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={'User-Agent': 'lowres-course-dataset-scout/1.0'})
            if r.status_code == 429 and attempt < attempts - 1:
                wait = int(r.headers.get('Retry-After', 2 + attempt * 2))
                print('rate limit, wait', wait, 'sec')
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(1 + attempt * 2)
    raise last_error

def as_int(value):
    \"\"\"Преобразует числовые поля OPUS в int, считая пустые значения нулем.\"\"\"
    if value in ('', None):
        return 0
    return int(value)

def show_json_fragment(obj, keys=None, limit=1600):
    \"\"\"Печатает небольшой фрагмент JSON, чтобы глазами увидеть форму ответа API.\"\"\"
    if keys and isinstance(obj, dict):
        obj = {key: obj.get(key) for key in keys}
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    print(text[:limit] + ('\\n...' if len(text) > limit else ''))

def opus_pair_page_url(source, target, corpus='Tatoeba'):
    \"\"\"Собирает ссылку на человеческую страницу OPUS для корпуса и языковой пары.\"\"\"
    return f'https://opus.nlpl.eu/datasets/{corpus}?hi={source}&pair={target}'

def opus_pair_api_url(source, target):
    \"\"\"Собирает ссылку на API-запрос OPUS для языковой пары.\"\"\"
    return f'{OPUS_API}?source={source}&target={target}&preprocessing=xml&version=latest'

def hf_dataset_page_url(dataset_id):
    \"\"\"Собирает ссылку на карточку датасета на Hugging Face.\"\"\"
    return f'https://huggingface.co/datasets/{dataset_id}'

def hf_dataset_api_url(dataset_id):
    \"\"\"Собирает ссылку на API-ответ Hugging Face по одному датасету.\"\"\"
    return f'{HF_DATASETS_API}/{dataset_id}'

def dataset_search_text(dataset):
    \"\"\"Склеивает metadata HF-датасета в текст для LLM-классификации.\"\"\"
    tags = dataset.get('tags') or []
    return ' '.join([
        str(dataset.get('id', '')),
        str(dataset.get('description', '') or ''),
        ' '.join(tags),
    ]).lower()

def hf_search_datasets(params, limit=50):
    \"\"\"Возвращает список датасетов Hugging Face по одному API-запросу.\"\"\"
    query = dict(params)
    query['limit'] = limit
    data = api_get(HF_DATASETS_API, query, attempts=2, timeout=20)
    return data if isinstance(data, list) else []

def hf_datasets_for_language(language, limit_per_query=50):
    \"\"\"Ищет все видимые через API HF-кандидаты для одного языка.

    Это не скачивает сами датасеты, а собирает карточки/метаданные из каталога.
    Основной критерий здесь только HF-теги `language:*`; текст карточки не используем
    как доказательство языка.
    \"\"\"
    tags = {
        f"language:{language.get('opus_code')}" if language.get('opus_code') else '',
        f"language:{language.get('iso639_3')}" if language.get('iso639_3') else '',
    }
    tags.discard('')
    seen = {}
    errors = []

    for tag in sorted(tags):
        try:
            for dataset in hf_search_datasets({'filter': tag}, limit=limit_per_query):
                if dataset.get('id'):
                    seen[dataset['id']] = dataset
        except Exception as exc:
            errors.append({'query_type': 'filter', 'query': tag, 'error': str(exc)})
    return list(seen.values()), errors

def hf_datasets_for_pair(left_language, right_language, limit_per_query=50):
    \"\"\"Собирает HF-кандидаты для языковой пары без текстовой псевдопроверки.

    У Hugging Face нет универсального надежного фильтра “дай все датасеты ровно
    для пары X-Y”. Поэтому берем датасеты с тегами обоих языков и добавляем
    результаты прямого pair-search. Дальше классификацию делает LLM.
    \"\"\"
    seen = {}
    errors = []
    for language in [left_language, right_language]:
        datasets, language_errors = hf_datasets_for_language(language, limit_per_query=limit_per_query)
        errors.extend(language_errors)
        for dataset in datasets:
            seen[dataset['id']] = dataset

    pair_terms = [
        f"{left_language.get('language_en')} {right_language.get('language_en')}",
        f"{right_language.get('language_en')} {left_language.get('language_en')}",
        f"{left_language.get('iso639_3')}-{right_language.get('iso639_3')}",
        f"{right_language.get('iso639_3')}-{left_language.get('iso639_3')}",
        f"{left_language.get('opus_code')}-{right_language.get('opus_code')}",
        f"{right_language.get('opus_code')}-{left_language.get('opus_code')}",
    ]
    for term in [term for term in pair_terms if 'None' not in term]:
        try:
            for dataset in hf_search_datasets({'search': term}, limit=limit_per_query):
                if dataset.get('id'):
                    seen[dataset['id']] = dataset
        except Exception as exc:
            errors.append({'query_type': 'pair_search', 'query': term, 'error': str(exc)})

    left_tags = {
        f"language:{left_language.get('opus_code')}" if left_language.get('opus_code') else '',
        f"language:{left_language.get('iso639_3')}" if left_language.get('iso639_3') else '',
    }
    right_tags = {
        f"language:{right_language.get('opus_code')}" if right_language.get('opus_code') else '',
        f"language:{right_language.get('iso639_3')}" if right_language.get('iso639_3') else '',
    }
    left_tags.discard('')
    right_tags.discard('')

    pair_candidates = []
    for dataset in seen.values():
        tags = set(dataset.get('tags') or [])
        has_both_language_tags = bool(tags & left_tags) and bool(tags & right_tags)
        came_from_pair_search = any(term.lower() in dataset_search_text(dataset) for term in pair_terms if 'None' not in term)
        if has_both_language_tags or came_from_pair_search:
            pair_candidates.append(dataset)
    return pair_candidates, errors

def hf_dataset_brief_table(datasets):
    \"\"\"Делает компактную таблицу из списка HF dataset API objects.\"\"\"
    rows = []
    for dataset in datasets:
        tags = dataset.get('tags') or []
        rows.append({
            'id': dataset.get('id'),
            'downloads': dataset.get('downloads'),
            'likes': dataset.get('likes'),
            'language_tags': '; '.join(tag for tag in tags if tag.startswith('language:')),
            'size_categories': '; '.join(tag.replace('size_categories:', '') for tag in tags if tag.startswith('size_categories:')),
            'url': hf_dataset_page_url(dataset.get('id', '')),
        })
    return pd.DataFrame(rows)

"""),
        md("""
## 3. Пример OPUS: страница пары, API-ответ и извлекаемые поля

Возьмем пару `ru-udm`: русский и удмуртский. У OPUS есть человеческие страницы корпусов и API. Для пайплайна важнее API, но страница нужна, чтобы человек мог быстро открыть источник и проверить контекст: корпус, лицензию, форматы скачивания, предупреждения OPUS.
"""),
        code("""
OPUS_EXAMPLE = {
    'source': 'ru',
    'target': 'udm',
    'human_page': opus_pair_page_url('ru', 'udm', corpus='Tatoeba'),
    'api_url': opus_pair_api_url('ru', 'udm'),
}
OPUS_EXAMPLE
"""),
        code("""
try:
    opus_raw = api_get(OPUS_API, {
        'source': OPUS_EXAMPLE['source'],
        'target': OPUS_EXAMPLE['target'],
        'preprocessing': 'xml',
        'version': 'latest',
    }, attempts=1, timeout=8)
    opus_raw_source = 'live OPUS API'
except Exception as exc:
    print('OPUS API сейчас не ответил:', exc)
    opus_raw = {'corpora': [], 'error': str(exc)}
    opus_raw_source = 'OPUS error: empty result, continue'

print('Источник примера:', opus_raw_source)
print('Страница пары/корпуса для человека:', OPUS_EXAMPLE['human_page'])
print('API URL для пайплайна:', OPUS_EXAMPLE['api_url'])
show_json_fragment(opus_raw, keys=['corpora'], limit=2200)
"""),
        code("""
opus_rows = pd.DataFrame(opus_raw.get('corpora', []))
expected_opus_columns = [
    'corpus',
    'source',
    'target',
    'alignment_pairs',
    'documents',
    'preprocessing',
    'version',
]
for column in expected_opus_columns:
    if column not in opus_rows.columns:
        opus_rows[column] = ''
opus_fields_we_extract = opus_rows[expected_opus_columns].copy()
display(opus_fields_we_extract)

print('Что пайплайн кладет в итоговую таблицу:')
display(pd.DataFrame([{
    'opus_ru_parallel_pairs': opus_fields_we_extract['alignment_pairs'].map(as_int).sum(),
    'opus_ru_parallel_documents': opus_fields_we_extract['documents'].map(as_int).sum(),
    'opus_ru_parallel_corpora': '; '.join(
        f"{row.corpus} ({row.alignment_pairs})"
        for row in opus_fields_we_extract.itertuples()
    ),
    'parallel_with_russian_source': OPUS_EXAMPLE['api_url'],
}]))
"""),
        md("""
## 4. Пример Hugging Face: карточка датасета, API-ответ и извлекаемые поля

На Hugging Face у каждого датасета есть страница-карточка и API-ответ. Страница нужна человеку: посмотреть README, лицензию, файлы, ограничения доступа. API нужен пайплайну: собрать id, теги языка, размер, число примеров, downloads и признаки параллельности.
"""),
        code("""
HF_EXAMPLE_ID = 'udmurtNLP/flores-250-rus-udm'
HF_EXAMPLE = {
    'dataset_id': HF_EXAMPLE_ID,
    'human_page': hf_dataset_page_url(HF_EXAMPLE_ID),
    'api_url': hf_dataset_api_url(HF_EXAMPLE_ID),
}
HF_EXAMPLE
"""),
        code("""
hf_raw = api_get(HF_DATASETS_API + '/' + HF_EXAMPLE_ID, {}, attempts=2, timeout=20)

print('Страница датасета для человека:', HF_EXAMPLE['human_page'])
print('API URL для пайплайна:', HF_EXAMPLE['api_url'])
show_json_fragment(hf_raw, keys=['id', 'tags', 'downloads', 'likes', 'cardData', 'siblings'], limit=2600)
"""),
        code("""
card_data = hf_raw.get('cardData') or {}
dataset_info = card_data.get('dataset_info') or {}
splits = dataset_info.get('splits') or []
features = dataset_info.get('features') or []

hf_fields_we_extract = {
    'hf_dataset_id': hf_raw.get('id'),
    'hf_page': HF_EXAMPLE['human_page'],
    'hf_downloads': hf_raw.get('downloads'),
    'hf_likes': hf_raw.get('likes'),
    'hf_language_tags': '; '.join(tag for tag in hf_raw.get('tags', []) if tag.startswith('language:')),
    'hf_size_categories': '; '.join(tag.replace('size_categories:', '') for tag in hf_raw.get('tags', []) if tag.startswith('size_categories:')),
    'hf_splits': '; '.join(f"{s.get('name')} ({s.get('num_examples')} examples)" for s in splits),
    'hf_features': '; '.join(f"{f.get('name')}:{f.get('dtype')}" for f in features),
    'hf_files': '; '.join(s.get('rfilename', '') for s in hf_raw.get('siblings', [])[:5]),
}
display(pd.DataFrame([hf_fields_we_extract]).T.rename(columns={0: 'value'}))
"""),
        md("""
### 4.1. Hugging Face API: все кандидаты по языку и по языковой паре

Да, через API можно собрать не только одну карточку, а список датасетов-кандидатов для конкретного языка. Для языковой пары сложнее: у HF нет одного надежного фильтра “ровно пара ru-udm”, поэтому мы комбинируем:

- `filter=language:<code>` для каждого языка;
- `search=<название языка>` и `search=<код-код>`;
- постфильтрацию по тегам и тексту карточки.

Это не скачивает все данные. Это собирает каталог кандидатов, которые потом можно открыть, скачать или отправить на human review.
"""),
        code("""
UDMURT = {'language_ru': 'удмуртский', 'language_en': 'Udmurt', 'iso639_3': 'udm', 'opus_code': 'udm'}
RUSSIAN = {'language_ru': 'русский', 'language_en': 'Russian', 'iso639_3': 'rus', 'opus_code': 'ru'}

hf_udmurt_datasets, hf_udmurt_errors = hf_datasets_for_language(UDMURT, limit_per_query=50)
print('HF candidates for Udmurt:', len(hf_udmurt_datasets))
if hf_udmurt_errors:
    print('HF language search errors:', hf_udmurt_errors[:3])
display(hf_dataset_brief_table(hf_udmurt_datasets).head(25))

save_artifact('lesson01_hf_udmurt_candidates.csv', hf_dataset_brief_table(hf_udmurt_datasets))
"""),
        code("""
hf_ru_udm_candidates, hf_ru_udm_errors = hf_datasets_for_pair(RUSSIAN, UDMURT, limit_per_query=50)
print('HF candidates for Russian-Udmurt pair:', len(hf_ru_udm_candidates))
if hf_ru_udm_errors:
    print('HF pair search errors:', hf_ru_udm_errors[:3])
display(hf_dataset_brief_table(hf_ru_udm_candidates).head(25))

save_artifact('lesson01_hf_ru_udm_pair_candidates.csv', hf_dataset_brief_table(hf_ru_udm_candidates))
"""),
        md("""
## 5. Где заканчивается пайплайн и начинается агент

Первую часть задачи лучше решать обычным алгоритмом. У нас есть фиксированный список языков, заранее известные API, понятные поля ответа и воспроизводимые шаги обработки. Если нужно проверить конкретную языковую пару в OPUS/HF или скачать конкретный датасет, это проще, надежнее и дешевле сделать обычным кодом или руками.

Агенты становятся уместны сразу после этого: когда вход неформализован, источники заранее неизвестны, а набор решений нельзя полностью подготовить до запуска. Например: “найди все пригодные материалы для коми-пермяцкого, не перепутай его с коми-зырянским, отдели готовые датасеты от просто текстовых источников, оцени лицензионные риски и предложи, что проверить человеку”.

Такой поиск датасетов — недетерминированная исследовательская разведка: надо пройтись по выдаче, открыть страницы, понять, есть ли там датасет, найти ссылку на скачивание, проверить формат и решить, что отправить человеку на review. Ниже мы сначала соберем алгоритмическую таблицу по всем языкам, а потом для одного языка добавим аккуратного агента веб-разведки.

После OPUS/HF-примеров появляются три реальные развилки:

1. **Web discovery agent**: искать параллельные корпуса через поисковик, GitHub, страницы проектов и архивы; открывать страницы; искать скачиваемые ссылки; отправлять кандидатов на human review.
2. **HF card LLM classifier**: обкачать Hugging Face по тегам/поиску, получить много кандидатов вроде `fineweb`, `wikimedia/wikipedia` или широких multilingual-корпусов, а потом через LLM-классификацию оставить только настоящие параллельные корпуса для нужной пары.
3. **Monolingual extraction pipeline**: если большой корпус вроде FineWeb реально содержит нужный язык, отдельно строить не агент, а воспроизводимый пайплайн фильтрации, language identification, дедупликации и datacard.
"""),
        md("## 6. Функции свертки источников в наблюдения"),
        code("""
def query_opus_for_language(opus_code):
    \"\"\"Собирает сводку OPUS по моноязычным и русско-параллельным данным языка.\"\"\"
    empty = {
        'opus_ru_parallel_pairs': 0,
        'opus_ru_parallel_documents': 0,
        'opus_ru_parallel_corpora': '',
        'opus_mono_pairs_or_segments': 0,
        'opus_mono_documents': 0,
        'opus_mono_corpora': '',
    }
    if not opus_code:
        return {'opus_checked': False, **empty}
    try:
        data = api_get(OPUS_API, {
            'source': 'ru',
            'target': opus_code,
            'preprocessing': 'xml',
            'version': 'latest',
        }, attempts=1, timeout=8)
    except Exception as exc:
        return {'opus_checked': False, 'opus_error': str(exc), **empty}
    corpora = data.get('corpora', [])
    parallel = [c for c in corpora if {c.get('source'), c.get('target')} == {'ru', opus_code}]
    mono = [c for c in corpora if c.get('source') == opus_code and not c.get('target')]
    return {
        'opus_checked': True,
        'opus_ru_parallel_pairs': sum(as_int(c.get('alignment_pairs')) for c in parallel),
        'opus_ru_parallel_documents': sum(as_int(c.get('documents')) for c in parallel),
        'opus_ru_parallel_corpora': '; '.join(f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in parallel),
        'opus_mono_pairs_or_segments': sum(as_int(c.get('alignment_pairs')) for c in mono),
        'opus_mono_documents': sum(as_int(c.get('documents')) for c in mono),
        'opus_mono_corpora': '; '.join(f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in mono),
    }

def query_huggingface_for_language(row):
    \"\"\"Ищет датасеты на Hugging Face и собирает сводку вероятных ресурсов языка.

    HF-поиск обычно не дает точного количества документов или предложений.
    Поэтому сохраняем разведочные метаданные: число кандидатов, вероятные русско-
    параллельные датасеты, топ id датасетов, скачивания и категории размера из тегов.
    \"\"\"
    language_tags = {
        f"language:{row.get('opus_code')}" if row.get('opus_code') else '',
        f"language:{row.get('iso639_3')}" if row.get('iso639_3') else '',
    }
    language_tags.discard('')
    search_terms = [row.get('language_en'), row.get('language_ru')]
    seen = {}
    attempted = 0
    successful = 0
    for tag in language_tags:
        attempted += 1
        try:
            results = api_get(HF_DATASETS_API, {'filter': tag, 'limit': 10}, attempts=2, timeout=20)
        except Exception:
            continue
        if not isinstance(results, list):
            continue
        successful += 1
        for dataset in results:
            dataset_id = dataset.get('id')
            if dataset_id:
                seen[dataset_id] = dataset

    for term in [x for x in search_terms if x]:
        attempted += 1
        try:
            results = api_get(HF_DATASETS_API, {'search': term, 'limit': 10}, attempts=2, timeout=20)
        except Exception:
            continue
        if not isinstance(results, list):
            continue
        successful += 1
        for dataset in results:
            dataset_id = dataset.get('id')
            if dataset_id:
                seen[dataset_id] = dataset

    lang_texts = [
        str(row.get('language_en', '')).lower(),
        str(row.get('language_ru', '')).lower(),
    ]

    def mentions_language_name(text, names):
        \"\"\"Проверяет, встречается ли полное название языка как отдельная фраза.\"\"\"
        for name in names:
            if not name:
                continue
            for part in re.split(r'\\s*/\\s*|\\s+-\\s+', name):
                part = part.strip()
                if len(part) >= 4 and re.search(rf'(?<![\\w-]){re.escape(part)}(?![\\w-])', text):
                    return True
        return False

    datasets = []
    for dataset in seen.values():
        tags = set(dataset.get('tags') or [])
        haystack = ' '.join([
            dataset.get('id', ''),
            dataset.get('description', '') or '',
            ' '.join(tags),
        ]).lower()
        tagged = bool(tags & language_tags)
        mentioned = mentions_language_name(haystack, lang_texts)
        if tagged or mentioned:
            datasets.append(dataset)

    def is_ru_parallel(dataset):
        \"\"\"Эвристически определяет, похож ли HF-датасет на русско-параллельный.\"\"\"
        tags = set(dataset.get('tags') or [])
        text = ' '.join([
            dataset.get('id', ''),
            dataset.get('description', '') or '',
            ' '.join(tags),
        ]).lower()
        return (
            'language:ru' in tags
            or 'russian' in text
            or 'рус' in text
            or '-rus-' in text
            or 'rus-' in text
        )

    def specificity_score(dataset):
        \"\"\"Ставит языково-специфичные датасеты выше широких многоязычных коллекций.\"\"\"
        tags = set(dataset.get('tags') or [])
        text = ' '.join([
            dataset.get('id', ''),
            dataset.get('description', '') or '',
        ]).lower()
        language_tag_count = sum(1 for tag in tags if tag.startswith('language:'))
        if mentions_language_name(text, lang_texts):
            return 2
        if language_tag_count <= 5:
            return 1
        return 0

    top = sorted(
        datasets,
        key=lambda d: (specificity_score(d), d.get('downloads') or 0),
        reverse=True,
    )[:5]
    size_categories = sorted({
        tag.replace('size_categories:', '')
        for dataset in datasets
        for tag in (dataset.get('tags') or [])
        if tag.startswith('size_categories:')
    })
    return {
        'hf_checked': successful > 0,
        'hf_query_attempts': attempted,
        'hf_query_successes': successful,
        'hf_dataset_count': len(datasets),
        'hf_ru_parallel_candidates': sum(1 for dataset in datasets if is_ru_parallel(dataset)),
        'hf_top_datasets': '; '.join(dataset.get('id', '') for dataset in top),
        'hf_downloads_sum': sum(int(dataset.get('downloads') or 0) for dataset in datasets),
        'hf_size_categories': '; '.join(size_categories),
        'hf_source_url': 'https://huggingface.co/datasets',
    }
"""),
        md("## 7. Запуск пайплайна по всем языкам"),
        code("""
def scout_language(row):
    \"\"\"Собирает все наблюдения по одному языку в одну сериализуемую строку.\"\"\"
    observation = dict(row)
    observation.update(query_opus_for_language(row.get('opus_code')))
    observation.update(query_huggingface_for_language(row))
    observation['parallel_with_russian_source'] = 'https://opus.nlpl.eu/opusapi'
    observation['monolingual_source'] = 'OPUS monolingual rows; Hugging Face dataset search'
    observation['checked_at_utc'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    return observation

def run_dataset_inventory_pipeline(languages):
    \"\"\"Запускает воспроизводимый пайплайн инвентаризации датасетов.\"\"\"
    result = {
        'sources': ['OPUS API', 'Hugging Face dataset API'],
        'languages_total': len(languages),
        'observations': [],
        'errors': [],
    }
    for i, lang in enumerate(languages, 1):
        print(f"[{i}/{len(languages)}] {lang['language_ru']}")
        try:
            result['observations'].append(scout_language(lang))
        except Exception as exc:
            result['errors'].append({'language_ru': lang['language_ru'], 'error': str(exc)})
    inventory = pd.DataFrame(result['observations'])
    inventory = inventory.sort_values(['family', 'branch', 'language_ru']).reset_index(drop=True)
    result['inventory'] = inventory
    result['summary'] = {
        'languages_total': len(languages),
        'languages_checked': len(inventory),
        'with_opus_ru_parallel': int((inventory['opus_ru_parallel_pairs'] > 0).sum()),
        'with_hf_candidates': int((inventory['hf_dataset_count'] > 0).sum()),
        'errors': len(result['errors']),
    }
    return result

pipeline_result = run_dataset_inventory_pipeline(LANGUAGES)
pipeline_result['summary']
"""),
        code("""
inventory = pipeline_result['inventory']
display(inventory.head(20))
display(inventory.groupby('family')[['opus_ru_parallel_pairs', 'opus_mono_pairs_or_segments']].sum().sort_values('opus_ru_parallel_pairs', ascending=False))

save_artifact('lesson01_language_dataset_inventory.csv', inventory)
save_artifact('lesson01_dataset_inventory_summary.json', json.dumps(pipeline_result['summary'], ensure_ascii=False, indent=2))
"""),
        md("""
## 8. Google Sheets

На занятии можно открыть готовый Google Sheet и править его как общий рабочий артефакт:

https://docs.google.com/spreadsheets/d/1Qfr6JCB5CF-NLwQBODStqfhesrYw9tIVh2s_A6cg0d8

В Colab эта тетрадка сохраняет CSV в `/content/lowres_lab/lesson01_language_dataset_inventory.csv`. Его можно загрузить в Google Sheets или использовать как основу для обновления общей таблицы.
"""),
        md("""
## 9. Как автоматизировать обновление

Разовый запуск полезен для старта, но карта датасетов быстро устаревает: в OPUS появляются новые релизы, в Hugging Face загружают корпуса, национальные проекты открывают новые таблицы, а часть ссылок ломается.

Для этого нужен регулярный фоновый пайплайн:

1. **Scheduler** запускает пайплайн по расписанию: например, раз в неделю или раз в месяц.
2. **Collector** заново обходит источники готовых датасетов: OPUS, Hugging Face, GitHub-релизы, национальные корпуса, каталоги открытых данных, архивы с опубликованными корпусами.
3. **State store** хранит предыдущий снимок таблицы: CSV в GitHub, Google Sheet, SQLite или маленький JSON.
4. **Diff checker** сравнивает старую и новую версии: новые языки, новые корпуса, рост/падение counts, ошибки API.
5. **Updater** обновляет Google Sheet только для безопасных полей: counts, даты проверки, ссылки на источники.
6. **Human review** получает спорные изменения: новый источник без понятной лицензии, резкое падение counts, объединение языков/вариантов, изменение классификации.

Где здесь веб-поиск? Его можно добавить отдельным ручным или полуавтоматическим слоем обнаружения: запросы вроде `"удмуртский корпус скачать"`, `"Udmurt dataset"`, `"site:github.com udmurt corpus"`, `"site:huggingface.co/datasets udmurt"`. Но веб-поиск лучше использовать как слой кандидатов, а не как источник финальных чисел. Найденные ссылки должны попадать в лист `candidates_for_review`, пока человек не подтвердит язык, лицензию, формат, объем и надежность источника.

Самый простой стек для курса:

- `scripts/build_language_dataset_inventory.py` лежит в GitHub;
- GitHub Actions запускает его по cron;
- скрипт сохраняет новый CSV;
- отдельный шаг через Google Sheets API обновляет таблицу;
- если diff большой или появились ошибки, workflow создает issue/комментарий для ручной проверки.

Colab для такого расписания не подходит: он хорош для занятия и ручного запуска, но не для надежного фонового мониторинга.
"""),
        code("""
def compare_inventory_snapshots(old_df, new_df):
    \"\"\"Возвращает изменения по строкам между двумя снимками инвентаризации.\"\"\"
    key = 'iso639_3'
    old = old_df.set_index(key)
    new = new_df.set_index(key)
    rows = []

    for code in sorted(set(old.index) | set(new.index)):
        if code not in old.index:
            rows.append({'iso639_3': code, 'change_type': 'new_language', 'needs_human_review': True})
            continue
        if code not in new.index:
            rows.append({'iso639_3': code, 'change_type': 'missing_language', 'needs_human_review': True})
            continue

        old_pairs = int(old.loc[code, 'opus_ru_parallel_pairs'])
        new_pairs = int(new.loc[code, 'opus_ru_parallel_pairs'])
        delta = new_pairs - old_pairs
        if delta != 0:
            rows.append({
                'iso639_3': code,
                'language_ru': new.loc[code, 'language_ru'],
                'change_type': 'parallel_count_changed',
                'old_pairs': old_pairs,
                'new_pairs': new_pairs,
                'delta': delta,
                'needs_human_review': abs(delta) > max(1000, old_pairs * 0.5),
            })

    return pd.DataFrame(rows)

# Мини-демо: имитируем, что через месяц OPUS нашел больше параллельных предложений для удмуртского.
old_snapshot = inventory.copy()
new_snapshot = inventory.copy()
new_snapshot.loc[new_snapshot['iso639_3'] == 'udm', 'opus_ru_parallel_pairs'] += 250

diff = compare_inventory_snapshots(old_snapshot, new_snapshot)
display(diff)
save_artifact('lesson01_inventory_diff_demo.csv', diff)
"""),
        md("""
### Пример GitHub Actions расписания

```yaml
name: update-language-dataset-inventory

on:
  schedule:
    - cron: "0 6 1 * *"  # 1 числа каждого месяца
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install pandas openpyxl requests
      - run: python scripts/build_language_dataset_inventory.py
      - name: Update Google Sheet
        run: python scripts/update_google_sheet.py
        env:
          GOOGLE_SERVICE_ACCOUNT_JSON: ${{ secrets.GOOGLE_SERVICE_ACCOUNT_JSON }}
          SPREADSHEET_ID: "1Qfr6JCB5CF-NLwQBODStqfhesrYw9tIVh2s_A6cg0d8"
```

В реальном проекте секреты Google API нельзя хранить в notebook. Их кладут в GitHub Secrets, Google Cloud Secret Manager или другой защищенный secret store.
"""),
        md("""
## 10. Агент веб-разведки для одного языка

Теперь берем место, где агенты действительно в тему. OPUS/HF-пайплайн выше работает по известным API. Но реальная разведка датасетов часто начинается с неформализованного запроса:

> Найди открытые датасеты для удмуртского языка. Не считай Wikipedia готовым датасетом. Отделяй готовые датасеты от просто источников текстов. Если видишь GitHub, архив, корпус или страницу проекта, попробуй понять, можно ли скачать данные и что должен проверить человек.

Это уже не таблица по заранее известной схеме. Здесь агенту нужны:

- **state**: цель, язык, поисковые запросы, найденные страницы, кандидаты, ошибки;
- **tools**: web search, открытие страницы, извлечение ссылок, probe скачивания;
- **policy**: когда страницу считать кандидатом в датасет, когда отправлять на human review;
- **LLM-слой**: опционально, для планирования запросов и классификации неоднозначных страниц.

Ниже все ограничено специально для занятия: один язык, несколько запросов, несколько результатов, маленькие скачивания/probe, без автономного бесконечного браузинга.
"""),
        code("""
DISCOVERY_LANGUAGE = {
    'language_ru': 'удмуртский',
    'language_en': 'Udmurt',
    'iso639_3': 'udm',
    'opus_code': 'udm',
}

DISCOVERY_GOAL = '''
Найти дополнительные открытые датасеты или корпуса для удмуртского языка.
Не считать Wikipedia готовым датасетом. Отделять готовые датасеты от просто
источников текстов. Для каждого кандидата записать evidence, возможную лицензию,
тип данных и что должен проверить человек.
'''

MAX_QUERIES = 4
MAX_RESULTS_PER_QUERY = 4
MAX_PAGES_TO_OPEN = 10
MAX_DOWNLOAD_BYTES = 200_000
"""),
        md("### 10.1. Tools: поиск, чтение страниц, проверка ссылок"),
        code("""
def normalize_ddg_url(href):
    \"\"\"Достает настоящий URL из redirect-ссылки DuckDuckGo, если он там спрятан.\"\"\"
    if not href:
        return ''
    parsed = urlparse(href)
    if 'duckduckgo.com' in parsed.netloc and parsed.path.startswith('/l/'):
        target = parse_qs(parsed.query).get('uddg', [''])[0]
        return unquote(target)
    return href

def web_search(query, max_results=5):
    \"\"\"Ищет страницы через DuckDuckGo HTML и возвращает короткий список результатов.

    Это не официальный Google Search API, а учебный бесплатный инструмент.
    В production лучше использовать SerpAPI, Google Custom Search API, Brave Search API
    или другой легальный поисковый API с понятными лимитами.
    \"\"\"
    url = 'https://duckduckgo.com/html/'
    try:
        r = requests.get(
            url,
            params={'q': query},
            headers={'User-Agent': 'Mozilla/5.0 lowres-course-dataset-discovery/1.0'},
            timeout=20,
        )
        r.raise_for_status()
    except Exception as exc:
        return [{'query': query, 'error': str(exc)}]

    if 'Unfortunately, bots use DuckDuckGo too' in r.text or 'anomaly-modal' in r.text:
        return [{
            'query': query,
            'error': 'DuckDuckGo challenge; no fallback used. Try later or replace web_search with an official search API.',
        }]

    soup = BeautifulSoup(r.text, 'html.parser')
    results = []
    for node in soup.select('a.result__a')[:max_results]:
        href = normalize_ddg_url(node.get('href'))
        title = ' '.join(node.get_text(' ', strip=True).split())
        snippet_node = node.find_parent('div', class_='result')
        snippet = ''
        if snippet_node:
            snippet = ' '.join(snippet_node.get_text(' ', strip=True).split())
        if href:
            results.append({
                'query': query,
                'title': title,
                'url': href,
                'snippet': snippet[:500],
                'search_mode': 'duckduckgo_html',
            })
    return results

def fetch_page(url, max_chars=8000):
    \"\"\"Открывает HTML-страницу и возвращает текст, ссылки и технические метаданные.\"\"\"
    try:
        r = requests.get(
            url,
            headers={'User-Agent': 'Mozilla/5.0 lowres-course-dataset-discovery/1.0'},
            timeout=20,
        )
        content_type = r.headers.get('Content-Type', '')
        status = r.status_code
        r.raise_for_status()
    except Exception as exc:
        return {
            'url': url,
            'status': 'error',
            'error': str(exc),
            'text': '',
            'links': [],
        }

    soup = BeautifulSoup(r.text, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    text = ' '.join(soup.get_text(' ', strip=True).split())[:max_chars]
    links = []
    for a in soup.find_all('a', href=True):
        link_url = urljoin(url, a['href'])
        label = ' '.join(a.get_text(' ', strip=True).split())
        links.append({'url': link_url, 'label': label[:160]})

    return {
        'url': url,
        'status': status,
        'content_type': content_type,
        'text': text,
        'links': links[:80],
    }

DOWNLOAD_HINTS = (
    '.zip', '.tar.gz', '.tgz', '.csv', '.tsv', '.json', '.jsonl', '.txt',
    '.xml', '.conllu', '.parquet', '.xlsx', '.wav', '.mp3'
)

def likely_download_link(link):
    \"\"\"Проверяет по URL и подписи, похожа ли ссылка на скачивание данных.\"\"\"
    url = link.get('url', '').lower()
    label = link.get('label', '').lower()
    joined = f'{url} {label}'
    return (
        any(hint in url for hint in DOWNLOAD_HINTS)
        or 'download' in joined
        or 'raw.githubusercontent.com' in url
        or '/resolve/' in url
        or 'releases/download' in url
    )

def probe_download(url, max_bytes=MAX_DOWNLOAD_BYTES):
    \"\"\"Аккуратно проверяет, доступна ли ссылка на данные, не скачивая большие файлы.\"\"\"
    result = {'download_url': url}
    try:
        head = requests.head(
            url,
            allow_redirects=True,
            timeout=15,
            headers={'User-Agent': 'Mozilla/5.0 lowres-course-dataset-discovery/1.0'},
        )
        result.update({
            'head_status': head.status_code,
            'content_type': head.headers.get('Content-Type', ''),
            'content_length': head.headers.get('Content-Length', ''),
        })
        length = int(head.headers.get('Content-Length') or 0)
        if length and length > max_bytes:
            result['probe_status'] = 'too_large_for_class_demo'
            return result
    except Exception as exc:
        result['head_error'] = str(exc)

    try:
        get = requests.get(
            url,
            stream=True,
            timeout=20,
            headers={'User-Agent': 'Mozilla/5.0 lowres-course-dataset-discovery/1.0'},
        )
        chunk = next(get.iter_content(chunk_size=min(max_bytes, 4096)), b'')
        result.update({
            'get_status': get.status_code,
            'sample_bytes': len(chunk),
            'sample_text': chunk[:500].decode('utf-8', errors='replace'),
            'probe_status': 'sample_downloaded',
        })
    except Exception as exc:
        result['get_error'] = str(exc)
        result['probe_status'] = 'probe_failed'
    return result
"""),
        md("### 10.2. Policy и опциональная LLM-классификация через OpenRouter"),
        code("""
DATASET_WORDS = [
    'dataset', 'corpus', 'parallel corpus', 'monolingual corpus', 'treebank',
    'download', 'github', 'huggingface', 'csv', 'jsonl', 'conllu', 'archive',
    'датасет', 'корпус', 'параллельный корпус', 'скачать', 'данные',
]

SOURCE_ONLY_WORDS = [
    'wikipedia', 'encyclopedia', 'news', 'article', 'blog', 'dictionary only',
    'википедия', 'новость', 'статья',
]

def heuristic_classify_page(page, language):
    \"\"\"Классифицирует страницу простыми правилами, если LLM-ключа нет.\"\"\"
    text = page.get('text', '').lower()
    url = page.get('url', '').lower()
    lang_hits = sum(token in text or token in url for token in [
        language['language_en'].lower(),
        language['language_ru'].lower(),
        language['iso639_3'].lower(),
    ])
    dataset_hits = sum(word in text or word in url for word in DATASET_WORDS)
    source_only_hits = sum(word in text or word in url for word in SOURCE_ONLY_WORDS)
    download_links = [link for link in page.get('links', []) if likely_download_link(link)]

    if dataset_hits >= 2 and lang_hits:
        label = 'dataset_candidate'
    elif dataset_hits >= 1 and download_links:
        label = 'possible_dataset_candidate'
    elif source_only_hits:
        label = 'source_or_reference_only'
    else:
        label = 'unclear'

    review = []
    if download_links:
        review.append('проверить скачиваемые ссылки и формат')
    if 'license' in text or 'лиценз' in text:
        review.append('проверить лицензию')
    else:
        review.append('лицензия не найдена автоматически')
    if source_only_hits:
        review.append('не считать источником финальных чисел без ручной проверки')

    return {
        'label': label,
        'confidence': 'medium' if label in {'dataset_candidate', 'source_or_reference_only'} else 'low',
        'dataset_type': 'unknown',
        'evidence': page.get('text', '')[:500],
        'needs_human_review': True,
        'human_review_note': '; '.join(review),
        'mode': 'heuristic',
    }

def call_openrouter_json(prompt, model='openai/gpt-oss-20b:free'):
    \"\"\"Вызывает OpenRouter и ожидает JSON-ответ, если доступен OPENROUTER_API_KEY.\"\"\"
    try:
        from google.colab import userdata
        api_key = userdata.get('OPENROUTER_API_KEY')
    except Exception:
        api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        return None

    response = requests.post(
        'https://openrouter.ai/api/v1/chat/completions',
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        json={
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'Ты аккуратно классифицируешь страницы про языковые датасеты. Отвечай только JSON.'},
                {'role': 'user', 'content': prompt},
            ],
            'temperature': 0.1,
        },
        timeout=60,
    )
    response.raise_for_status()
    raw = response.json()['choices'][0]['message']['content']
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {'label': 'llm_unparsed', 'raw_answer': raw, 'needs_human_review': True}

def classify_page(page, language, goal):
    \"\"\"Классифицирует страницу через LLM, а без ключа использует эвристики.\"\"\"
    prompt = f\"\"\"Цель:
{goal}

Язык:
{json.dumps(language, ensure_ascii=False)}

Страница:
URL: {page.get('url')}
TEXT:
{page.get('text', '')[:5000]}

Верни JSON:
{{
  "label": "dataset_candidate | possible_dataset_candidate | source_or_reference_only | irrelevant | unclear",
  "confidence": "high | medium | low",
  "dataset_type": "parallel_text | monolingual_text | speech | dictionary | ocr | mixed | unknown",
  "evidence": "короткая цитата/пересказ признаков",
  "license_hint": "что видно про лицензию или пусто",
  "needs_human_review": true,
  "human_review_note": "что именно проверить человеку"
}}
\"\"\"
    llm_answer = call_openrouter_json(prompt)
    if llm_answer is not None:
        llm_answer['mode'] = 'openrouter'
        return llm_answer
    return heuristic_classify_page(page, language)
"""),
        md("### 10.3. Сценарий 2: LLM-классификатор HF-карточек"),
        md("""
Перед веб-поиском можно сделать более контролируемую LLM-задачу: взять кандидатов из HF API и отсеять широкие корпуса, которые только содержат языковой тег, но не являются параллельным корпусом для нужной пары.

Например, `fineweb`, `wikimedia/wikipedia`, `GlotCC` или `DCAD` могут быть полезны для моноязычного сценария, но они не становятся русско-удмуртским параллельным корпусом только потому, что где-то содержат `language:udm` или слово `Udmurt`.
Это не агент: здесь нет планирования, выбора tools и изменения стратегии. Это LLM-классификация заранее собранного списка.
"""),
        code("""
def needs_openrouter_classification(dataset):
    \"\"\"Возвращает честную заглушку, если OpenRouter-ключ не подключен.\"\"\"
    tags = dataset.get('tags') or []
    return {
        'dataset_id': dataset.get('id'),
        'label': 'needs_openrouter_key',
        'confidence': 'none',
        'reason': 'классификация параллельности отключена: нет OPENROUTER_API_KEY',
        'evidence': dataset_search_text(dataset)[:500],
        'needs_human_review': True,
        'downloads': dataset.get('downloads'),
        'language_tags': '; '.join(tag for tag in tags if tag.startswith('language:')),
        'task_tags': '; '.join(tag for tag in tags if tag.startswith('task_categories:')),
        'url': hf_dataset_page_url(dataset.get('id', '')),
        'mode': 'no_llm',
    }

def classify_hf_parallel_candidate(dataset, left_language, right_language):
    \"\"\"Классифицирует HF-карточку через OpenRouter и возвращает JSON-решение.\"\"\"
    compact_text = dataset_search_text(dataset)
    prompt = f\"\"\"Нужно понять, является ли Hugging Face dataset настоящим параллельным корпусом для языковой пары.

Левый язык:
{json.dumps(left_language, ensure_ascii=False)}

Правый язык:
{json.dumps(right_language, ensure_ascii=False)}

Dataset object:
{json.dumps({
    'id': dataset.get('id'),
    'description': dataset.get('description'),
    'tags': dataset.get('tags'),
    'metadata_text': compact_text[:3000],
    'downloads': dataset.get('downloads'),
    'likes': dataset.get('likes'),
}, ensure_ascii=False)[:5000]}

Верни JSON:
{{
  "dataset_id": "...",
  "label": "parallel_corpus | parallel_candidate_needs_review | not_parallel_broad_multilingual | language_related_not_parallel | irrelevant_or_unclear",
  "confidence": "high | medium | low",
  "reason": "почему",
  "evidence": "короткие признаки из карточки",
  "needs_human_review": true
}}
\"\"\"
    llm_answer = call_openrouter_json(prompt)
    if llm_answer is not None:
        llm_answer['dataset_id'] = llm_answer.get('dataset_id') or dataset.get('id')
        llm_answer['url'] = hf_dataset_page_url(dataset.get('id', ''))
        llm_answer['mode'] = 'openrouter'
        return llm_answer
    return needs_openrouter_classification(dataset)

def run_hf_card_review_classifier(datasets, left_language, right_language, max_cards=30):
    \"\"\"Проходит по HF-кандидатам и классифицирует параллельность через LLM.\"\"\"
    state = {
        'goal': 'отфильтровать HF-кандидаты до параллельных корпусов для языковой пары',
        'left_language': left_language,
        'right_language': right_language,
        'cards_total': len(datasets),
        'cards_reviewed': 0,
        'decisions': [],
    }
    for dataset in datasets[:max_cards]:
        decision = classify_hf_parallel_candidate(dataset, left_language, right_language)
        state['decisions'].append(decision)
        state['cards_reviewed'] += 1
    return state

hf_review_state = run_hf_card_review_classifier(hf_ru_udm_candidates, RUSSIAN, UDMURT, max_cards=30)
hf_review_df = pd.DataFrame(hf_review_state['decisions'])
display(hf_review_df.sort_values(['label', 'confidence']).head(30))
save_artifact('lesson01_hf_parallel_review_classifier.csv', hf_review_df)
"""),
        code("""
if len(hf_review_df):
    display(hf_review_df['label'].value_counts().reset_index(name='count').rename(columns={'index': 'label'}))
    display(hf_review_df[hf_review_df['label'].isin(['parallel_corpus', 'parallel_candidate_needs_review'])][[
        'dataset_id', 'label', 'confidence', 'reason', 'url'
    ]])
"""),
        md("""
Это не агентная задача: вход уже собран алгоритмом, tools не выбираются, стратегия не меняется. Но это хороший пример LLM-классификации: модель читает metadata карточек и возвращает JSON с решением, является ли карточка параллельным корпусом или широким многоязычным/моноязычным ресурсом.
"""),
        md("### 10.4. Сценарий 1: агент руками для web discovery"),
        code("""
def plan_search_queries(language, goal):
    \"\"\"Планирует несколько поисковых запросов для одного языка.\"\"\"
    prompt = f\"\"\"Составь до 4 поисковых запросов для поиска датасетов языка.
Язык: {json.dumps(language, ensure_ascii=False)}
Цель: {goal}
Верни JSON: {{"queries": ["..."]}}
\"\"\"
    llm_answer = call_openrouter_json(prompt)
    if llm_answer and isinstance(llm_answer.get('queries'), list):
        return llm_answer['queries'][:MAX_QUERIES]
    return [
        f"{language['language_en']} dataset corpus",
        f"{language['language_en']} Russian parallel corpus",
        f"{language['language_en']} language GitHub corpus",
        f"{language['language_ru']} корпус скачать датасет",
    ][:MAX_QUERIES]

def run_manual_discovery_agent(language, goal):
    \"\"\"Запускает маленького агента веб-разведки без фреймворка.\"\"\"
    state = {
        'goal': goal,
        'language': language,
        'queries': [],
        'search_results': [],
        'pages': [],
        'candidates': [],
        'download_probes': [],
        'errors': [],
    }
    state['queries'] = plan_search_queries(language, goal)

    seen_urls = set()
    for query in state['queries']:
        print('search:', query)
        results = web_search(query, max_results=MAX_RESULTS_PER_QUERY)
        state['search_results'].extend(results)
        for item in results:
            url = item.get('url')
            if not url or url in seen_urls or item.get('error'):
                continue
            seen_urls.add(url)
            if len(state['pages']) >= MAX_PAGES_TO_OPEN:
                break
            page = fetch_page(url)
            state['pages'].append(page)
            if page.get('status') == 'error':
                state['errors'].append({'url': url, 'error': page.get('error')})
                continue
            classification = classify_page(page, language, goal)
            candidate = {
                'language_ru': language['language_ru'],
                'language_en': language['language_en'],
                'url': url,
                'title': item.get('title', ''),
                'query': query,
                **classification,
            }
            candidate_links = [link for link in page.get('links', []) if likely_download_link(link)][:3]
            candidate['download_link_count'] = len(candidate_links)
            candidate['download_links'] = '; '.join(link['url'] for link in candidate_links)
            state['candidates'].append(candidate)
            for link in candidate_links[:1]:
                state['download_probes'].append({
                    'page_url': url,
                    **probe_download(link['url']),
                })
    return state

manual_agent_state = run_manual_discovery_agent(DISCOVERY_LANGUAGE, DISCOVERY_GOAL)
print('pages opened:', len(manual_agent_state['pages']))
print('candidates:', len(manual_agent_state['candidates']))
print('download probes:', len(manual_agent_state['download_probes']))
"""),
        code("""
manual_candidates = pd.DataFrame(manual_agent_state['candidates'])
if len(manual_candidates):
    display(manual_candidates[[
        'language_ru', 'label', 'confidence', 'dataset_type',
        'title', 'url', 'download_link_count', 'human_review_note', 'mode'
    ]].head(20))
    save_artifact('lesson01_manual_agent_candidates.csv', manual_candidates)

manual_probes = pd.DataFrame(manual_agent_state['download_probes'])
if len(manual_probes):
    display(manual_probes.head(10))
    save_artifact('lesson01_manual_agent_download_probes.csv', manual_probes)
"""),
        md("""
Что здесь агентского:

- вход свободный, а не набор параметров API;
- агент сам превращает цель в поисковые запросы;
- список страниц заранее неизвестен;
- для каждой страницы надо принять неоднозначное решение: датасет, источник текстов, мусор или кандидат на review;
- если на странице есть скачиваемые ссылки, агент сам выбирает, что аккуратно проверить;
- результат не финальная истина, а очередь для human review.

OpenRouter здесь отвечает только за LLM-решения. Он не гуглит и не скачивает сам. Поиск, чтение страниц и probe скачивания — это tools, которые мы написали отдельно.
"""),
        md("### 10.5. Та же web-разведка в LangGraph"),
        code("""
class DiscoveryState(TypedDict, total=False):
    goal: str
    language: Dict[str, Any]
    queries: List[str]
    search_results: List[Dict[str, Any]]
    pages: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    download_probes: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    report: Dict[str, Any]

def plan_node(state: DiscoveryState):
    \"\"\"Планирует поисковые запросы из свободной цели и языка.\"\"\"
    return {'queries': plan_search_queries(state['language'], state['goal'])}

def search_node(state: DiscoveryState):
    \"\"\"Запускает web_search по всем запланированным запросам.\"\"\"
    results = []
    for query in state['queries'][:MAX_QUERIES]:
        results.extend(web_search(query, max_results=MAX_RESULTS_PER_QUERY))
    return {'search_results': results}

def inspect_node(state: DiscoveryState):
    \"\"\"Открывает найденные страницы, классифицирует их и проверяет ссылки на данные.\"\"\"
    pages = []
    candidates = []
    probes = []
    errors = []
    seen = set()
    for item in state['search_results']:
        url = item.get('url')
        if not url or url in seen or item.get('error'):
            continue
        seen.add(url)
        if len(pages) >= MAX_PAGES_TO_OPEN:
            break
        page = fetch_page(url)
        pages.append(page)
        if page.get('status') == 'error':
            errors.append({'url': url, 'error': page.get('error')})
            continue
        classification = classify_page(page, state['language'], state['goal'])
        links = [link for link in page.get('links', []) if likely_download_link(link)][:3]
        candidates.append({
            'language_ru': state['language']['language_ru'],
            'language_en': state['language']['language_en'],
            'url': url,
            'title': item.get('title', ''),
            'query': item.get('query', ''),
            **classification,
            'download_link_count': len(links),
            'download_links': '; '.join(link['url'] for link in links),
        })
        for link in links[:1]:
            probes.append({'page_url': url, **probe_download(link['url'])})
    return {'pages': pages, 'candidates': candidates, 'download_probes': probes, 'errors': errors}

def report_node(state: DiscoveryState):
    \"\"\"Собирает короткий отчет по результатам разведки.\"\"\"
    candidates = state.get('candidates', [])
    useful = [c for c in candidates if c.get('label') in {'dataset_candidate', 'possible_dataset_candidate'}]
    return {'report': {
        'language_ru': state['language']['language_ru'],
        'queries': state.get('queries', []),
        'search_results': len(state.get('search_results', [])),
        'pages_opened': len(state.get('pages', [])),
        'candidates_total': len(candidates),
        'dataset_candidates': len(useful),
        'download_probes': len(state.get('download_probes', [])),
        'errors': len(state.get('errors', [])),
    }}

discovery_graph = StateGraph(DiscoveryState)
discovery_graph.add_node('plan', plan_node)
discovery_graph.add_node('search', search_node)
discovery_graph.add_node('inspect', inspect_node)
discovery_graph.add_node('report', report_node)
discovery_graph.set_entry_point('plan')
discovery_graph.add_edge('plan', 'search')
discovery_graph.add_edge('search', 'inspect')
discovery_graph.add_edge('inspect', 'report')
discovery_graph.add_edge('report', END)

dataset_discovery_agent = discovery_graph.compile()
graph_agent_state = dataset_discovery_agent.invoke({
    'language': DISCOVERY_LANGUAGE,
    'goal': DISCOVERY_GOAL,
})
graph_agent_state['report']
"""),
        code("""
graph_candidates = pd.DataFrame(graph_agent_state['candidates'])
if len(graph_candidates):
    display(graph_candidates[[
        'language_ru', 'label', 'confidence', 'dataset_type',
        'title', 'url', 'download_link_count', 'human_review_note', 'mode'
    ]].head(20))
    save_artifact('lesson01_langgraph_agent_candidates.csv', graph_candidates)

graph_probes = pd.DataFrame(graph_agent_state['download_probes'])
if len(graph_probes):
    display(graph_probes.head(10))
    save_artifact('lesson01_langgraph_agent_download_probes.csv', graph_probes)
"""),
        md("""
LangGraph не делает агента умнее сам по себе. Его смысл здесь в другом: он явно фиксирует архитектуру и состояние.

В plain Python версии мы сами держали порядок шагов в цикле. В LangGraph версии те же решения разложены по узлам:

1. `plan`: превратить свободную цель в поисковые запросы;
2. `search`: собрать выдачу;
3. `inspect`: открыть страницы, классифицировать, проверить ссылки;
4. `report`: собрать краткий отчет.

Дальше этот граф можно расширять: добавить условный повтор поиска, отдельного license critic, human-in-the-loop узел, запись в Google Sheet, ограничение бюджета и сохранение state между запусками.
"""),
        md("""
### 10.6. Сценарий 3: моноязычные данные из больших корпусов

Это уже скорее не агент, а отдельный воспроизводимый pipeline. Если HF-card review показывает, что `fineweb`, `GlotCC`, `DCAD` или другой большой корпус содержит нужный язык, дальше задача меняется:

1. скачать или стримить только нужный shard/конфигурацию;
2. прогнать language identification;
3. отфильтровать строки нужного языка;
4. убрать дубликаты, boilerplate и слишком короткие/битые тексты;
5. посчитать объем, примеры, домены, риски;
6. оформить datacard и список ограничений.

Агент может помочь спроектировать такой pipeline для конкретного корпуса или проверить datacard, но сам extraction лучше делать обычным кодом: так результат воспроизводим и его можно перепроверить.
"""),
        md("""
## Вопросы для отчета

1. Какие языковые семьи в таблице оказываются лучше всего покрыты параллельными данными с русским?
2. Где есть Википедия, но почти нет параллельных данных?
3. Где OPUS показывает нули: это значит “данных нет” или “мы не нашли правильный код/источник”?
4. Какие источники надо добавить следующими: национальные корпуса, сайты СМИ, библиотеки, архивы, Hugging Face, GitHub?
5. Какие результаты HF-поиска выглядят полезными, но требуют ручной проверки?
6. Какие веб-поисковые запросы вы бы добавили для своего языка?
7. Какие поля можно обновлять автоматически, а какие требуют human review?
8. Как часто стоит запускать фоновое обновление для такой таблицы и почему?
9. Чем веб-разведка датасетов отличается от OPUS/HF-пайплайна?
10. Какие решения агента в этом примере обязательно должен проверить человек?
"""),
    ]


NOTEBOOKS = {
    "01_dataset_inventory_pipeline.ipynb": lesson01_dataset_scout_cells(),
    "02_web_scraping_sources.ipynb": [
        md("""
# Занятие 2. Веб-сбор, источники и правовая рамка

**Цель практики:** собрать маленькую таблицу источников через открытые API, извлечь текст, проверить шум и зафиксировать лицензионные/этические риски.

Работает в бесплатном Colab на CPU. Используем Wikipedia API вместо агрессивного скрейпинга сайтов.
"""),
        code(COMMON_SETUP),
        md("## 1. Получаем страницы из Удмуртской Википедии"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

def search_pages(query, limit=10):
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'srlimit': limit,
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    return r.json()['query']['search']

hits = search_pages('удмурт', 10)
sources = pd.DataFrame([{
    'title': h['title'],
    'pageid': h['pageid'],
    'url': f'https://{LANG}.wikipedia.org/wiki/' + requests.utils.quote(h['title'].replace(' ', '_')),
    'type': 'wiki_article',
    'license_or_access': 'Wikipedia text: CC BY-SA, check page history and terms',
    'notes': 'API search result',
} for h in hits])
show_df(sources, 10)
save_artifact('lesson02_sources.csv', sources)
"""),
        md("## 2. Извлекаем тексты и считаем базовый шум"),
        code("""
def get_extract(pageid):
    params = {
        'action': 'query',
        'pageids': pageid,
        'prop': 'extracts|info',
        'explaintext': 1,
        'inprop': 'url',
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    page = next(iter(r.json()['query']['pages'].values()))
    return page.get('extract', ''), page.get('fullurl')

texts = []
for _, row in sources.iterrows():
    txt, url = get_extract(row.pageid)
    texts.append({'pageid': row.pageid, 'title': row.title, 'url': url, 'text': txt, 'chars': len(txt)})

corpus = pd.DataFrame(texts)
corpus['cyrillic_share'] = corpus['text'].apply(lambda x: len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', x)) / max(len(x), 1))
corpus['line_count'] = corpus['text'].apply(lambda x: len([l for l in x.splitlines() if l.strip()]))
show_df(corpus[['title', 'chars', 'cyrillic_share', 'line_count', 'url']], 10)
save_artifact('lesson02_raw_corpus.csv', corpus)
"""),
        md("## 3. Простая фильтрация и отчет агента-ревьюера"),
        code("""
filtered = corpus[(corpus['chars'] >= 200) & (corpus['cyrillic_share'] > 0.45)].copy()
filtered['duplicate_text'] = filtered.duplicated('text')

review = {
    'n_sources': len(sources),
    'n_texts_kept': len(filtered),
    'too_short': int((corpus['chars'] < 200).sum()),
    'low_cyrillic_share': int((corpus['cyrillic_share'] <= 0.45).sum()),
    'duplicates': int(filtered['duplicate_text'].sum()),
    'human_checks': [
        'проверить лицензию и условия использования',
        'проверить, что текст действительно на нужном языке',
        'убрать навигацию, списки и служебные фрагменты',
    ],
}
print(json.dumps(review, ensure_ascii=False, indent=2))
save_artifact('lesson02_filtered_corpus.csv', filtered)
save_artifact('lesson02_review.json', json.dumps(review, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Какие страницы пришлось бы исключить и почему?
2. Достаточно ли API-источника для курса или нужны другие сайты/архивы?
3. Какие поля метаданных нужно добавить в `sources.csv`?
4. Что агент может сделать сам, а что должен подтвердить человек?
"""),
    ],
    "03_ocr_udmurt_commons.ipynb": [
        md("""
# Занятие 3. OCR для открытого скана на малоресурсном языке России

**Цель практики:** взять открытое изображение из Wikimedia Commons, прогнать лёгкий OCR baseline и понять, получается ли текст на вменяемом языке.

Пример использует материалы категории **Udmurt Dunne** на Wikimedia Commons и Tesseract с русской OCR-моделью как baseline для кириллического удмуртского текста. Это не “правильная удмуртская OCR-модель”, а проверка: насколько далеко можно зайти простым бесплатным инструментом.

Работает в бесплатном Colab на CPU.
"""),
        code("""
!apt-get -qq update
!apt-get -qq install -y tesseract-ocr tesseract-ocr-rus
!pip -q install pytesseract pillow opencv-python-headless pandas matplotlib
"""),
        code(COMMON_SETUP + "\nfrom PIL import Image\nimport pytesseract\nimport matplotlib.pyplot as plt\nimport cv2\n"),
        md("## 1. Находим файл в Wikimedia Commons через API"),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
FILE_TITLE = 'File:Удномер.jpg'  # из категории Wikimedia Commons: Udmurt Dunne

params = {
    'action': 'query',
    'titles': FILE_TITLE,
    'prop': 'imageinfo',
    'iiprop': 'url|mime|size|extmetadata',
    'format': 'json',
}
r = requests.get(COMMONS_API, params=params, timeout=30)
r.raise_for_status()
page = next(iter(r.json()['query']['pages'].values()))
info = page['imageinfo'][0]
image_url = info['url']
print(image_url)
print('license:', info.get('extmetadata', {}).get('LicenseShortName', {}).get('value'))
"""),
        md("## 2. Скачиваем изображение и смотрим на него"),
        code("""
img_path = DATA_DIR / 'udmurt_sample.jpg'
img_path.write_bytes(requests.get(image_url, timeout=60).content)
img = Image.open(img_path)
print(img.size)
plt.figure(figsize=(6, 8))
plt.imshow(img)
plt.axis('off');
"""),
        md("## 3. Запускаем Tesseract baseline"),
        code("""
raw_text = pytesseract.image_to_string(img, lang='rus')
print(raw_text[:2000])
save_artifact('lesson03_ocr_raw.txt', raw_text)
"""),
        md("## 4. Улучшаем препроцессинг и сравниваем"),
        code("""
gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.fastNlMeansDenoising(gray, h=20)
thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
prep_path = DATA_DIR / 'udmurt_sample_preprocessed.png'
cv2.imwrite(str(prep_path), thr)

prep_img = Image.open(prep_path)
prep_text = pytesseract.image_to_string(prep_img, lang='rus')
print(prep_text[:2000])
save_artifact('lesson03_ocr_preprocessed.txt', prep_text)

plt.figure(figsize=(6, 8))
plt.imshow(prep_img, cmap='gray')
plt.axis('off');
"""),
        md("## 5. Быстрая диагностика качества без ground truth"),
        code("""
def ocr_diagnostics(text):
    chars = len(text)
    letters = re.findall(r'[А-Яа-яЁёӐ-ӿ]', text)
    tokens = re.findall(r'[А-Яа-яЁёӐ-ӿ]{2,}', text)
    weird = re.findall(r'[^А-Яа-яЁёӐ-ӿ0-9\\s.,:;!?()\\-—"«»]', text)
    return {
        'chars': chars,
        'cyrillic_letters': len(letters),
        'tokens_2plus': len(tokens),
        'unique_tokens_2plus': len(set(t.lower() for t in tokens)),
        'weird_char_share': len(weird) / max(chars, 1),
    }

diag = pd.DataFrame([
    {'version': 'raw', **ocr_diagnostics(raw_text)},
    {'version': 'preprocessed', **ocr_diagnostics(prep_text)},
])
show_df(diag)
save_artifact('lesson03_ocr_diagnostics.csv', diag)
"""),
        md("""
## 6. OCR как маленький агентский workflow

Этот блок переносит агентскую идею из вводного занятия туда, где ей место: в OCR-задачу. Здесь `state` хранит источник, версии OCR, диагностику и решение о следующем человеческом шаге.
"""),
        code("""
def decide_next_ocr_step(diag_df):
    best = diag_df.sort_values(['tokens_2plus', 'weird_char_share'], ascending=[False, True]).iloc[0]
    if best['tokens_2plus'] < 20:
        return 'Нужен другой скан или ручная разметка маленького ground truth: baseline почти не читает текст.'
    if best['weird_char_share'] > 0.1:
        return 'Нужна дополнительная предобработка и проверка символов.'
    return 'Можно отобрать 30-50 строк для ручной оценки ошибок OCR.'

ocr_agent_state = {
    'task': 'оценить OCR baseline для открытого удмуртского скана',
    'source_file': FILE_TITLE,
    'source_url': image_url,
    'versions': {
        'raw': raw_text,
        'preprocessed': prep_text,
    },
    'diagnostics': diag.to_dict('records'),
    'next_human_step': decide_next_ocr_step(diag),
}

print(json.dumps(ocr_agent_state, ensure_ascii=False, indent=2)[:3000])
save_artifact('lesson03_ocr_agent_state.json', json.dumps(ocr_agent_state, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Можно ли читать результат глазами? Какие слова/буквы распознаются хуже всего?
2. Улучшил ли препроцессинг результат?
3. Почему русская OCR-модель может ошибаться на удмуртском?
4. Какое поле `state` оказалось самым важным для решения о следующем шаге?
5. Какие 30-50 строк стоит вручную разметить как ground truth для следующего шага?
"""),
    ],
    "04_asr_udmurt_whisper_tiny.ipynb": [
        md("""
# Занятие 4. ASR baseline на открытом аудио удмуртской речи

**Цель практики:** взять короткий открытый аудиофайл на удмуртском языке с Wikimedia Commons, прогнать маленькую ASR-модель и понять, получается ли осмысленная транскрипция.

Используем `openai/whisper-tiny` через Transformers. Это маленькая модель; она удобна для бесплатного Colab, но может плохо работать с языками, которых нет в её сильных режимах. Это и есть предмет анализа.
"""),
        code("""
!pip -q install transformers accelerate librosa soundfile pandas
"""),
        code(COMMON_SETUP + "\nimport librosa, soundfile as sf\nfrom IPython.display import Audio\nfrom transformers import pipeline\n"),
        md("## 1. Скачиваем открытое аудио из Wikimedia Commons"),
        code("""
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'
FILE_TITLE = 'File:Udmurt.ogg'
params = {
    'action': 'query',
    'titles': FILE_TITLE,
    'prop': 'imageinfo',
    'iiprop': 'url|mime|size|extmetadata',
    'format': 'json',
}
r = requests.get(COMMONS_API, params=params, timeout=30)
r.raise_for_status()
page = next(iter(r.json()['query']['pages'].values()))
info = page['imageinfo'][0]
audio_url = info['url']
print(audio_url)
print('license:', info.get('extmetadata', {}).get('LicenseShortName', {}).get('value'))

audio_path = DATA_DIR / 'udmurt.ogg'
audio_path.write_bytes(requests.get(audio_url, timeout=60).content)
Audio(str(audio_path))
"""),
        md("## 2. Нормализуем аудио под ASR"),
        code("""
y, sr = librosa.load(audio_path, sr=16000, mono=True)
trimmed = y[: 30 * 16000]
wav_path = DATA_DIR / 'udmurt_16k.wav'
sf.write(wav_path, trimmed, 16000)
print('seconds:', round(len(trimmed) / 16000, 2))
Audio(str(wav_path))
"""),
        md("## 3. Запускаем Whisper tiny"),
        code("""
asr = pipeline('automatic-speech-recognition', model='openai/whisper-tiny')
result = asr(str(wav_path), generate_kwargs={'task': 'transcribe'})
print(result['text'])
save_artifact('lesson04_asr_whisper_tiny.txt', result['text'])
"""),
        md("## 4. Анализируем результат как baseline, а не как истину"),
        code("""
transcript = result['text']
analysis = {
    'chars': len(transcript),
    'tokens': len(transcript.split()),
    'cyrillic_share': len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', transcript)) / max(len(transcript), 1),
    'latin_share': len(re.findall(r'[A-Za-z]', transcript)) / max(len(transcript), 1),
    'questions_for_human': [
        'на каком языке модель фактически распознала речь?',
        'есть ли совпадающие удмуртские слова?',
        'нужна ли ручная транскрипция хотя бы 30 секунд?',
        'какие шумы или особенности голоса мешают?',
    ],
}
print(json.dumps(analysis, ensure_ascii=False, indent=2))
save_artifact('lesson04_asr_analysis.json', json.dumps(analysis, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Похожа ли транскрипция на речь в аудио?
2. Модель ошиблась языком или просто дала шумный текст?
3. Какой минимальный набор ручной разметки нужен для честной оценки?
4. Что должно быть в протоколе записи для будущих полевых данных?
"""),
    ],
    "05_corpus_cleaning_datacard.ipynb": [
        md("""
# Занятие 5. Очистка, фильтрация и data card маленького корпуса

**Цель практики:** собрать мини-корпус через Wikipedia API, удалить дубли/короткие тексты, посчитать простые признаки качества и создать черновик data card.

Работает в бесплатном Colab на CPU.
"""),
        code(COMMON_SETUP),
        md("## 1. Скачиваем мини-корпус"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

def random_extracts(n=25):
    params = {
        'action': 'query',
        'generator': 'random',
        'grnnamespace': 0,
        'grnlimit': n,
        'prop': 'extracts|info',
        'explaintext': 1,
        'inprop': 'url',
        'format': 'json',
    }
    r = requests.get(API, params=params, timeout=30)
    r.raise_for_status()
    pages = r.json().get('query', {}).get('pages', {})
    return pd.DataFrame([{
        'title': p.get('title'),
        'url': p.get('fullurl'),
        'text': p.get('extract', ''),
    } for p in pages.values()])

df = random_extracts(25)
df.loc[len(df)] = df.iloc[0]  # искусственный дубль для практики
show_df(df[['title', 'url']], 10)
"""),
        md("## 2. Чистим текст и считаем признаки"),
        code("""
def normalize_space(text):
    return re.sub(r'\\s+', ' ', str(text)).strip()

clean = df.copy()
clean['text_clean'] = clean['text'].apply(normalize_space)
clean['chars'] = clean['text_clean'].str.len()
clean['tokens'] = clean['text_clean'].apply(lambda x: len(re.findall(r'\\w+', x)))
clean['cyrillic_share'] = clean['text_clean'].apply(lambda x: len(re.findall(r'[А-Яа-яЁёӐ-ӿ]', x)) / max(len(x), 1))
clean['duplicate'] = clean.duplicated('text_clean')
clean['keep'] = (clean['chars'] >= 300) & (clean['cyrillic_share'] >= 0.45) & (~clean['duplicate'])

show_df(clean[['title', 'chars', 'tokens', 'cyrillic_share', 'duplicate', 'keep']], 30)
filtered = clean[clean['keep']].copy()
save_artifact('lesson05_clean_corpus.csv', filtered[['title', 'url', 'text_clean', 'chars', 'tokens', 'cyrillic_share']])
"""),
        md("## 3. Делим на train/dev/test без обучения модели"),
        code("""
filtered = filtered.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(filtered)
filtered['split'] = 'train'
filtered.loc[filtered.index >= int(n * 0.8), 'split'] = 'dev'
filtered.loc[filtered.index >= int(n * 0.9), 'split'] = 'test'
show_df(filtered[['title', 'chars', 'split']], 30)
save_artifact('lesson05_splits.csv', filtered[['title', 'url', 'split', 'text_clean']])
"""),
        md("## 4. Черновик data card"),
        code("""
data_card = f'''# Data card: mini {LANG} Wikipedia corpus

## Source
Wikipedia API, language edition: {LANG}.wikipedia.org

## Size
- Raw documents: {len(df)}
- Kept documents: {len(filtered)}
- Total kept characters: {int(filtered['chars'].sum()) if len(filtered) else 0}

## Filtering
- Removed exact duplicates
- Removed documents shorter than 300 characters
- Required Cyrillic-script share >= 0.45

## Known limitations
- Wikipedia is not representative of all language varieties
- Articles may contain named entities, Russian borrowings, lists and formatting remnants
- Human language review is still required

## License
Check Wikipedia/Wikimedia terms and page histories before redistribution.
'''
print(data_card)
save_artifact('lesson05_DATA_CARD.md', data_card)
"""),
        md("""
## Вопросы для отчёта

1. Что фильтры удалили ошибочно?
2. Какие признаки качества не видны из автоматических метрик?
3. Что обязательно должен проверить носитель/эксперт?
4. Что нужно добавить в data card перед публикацией?
"""),
    ],
    "06_parallel_alignment_mt.ipynb": [
        md("""
# Занятие 6. Параллельные тексты, alignment и переводческий baseline

**Цель практики:** загрузить маленький татарско-русский параллельный корпус, проверить кандидаты выравнивания и попробовать простой multilingual embedding baseline.

Работает в бесплатном Colab на CPU. Если Hugging Face загрузка недоступна, тетрадка использует встроенный мини-набор для демонстрации.
"""),
        code("""
!pip -q install datasets sentence-transformers pandas scikit-learn
"""),
        code(COMMON_SETUP + "\nfrom sklearn.metrics.pairwise import cosine_similarity\n"),
        md("## 1. Загружаем небольшой фрагмент параллельного корпуса"),
        code("""
fallback = pd.DataFrame({
    'tat': [
        'Мин мәктәпкә барам.',
        'Бу китап бик кызык.',
        'Без бүген яңа проект башлыйбыз.',
        'Телне саклау өчен мәгълүмат кирәк.',
        'Укытучы сораулар бирә.',
    ],
    'rus': [
        'Я иду в школу.',
        'Эта книга очень интересная.',
        'Сегодня мы начинаем новый проект.',
        'Для сохранения языка нужны данные.',
        'Учитель задает вопросы.',
    ],
})

try:
    from datasets import load_dataset
    ds = load_dataset('AigizK/tatar-russian-parallel-corpora', split='train[:80]')
    pairs = ds.to_pandas()
    print(pairs.columns)
    # Пытаемся найти текстовые колонки автоматически.
    text_cols = [c for c in pairs.columns if pairs[c].dtype == 'object']
    pairs = pairs[text_cols[:2]].dropna().head(50)
    pairs.columns = ['tat', 'rus']
except Exception as e:
    print('HF loading failed, using fallback:', e)
    pairs = fallback

show_df(pairs, 10)
save_artifact('lesson06_parallel_pairs.csv', pairs)
"""),
        md("## 2. Создаем испорченный порядок и ищем alignment"),
        code("""
left = pairs['tat'].reset_index(drop=True)
right = pairs['rus'].sample(frac=1, random_state=7).reset_index(drop=True)
candidate_df = pd.DataFrame({'tat': left, 'rus_shuffled': right})
show_df(candidate_df, 10)
"""),
        md("## 3. Embedding baseline для поиска пар"),
        code("""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
emb_left = model.encode(left.tolist(), normalize_embeddings=True)
emb_right = model.encode(right.tolist(), normalize_embeddings=True)
sim = cosine_similarity(emb_left, emb_right)

matches = []
for i in range(len(left)):
    j = int(sim[i].argmax())
    matches.append({
        'tat': left.iloc[i],
        'best_rus': right.iloc[j],
        'score': float(sim[i, j]),
        'gold_rus': pairs['rus'].iloc[i],
        'correct_exact': right.iloc[j] == pairs['rus'].iloc[i],
    })
matches = pd.DataFrame(matches)
show_df(matches, 20)
print('exact alignment accuracy:', matches['correct_exact'].mean())
save_artifact('lesson06_alignment_candidates.csv', matches)
"""),
        md("## 4. Что делать с переводом"),
        code("""
low_conf = matches[matches['score'] < 0.5]
report = {
    'pairs': len(matches),
    'exact_alignment_accuracy': float(matches['correct_exact'].mean()),
    'low_confidence_pairs': len(low_conf),
    'next_steps': [
        'проверить top-20 кандидатов вручную',
        'сохранять score и источник каждой пары',
        'не обучать MT на непроверенных парах',
        'сделать dev/test из вручную подтвержденных пар',
    ],
}
print(json.dumps(report, ensure_ascii=False, indent=2))
save_artifact('lesson06_alignment_report.json', json.dumps(report, ensure_ascii=False, indent=2))
"""),
        md("""
## Вопросы для отчёта

1. Какие пары baseline нашел неверно?
2. Можно ли доверять score без ручной проверки?
3. Сколько проверенных пар нужно собрать до первого MT baseline?
4. Какие права и источники нужно сохранить рядом с парами?
"""),
    ],
    "07_spellchecker_normalization.ipynb": [
        md("""
# Занятие 7. Частотный словарь, нормализация и простой спелчекер

**Цель практики:** собрать маленький словарь из открытых текстов, искусственно внести ошибки и проверить простой baseline исправления через edit distance.

Работает в бесплатном Colab на CPU.
"""),
        code("""
!pip -q install rapidfuzz pandas
"""),
        code(COMMON_SETUP + "\nfrom rapidfuzz import process, fuzz\n"),
        md("## 1. Собираем слова из мини-корпуса"),
        code("""
LANG = 'udm'
API = f'https://{LANG}.wikipedia.org/w/api.php'

params = {
    'action': 'query',
    'generator': 'random',
    'grnnamespace': 0,
    'grnlimit': 20,
    'prop': 'extracts',
    'explaintext': 1,
    'format': 'json',
}
r = requests.get(API, params=params, timeout=30)
r.raise_for_status()
pages = r.json().get('query', {}).get('pages', {})
texts = [p.get('extract', '') for p in pages.values()]
tokens = []
for t in texts:
    tokens.extend(re.findall(r'[А-Яа-яЁёӐ-ӿ]{3,}', t.lower()))

freq = pd.Series(tokens).value_counts().reset_index()
freq.columns = ['word', 'count']
freq = freq[freq['count'] >= 1].head(300)
show_df(freq, 20)
save_artifact('lesson07_frequency_dictionary.csv', freq)
"""),
        md("## 2. Делаем искусственные ошибки"),
        code("""
alphabet = sorted(set(''.join(freq['word'].head(100).tolist())))

def corrupt(word):
    if len(word) < 4:
        return word
    i = random.randrange(len(word))
    op = random.choice(['delete', 'swap', 'replace'])
    if op == 'delete':
        return word[:i] + word[i+1:]
    if op == 'swap' and i < len(word) - 1:
        return word[:i] + word[i+1] + word[i] + word[i+2:]
    return word[:i] + random.choice(alphabet or ['а']) + word[i+1:]

test_words = freq['word'].head(40).tolist()
typos = pd.DataFrame({'gold': test_words})
typos['typo'] = typos['gold'].apply(corrupt)
show_df(typos, 20)
"""),
        md("## 3. Исправляем через ближайшее слово"),
        code("""
vocab = freq['word'].tolist()

def suggest(word, limit=3):
    return process.extract(word, vocab, scorer=fuzz.WRatio, limit=limit)

rows = []
for _, row in typos.iterrows():
    sugg = suggest(row['typo'])
    rows.append({
        'typo': row['typo'],
        'gold': row['gold'],
        'top1': sugg[0][0] if sugg else None,
        'top1_score': sugg[0][1] if sugg else None,
        'top3': [s[0] for s in sugg],
    })

eval_df = pd.DataFrame(rows)
eval_df['top1_correct'] = eval_df['top1'] == eval_df['gold']
eval_df['top3_correct'] = eval_df.apply(lambda r: r['gold'] in r['top3'], axis=1)
show_df(eval_df, 40)
print('top1 accuracy:', eval_df['top1_correct'].mean())
print('top3 accuracy:', eval_df['top3_correct'].mean())
save_artifact('lesson07_spellchecker_eval.csv', eval_df)
"""),
        md("## 4. Где нормализация отличается от исправления"),
        code("""
normalization_questions = pd.DataFrame([
    {'case': 'вариант орфографии', 'should_fix?': 'не всегда', 'human_check': 'является ли форма допустимой нормой?'},
    {'case': 'OCR-ошибка', 'should_fix?': 'часто да', 'human_check': 'есть ли изображение-источник?'},
    {'case': 'диалектная форма', 'should_fix?': 'нет без решения сообщества', 'human_check': 'какой вариант нужен в корпусе?'},
    {'case': 'заимствование/имя', 'should_fix?': 'осторожно', 'human_check': 'это слово языка или шум?'},
])
show_df(normalization_questions)
save_artifact('lesson07_normalization_questions.csv', normalization_questions)
"""),
        md("""
## Вопросы для отчёта

1. Какие ошибки baseline исправляет хорошо?
2. Какие “ошибки” могут оказаться нормальными вариантами?
3. Что нужно хранить в словаре: частоты, источники, варианты, комментарии?
4. Кто должен принимать решение о норме?
"""),
    ],
    "08_final_package_review.ipynb": [
        md("""
# Занятие 8. Финальная интеграция и агентная проверка пакета

**Цель практики:** собрать минимальный финальный пакет проекта в Colab, проверить его чек-листом и получить список задач перед публикацией/летней школой.

Работает в бесплатном Colab на CPU.
"""),
        code(COMMON_SETUP),
        md("## 1. Создаем мини-пакет проекта"),
        code("""
PROJECT = DATA_DIR / 'final_package'
PROJECT.mkdir(exist_ok=True)

files = {
    'README.md': '''# Mini low-resource language project

Goal: create a tiny documented corpus prototype.
Language: Udmurt / replace with your language.
Status: classroom prototype, not production.
''',
    'DATA_CARD.md': '''# Data card

Sources: Wikipedia API / Wikimedia Commons / replace with your sources.
License: check source pages before redistribution.
Known limitations: small sample, not representative, needs human review.
''',
    'EVAL_REPORT.md': '''# Evaluation report

Metric: manual inspection + simple script diagnostics.
Result: baseline works partially.
Errors: needs language expert review.
''',
    'sources.csv': 'title,url,type,license_or_access,notes\\nExample,https://example.org,web,unknown,replace me\\n',
}
for name, content in files.items():
    (PROJECT / name).write_text(content, encoding='utf-8')

print('created files:')
for p in sorted(PROJECT.iterdir()):
    print('-', p.name)
"""),
        md("## 2. Проверяем структуру пакета"),
        code("""
required = ['README.md', 'DATA_CARD.md', 'EVAL_REPORT.md', 'sources.csv']
checks = []
for name in required:
    p = PROJECT / name
    checks.append({
        'check': f'{name} exists',
        'ok': p.exists(),
        'details': str(p),
    })
    if p.exists():
        txt = p.read_text(encoding='utf-8')
        checks.append({
            'check': f'{name} is not empty',
            'ok': len(txt.strip()) > 40,
            'details': f'{len(txt)} chars',
        })

sources = pd.read_csv(PROJECT / 'sources.csv')
for col in ['title', 'url', 'type', 'license_or_access', 'notes']:
    checks.append({'check': f'sources.csv has column {col}', 'ok': col in sources.columns, 'details': ''})

check_df = pd.DataFrame(checks)
show_df(check_df, 30)
save_artifact('lesson08_package_checks.csv', check_df)
"""),
        md("## 3. Агентный review без LLM: правила и задачи"),
        code("""
tasks = []

if not check_df['ok'].all():
    tasks.append('исправить отсутствующие или пустые обязательные файлы')

if 'unknown' in sources['license_or_access'].fillna('').str.lower().to_string():
    tasks.append('уточнить лицензии в sources.csv')

readme = (PROJECT / 'README.md').read_text(encoding='utf-8').lower()
if 'replace' in readme:
    tasks.append('заменить placeholder-описания в README.md')

data_card = (PROJECT / 'DATA_CARD.md').read_text(encoding='utf-8').lower()
if 'human review' in data_card or 'needs' in data_card:
    tasks.append('запланировать ручную проверку носителем/экспертом')

report = {
    'package_ready': len(tasks) == 0,
    'tasks_before_publication': tasks,
    'summer_school_angle': [
        'какой артефакт можно показать участникам',
        'какие данные нужно дособрать',
        'какие роли нужны в команде',
        'какие риски нельзя автоматизировать',
    ],
}
print(json.dumps(report, ensure_ascii=False, indent=2))
save_artifact('lesson08_agent_review.json', json.dumps(report, ensure_ascii=False, indent=2))
"""),
        md("## 4. Финальная таблица для защиты"),
        code("""
def status(ok):
    return 'готово' if ok else 'нужно доработать'

def exists(name):
    return (PROJECT / name).exists()

def nonempty(name):
    p = PROJECT / name
    return p.exists() and len(p.read_text(encoding='utf-8').strip()) > 40

defense = pd.DataFrame([
    {'artifact': 'README', 'status': status(nonempty('README.md')), 'next_step': 'уточнить цель и пользователей'},
    {'artifact': 'sources.csv', 'status': status(exists('sources.csv')), 'next_step': 'добавить реальные лицензии'},
    {'artifact': 'DATA_CARD', 'status': status(nonempty('DATA_CARD.md')), 'next_step': 'описать ограничения'},
    {'artifact': 'EVAL_REPORT', 'status': status(nonempty('EVAL_REPORT.md')), 'next_step': 'добавить примеры ошибок'},
])
show_df(defense)
save_artifact('lesson08_defense_table.csv', defense)
"""),
        md("""
## Вопросы для отчёта

1. Что в пакете уже можно показать внешнему человеку?
2. Что нельзя публиковать без дополнительной проверки?
3. Какие задачи переходят в летнюю школу?
4. Какие проверки стоит автоматизировать агентом?
"""),
    ],
}


README = """# Colab notebooks

These notebooks are classroom practices for the course sessions. They are designed for free Google Colab on CPU or, where useful, the free GPU tier. Each notebook keeps data small, writes artifacts under `/content/lowres_lab`, and asks students to inspect the output rather than trust a model blindly.

| Notebook | Practice | Open in Colab |
|---|---|---|
| `01_dataset_inventory_pipeline.ipynb` | dataset inventory pipeline for languages of Russia using OPUS and Hugging Face dataset APIs | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/01_dataset_inventory_pipeline.ipynb) |
| `02_web_scraping_sources.ipynb` | source table, API collection, basic noise checks | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/02_web_scraping_sources.ipynb) |
| `03_ocr_udmurt_commons.ipynb` | OCR baseline on an Udmurt Wikimedia Commons scan | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/03_ocr_udmurt_commons.ipynb) |
| `04_asr_udmurt_whisper_tiny.ipynb` | ASR baseline on open Udmurt audio | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/04_asr_udmurt_whisper_tiny.ipynb) |
| `05_corpus_cleaning_datacard.ipynb` | mini-corpus cleaning, splits, data card | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/05_corpus_cleaning_datacard.ipynb) |
| `06_parallel_alignment_mt.ipynb` | Tatar-Russian alignment baseline | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/06_parallel_alignment_mt.ipynb) |
| `07_spellchecker_normalization.ipynb` | frequency dictionary and edit-distance spellchecker | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/07_spellchecker_normalization.ipynb) |
| `08_final_package_review.ipynb` | final package checklist and review | [Colab](https://colab.research.google.com/github/AnnaLebedeva/lowres-course/blob/main/colab_notebooks/08_final_package_review.ipynb) |

Main open sources used or referenced:

- Wikimedia Commons category `Udmurt Dunne` and file `Удномер.jpg`
- Wikimedia Commons file `Udmurt.ogg`
- Wikipedia API language editions: `udm`, `kv`, `sah`, `tt`, `mhr`
- Hugging Face dataset `AigizK/tatar-russian-parallel-corpora`
- Tatoeba downloads as an optional source for sentence pairs
- Mozilla Common Voice as an optional source for ASR data when access/terms are suitable
"""


def main():
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        (OUT / name).write_text(json.dumps(nb(cells), ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(README, encoding="utf-8")
    print(f"wrote {len(NOTEBOOKS)} notebooks to {OUT}")


if __name__ == "__main__":
    main()
