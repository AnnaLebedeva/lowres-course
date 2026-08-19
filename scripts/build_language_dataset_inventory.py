import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/Users/ann.lebedeva/Documents/lowres")
OUT = ROOT / "generated_datasets"
OUT.mkdir(exist_ok=True)

OPUS_API = "https://opus.nlpl.eu/opusapi"
HF_DATASETS_API = "https://huggingface.co/api/datasets"


LANGUAGES = [
    # Turkic
    {"language_ru": "татарский", "language_en": "Tatar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "tat", "opus_code": "tt", "wiki_code": "tt"},
    {"language_ru": "башкирский", "language_en": "Bashkir", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "bak", "opus_code": "ba", "wiki_code": "ba"},
    {"language_ru": "чувашский", "language_en": "Chuvash", "family": "Тюркская", "branch": "огурская", "iso639_3": "chv", "opus_code": "chv", "wiki_code": "cv"},
    {"language_ru": "якутский / саха", "language_en": "Sakha / Yakut", "family": "Тюркская", "branch": "сибирская", "iso639_3": "sah", "opus_code": "sah", "wiki_code": "sah"},
    {"language_ru": "тувинский", "language_en": "Tuvan", "family": "Тюркская", "branch": "сибирская", "iso639_3": "tyv", "opus_code": "tyv", "wiki_code": "tyv"},
    {"language_ru": "хакасский", "language_en": "Khakas", "family": "Тюркская", "branch": "сибирская", "iso639_3": "kjh", "opus_code": "kjh", "wiki_code": None},
    {"language_ru": "алтайский", "language_en": "Altai", "family": "Тюркская", "branch": "сибирская", "iso639_3": "alt", "opus_code": "alt", "wiki_code": "alt"},
    {"language_ru": "кумыкский", "language_en": "Kumyk", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "kum", "opus_code": "kum", "wiki_code": None},
    {"language_ru": "карачаево-балкарский", "language_en": "Karachay-Balkar", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "krc", "opus_code": "krc", "wiki_code": "krc"},
    {"language_ru": "ногайский", "language_en": "Nogai", "family": "Тюркская", "branch": "кыпчакская", "iso639_3": "nog", "opus_code": "nog", "wiki_code": None},
    {"language_ru": "крымскотатарский", "language_en": "Crimean Tatar", "family": "Тюркская", "branch": "кыпчакско-огузская", "iso639_3": "crh", "opus_code": "crh", "wiki_code": "crh"},
    # Uralic
    {"language_ru": "удмуртский", "language_en": "Udmurt", "family": "Уральская", "branch": "пермская", "iso639_3": "udm", "opus_code": "udm", "wiki_code": "udm"},
    {"language_ru": "коми-зырянский", "language_en": "Komi-Zyrian", "family": "Уральская", "branch": "пермская", "iso639_3": "kpv", "opus_code": "kpv", "wiki_code": "kv"},
    {"language_ru": "коми-пермяцкий", "language_en": "Komi-Permyak", "family": "Уральская", "branch": "пермская", "iso639_3": "koi", "opus_code": "koi", "wiki_code": "koi"},
    {"language_ru": "эрзянский", "language_en": "Erzya", "family": "Уральская", "branch": "мордовская", "iso639_3": "myv", "opus_code": "myv", "wiki_code": "myv"},
    {"language_ru": "мокшанский", "language_en": "Moksha", "family": "Уральская", "branch": "мордовская", "iso639_3": "mdf", "opus_code": "mdf", "wiki_code": "mdf"},
    {"language_ru": "марийский луговой", "language_en": "Meadow Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mhr", "opus_code": "mhr", "wiki_code": "mhr"},
    {"language_ru": "марийский горный", "language_en": "Hill Mari", "family": "Уральская", "branch": "марийская", "iso639_3": "mrj", "opus_code": "mrj", "wiki_code": "mrj"},
    {"language_ru": "карельский", "language_en": "Karelian", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "krl", "opus_code": "krl", "wiki_code": "krl"},
    {"language_ru": "вепсский", "language_en": "Veps", "family": "Уральская", "branch": "прибалтийско-финская", "iso639_3": "vep", "opus_code": "vep", "wiki_code": "vep"},
    {"language_ru": "хантыйский", "language_en": "Khanty", "family": "Уральская", "branch": "угорская", "iso639_3": "kca", "opus_code": None, "wiki_code": None},
    {"language_ru": "мансийский", "language_en": "Mansi", "family": "Уральская", "branch": "угорская", "iso639_3": "mns", "opus_code": "mns", "wiki_code": None},
    {"language_ru": "ненецкий", "language_en": "Nenets", "family": "Уральская", "branch": "самодийская", "iso639_3": "yrk", "opus_code": "yrk", "wiki_code": None},
    # North Caucasian
    {"language_ru": "чеченский", "language_en": "Chechen", "family": "Северокавказская", "branch": "нахская", "iso639_3": "che", "opus_code": "ce", "wiki_code": "ce"},
    {"language_ru": "ингушский", "language_en": "Ingush", "family": "Северокавказская", "branch": "нахская", "iso639_3": "inh", "opus_code": "inh", "wiki_code": "inh"},
    {"language_ru": "аварский", "language_en": "Avar", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "ava", "opus_code": "av", "wiki_code": "av"},
    {"language_ru": "даргинский", "language_en": "Dargwa", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "dar", "opus_code": "dar", "wiki_code": None},
    {"language_ru": "лезгинский", "language_en": "Lezgian", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lez", "opus_code": "lez", "wiki_code": "lez"},
    {"language_ru": "лакский", "language_en": "Lak", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "lbe", "opus_code": "lbe", "wiki_code": "lbe"},
    {"language_ru": "рутульский", "language_en": "Rutul", "family": "Северокавказская", "branch": "нахско-дагестанская", "iso639_3": "rut", "opus_code": "rut", "wiki_code": None},
    {"language_ru": "адыгейский", "language_en": "Adyghe", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "ady", "opus_code": "ady", "wiki_code": "ady"},
    {"language_ru": "кабардино-черкесский", "language_en": "Kabardian", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "kbd", "opus_code": "kbd", "wiki_code": "kbd"},
    {"language_ru": "абазинский", "language_en": "Abaza", "family": "Северокавказская", "branch": "абхазо-адыгская", "iso639_3": "abq", "opus_code": None, "wiki_code": None},
    # Mongolic and Tungusic
    {"language_ru": "бурятский", "language_en": "Buryat", "family": "Монгольская", "branch": "монгольская", "iso639_3": "bxr", "opus_code": "bxr", "wiki_code": "bxr"},
    {"language_ru": "калмыцкий", "language_en": "Kalmyk", "family": "Монгольская", "branch": "ойратская", "iso639_3": "xal", "opus_code": "xal", "wiki_code": "xal"},
    {"language_ru": "эвенкийский", "language_en": "Evenki", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "evn", "opus_code": "evn", "wiki_code": None},
    {"language_ru": "нанайский", "language_en": "Nanai", "family": "Тунгусо-маньчжурская", "branch": "тунгусская", "iso639_3": "gld", "opus_code": "gld", "wiki_code": None},
    # Other families
    {"language_ru": "нивхский", "language_en": "Nivkh", "family": "изолят / палеоазиатская группа", "branch": "нивхская", "iso639_3": "niv", "opus_code": None, "wiki_code": None},
    {"language_ru": "чукотский", "language_en": "Chukchi", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "ckt", "opus_code": None, "wiki_code": None},
    {"language_ru": "корякский", "language_en": "Koryak", "family": "чукотско-камчатская", "branch": "чукотская", "iso639_3": "kpy", "opus_code": None, "wiki_code": None},
    {"language_ru": "алеутский", "language_en": "Aleut", "family": "эскимосско-алеутская", "branch": "алеутская", "iso639_3": "ale", "opus_code": "ale", "wiki_code": None},
    {"language_ru": "эскимосский / юпик", "language_en": "Yupik", "family": "эскимосско-алеутская", "branch": "эскимосская", "iso639_3": "ess", "opus_code": None, "wiki_code": None},
]


def api_json(url, params, attempts=3, timeout=30):
    """Загружает JSON из публичного API с короткими повторами при сбоях."""
    query = urllib.parse.urlencode(params)
    full_url = f"{url}?{query}"
    last_error = None
    for attempt in range(attempts):
        req = urllib.request.Request(full_url, headers={"User-Agent": "lowres-course-dataset-scout/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            time.sleep(1 + attempt * 2)
    raise last_error


def opus_summary(opus_code):
    """Собирает сводку OPUS по моноязычным и русско-параллельным данным языка."""
    empty = {
        "opus_ru_parallel_pairs": 0,
        "opus_ru_parallel_documents": 0,
        "opus_ru_parallel_corpora": "",
        "opus_mono_pairs_or_segments": 0,
        "opus_mono_documents": 0,
        "opus_mono_corpora": "",
    }
    if not opus_code:
        return {"opus_checked": False, **empty}

    try:
        data = api_json(OPUS_API, {
            "source": "ru",
            "target": opus_code,
            "preprocessing": "xml",
            "version": "latest",
        }, attempts=1, timeout=8)
    except Exception as exc:
        return {"opus_checked": False, "opus_error": str(exc), **empty}
    corpora = data.get("corpora", [])
    parallel = [
        c for c in corpora
        if {c.get("source"), c.get("target")} == {"ru", opus_code}
    ]
    mono = [
        c for c in corpora
        if c.get("source") == opus_code and not c.get("target")
    ]

    def as_int(value):
        """Преобразует числовые поля OPUS в int, считая пустые значения нулем."""
        if value in ("", None):
            return 0
        return int(value)

    return {
        "opus_checked": True,
        "opus_ru_parallel_pairs": sum(as_int(c.get("alignment_pairs")) for c in parallel),
        "opus_ru_parallel_documents": sum(as_int(c.get("documents")) for c in parallel),
        "opus_ru_parallel_corpora": "; ".join(
            f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in parallel
        ),
        "opus_mono_pairs_or_segments": sum(as_int(c.get("alignment_pairs")) for c in mono),
        "opus_mono_documents": sum(as_int(c.get("documents")) for c in mono),
        "opus_mono_corpora": "; ".join(
            f"{c.get('corpus')} ({c.get('alignment_pairs') or 0})" for c in mono
        ),
    }


def wiki_summary(wiki_code):
    """Читает статистику языкового раздела Википедии, если такой раздел существует."""
    if not wiki_code:
        return {"wiki_checked": False, "wiki_articles": "", "wiki_pages": "", "wiki_source_url": ""}
    try:
        data = api_json(f"https://{wiki_code}.wikipedia.org/w/api.php", {
            "action": "query",
            "meta": "siteinfo",
            "siprop": "statistics",
            "format": "json",
        })
        stats = data.get("query", {}).get("statistics", {})
        return {
            "wiki_checked": True,
            "wiki_articles": stats.get("articles", ""),
            "wiki_pages": stats.get("pages", ""),
            "wiki_source_url": f"https://{wiki_code}.wikipedia.org/",
        }
    except Exception as exc:
        return {
            "wiki_checked": False,
            "wiki_articles": "",
            "wiki_pages": "",
            "wiki_source_url": f"https://{wiki_code}.wikipedia.org/",
            "wiki_error": str(exc),
        }


def hf_dataset_summary(item):
    """Ищет датасеты на Hugging Face и собирает сводку вероятных языковых ресурсов.

    Поиск Hugging Face не всегда дает надежное количество документов или предложений.
    Поэтому сохраняем разведочные метаданные: число кандидатов, вероятные русско-
    параллельные датасеты, топ id датасетов, скачивания и категории размера из тегов.
    """
    language_tags = {
        f"language:{item.get('opus_code')}" if item.get("opus_code") else "",
        f"language:{item.get('iso639_3')}" if item.get("iso639_3") else "",
    }
    language_tags.discard("")
    search_terms = [item.get("language_en"), item.get("language_ru")]
    seen = {}
    attempted = 0
    successful = 0
    for tag in language_tags:
        attempted += 1
        try:
            results = api_json(HF_DATASETS_API, {"filter": tag, "limit": 10}, attempts=2, timeout=20)
        except Exception:
            continue
        if not isinstance(results, list):
            continue
        successful += 1
        for dataset in results:
            dataset_id = dataset.get("id")
            if dataset_id:
                seen[dataset_id] = dataset

    for term in [x for x in search_terms if x]:
        attempted += 1
        try:
            results = api_json(HF_DATASETS_API, {"search": term, "limit": 10}, attempts=2, timeout=20)
        except Exception:
            continue
        if not isinstance(results, list):
            continue
        successful += 1
        for dataset in results:
            dataset_id = dataset.get("id")
            if dataset_id:
                seen[dataset_id] = dataset

    lang_texts = [
        str(item.get("language_en", "")).lower(),
        str(item.get("language_ru", "")).lower(),
    ]

    def mentions_language_name(text, names):
        """Проверяет, встречается ли полное название языка как отдельная фраза."""
        for name in names:
            if not name:
                continue
            for part in re.split(r"\s*/\s*|\s+-\s+", name):
                part = part.strip()
                if len(part) >= 4 and re.search(rf"(?<![\w-]){re.escape(part)}(?![\w-])", text):
                    return True
        return False

    datasets = []
    for dataset in seen.values():
        tags = set(dataset.get("tags") or [])
        haystack = " ".join([
            dataset.get("id", ""),
            dataset.get("description", "") or "",
            " ".join(tags),
        ]).lower()
        tagged = bool(tags & language_tags)
        mentioned = mentions_language_name(haystack, lang_texts)
        if tagged or mentioned:
            datasets.append(dataset)

    def is_ru_parallel(dataset):
        """Эвристически определяет, похож ли HF-датасет на русско-параллельный."""
        tags = set(dataset.get("tags") or [])
        text = " ".join([
            dataset.get("id", ""),
            dataset.get("description", "") or "",
            " ".join(tags),
        ]).lower()
        return (
            "language:ru" in tags
            or "russian" in text
            or "рус" in text
            or "-rus-" in text
            or "rus-" in text
        )

    def specificity_score(dataset):
        """Ставит языково-специфичные датасеты выше широких многоязычных коллекций."""
        tags = set(dataset.get("tags") or [])
        text = " ".join([
            dataset.get("id", ""),
            dataset.get("description", "") or "",
        ]).lower()
        language_tag_count = sum(1 for tag in tags if tag.startswith("language:"))
        if mentions_language_name(text, lang_texts):
            return 2
        if language_tag_count <= 5:
            return 1
        return 0

    top = sorted(
        datasets,
        key=lambda d: (specificity_score(d), d.get("downloads") or 0),
        reverse=True,
    )[:5]
    size_categories = sorted({
        tag.replace("size_categories:", "")
        for dataset in datasets
        for tag in (dataset.get("tags") or [])
        if tag.startswith("size_categories:")
    })
    return {
        "hf_checked": successful > 0,
        "hf_query_attempts": attempted,
        "hf_query_successes": successful,
        "hf_dataset_count": len(datasets),
        "hf_ru_parallel_candidates": sum(1 for dataset in datasets if is_ru_parallel(dataset)),
        "hf_top_datasets": "; ".join(dataset.get("id", "") for dataset in top),
        "hf_downloads_sum": sum(int(dataset.get("downloads") or 0) for dataset in datasets),
        "hf_size_categories": "; ".join(size_categories),
        "hf_source_url": "https://huggingface.co/datasets",
    }


def build_inventory():
    """Собирает полную таблицу инвентаризации для заданного списка языков."""
    rows = []
    checked_at = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    previous_rows = {}
    previous_csv = OUT / "russia_languages_dataset_inventory.csv"
    if previous_csv.exists():
        previous = pd.read_csv(previous_csv).fillna("")
        previous_rows = {row["iso639_3"]: row for _, row in previous.iterrows()}

    for item in LANGUAGES:
        print("checking", item["language_ru"], flush=True)
        row = dict(item)
        opus = opus_summary(item.get("opus_code"))
        if opus.get("opus_error") and item["iso639_3"] in previous_rows:
            previous = previous_rows[item["iso639_3"]]
            for column in [
                "opus_ru_parallel_pairs",
                "opus_ru_parallel_documents",
                "opus_ru_parallel_corpora",
                "opus_mono_pairs_or_segments",
                "opus_mono_documents",
                "opus_mono_corpora",
            ]:
                opus[column] = previous.get(column, opus[column])
            opus["opus_cached_from_previous_run"] = True
        else:
            opus["opus_cached_from_previous_run"] = False
        row.update(opus)
        row.update(wiki_summary(item.get("wiki_code")))
        row.update(hf_dataset_summary(item))
        row["parallel_with_russian_source"] = "https://opus.nlpl.eu/opusapi"
        row["monolingual_source"] = "OPUS monolingual rows; Wikipedia statistics; Hugging Face dataset search"
        row["checked_at_utc"] = checked_at
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Записывает инвентаризацию в CSV и XLSX-артефакты."""
    df = build_inventory()
    df = df.sort_values(["family", "branch", "language_ru"]).reset_index(drop=True)
    csv_path = OUT / "russia_languages_dataset_inventory.csv"
    xlsx_path = OUT / "russia_languages_dataset_inventory.xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="inventory", index=False)
        meta = pd.DataFrame([
            {"field": "scope", "value": "Initial course inventory: major living languages of peoples of Russia, no dialect-level coverage."},
            {"field": "sources", "value": "OPUS API; Wikipedia siteinfo statistics; Hugging Face dataset API."},
            {"field": "checked_at_utc", "value": df["checked_at_utc"].iloc[0] if len(df) else ""},
            {"field": "caveat", "value": "Counts are API snapshots and need linguistic/community review before publication."},
        ])
        meta.to_excel(writer, sheet_name="metadata", index=False)
    print(csv_path)
    print(xlsx_path)


if __name__ == "__main__":
    main()
