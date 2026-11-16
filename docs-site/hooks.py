"""Хуки MkDocs/ProperDocs для сайта-документации.

Хуки выполняются после плагинов из mkdocs.yml. Здесь же задаём настройки,
которым нужны Python-объекты, вместо YAML-тегов `!!python/...`.
"""

from mkdocs.structure.files import InclusionLevel
from pymdownx.slugs import slugify

NAV_FILE = "SUMMARY.md"


def on_config(config):
    # Читаемые якоря для русских заголовков (#идея-первая-...) вместо #_1, #_2.
    config.mdx_configs.setdefault("toc", {})["slugify"] = slugify(case="lower")
    return config


def on_files(files, config):
    # literate-nav уже прочитал меню из SUMMARY.md — отдельной страницей оно не нужно.
    nav_file = files.get_file_from_path(NAV_FILE)
    if nav_file is not None:
        nav_file.inclusion = InclusionLevel.EXCLUDED
    return files
