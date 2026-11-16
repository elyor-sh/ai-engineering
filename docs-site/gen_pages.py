"""Собирает сайт-документацию из COURSE_PLAN.md и всего, что лежит в lessons/.

Скрипт запускает плагин mkdocs-gen-files при каждой сборке. В режиме
`mkdocs serve` сборка повторяется при любом изменении в lessons/ или
COURSE_PLAN.md, поэтому новые уроки, скрипты и графики сами появляются
в боковом меню — регистрировать их руками не нужно.

* .md            -> страница (GitHub-разметка слегка подправляется под Python-Markdown);
* .py и прочий текст -> страница с подсвеченным кодом;
* картинки       -> файл + страница с картинкой;
* SUMMARY.md     -> боковое меню (его читает плагин mkdocs-literate-nav).
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parent.parent
LESSONS_DIR = ROOT / "lessons"
COURSE_PLAN = ROOT / "COURSE_PLAN.md"

SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".so", ".dylib", ".npy", ".npz", ".pkl", ".bin", ".gguf", ".safetensors"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
CODE_LANGS = {
    ".py": "python",
    ".sh": "bash",
    ".toml": "toml",
    ".json": "json",
    ".jsonl": "json",
    ".ipynb": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
    ".html": "html",
    ".js": "javascript",
    ".ts": "typescript",
    ".ini": "ini",
    ".cfg": "ini",
}
MAX_TEXT_BYTES = 1_000_000

ICONS = {
    "plan": "material/map-outline",
    "lesson": "material/book-open-page-variant-outline",
    "practice": "material/flask-outline",
    "markdown": "material/file-document-outline",
    "python": "material/language-python",
    "code": "material/code-braces",
    "image": "material/image-outline",
}


@dataclass
class Page:
    source: Path  # исходный файл
    doc_path: str  # путь страницы внутри сайта (posix, .md)
    label: str  # подпись в меню
    kind: str  # plan | lesson | practice | markdown | code | image


# ---------------------------------------------------------------------------
# Совместимость GitHub-разметки с Python-Markdown
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^( *)(`{3,}|~{3,})")
LIST_ITEM_RE = re.compile(r"^( *)(?:[-*+]|\d{1,9}[.)])(?: +|$)")
# Как в GitHub: прервать абзац может маркированный список или нумерованный с «1.».
INTERRUPTING_ITEM_RE = re.compile(r"^(?:[-*+]|1[.)]) +\S")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _closes_fence(line: str, fence: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= len(fence) and set(stripped) == {fence[0]}


def _list_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Диапазоны [start, end) строк, образующих списки верхнего уровня."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    fence: str | None = None
    for i, line in enumerate(lines):
        if fence:
            if _closes_fence(line, fence):
                fence = None
            continue
        if m := FENCE_RE.match(line):
            fence = m.group(2)
            if not m.group(1) and start is not None:
                blocks.append((start, i))
                start = None
            continue
        if not line.strip() or line.startswith(" "):
            continue
        if LIST_ITEM_RE.match(line):
            if start is None:
                start = i
        elif start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, len(lines)))
    return blocks


def _reindent_list_block(lines: list[str]) -> list[str]:
    """GitHub вкладывает списки с отступом 2-3 пробела, Python-Markdown требует 4."""
    item_indents = [_indent(line) for line in lines if (m := LIST_ITEM_RE.match(line)) and m.group(1)]
    unit = min(item_indents, default=4)
    if unit >= 4:
        return lines

    out: list[str] = []
    fence: str | None = None
    shift = 0
    for line in lines:
        if fence:
            out.append(" " * shift + line if line.strip() else line)
            if _closes_fence(line, fence):
                fence = None
            continue
        old = _indent(line)
        new = (old // unit) * 4 if line.strip() else 0
        if m := FENCE_RE.match(line):
            fence, shift = m.group(2), max(new - old, 0)
        out.append(" " * new + line[old:] if line.strip() else line)
    return out


def github_to_python_markdown(text: str) -> str:
    lines = text.split("\n")
    for start, end in reversed(_list_blocks(lines)):
        lines[start:end] = _reindent_list_block(lines[start:end])

    out: list[str] = []
    fence: str | None = None
    prev = ""
    for line in lines:
        if fence:
            out.append(line)
            if _closes_fence(line, fence):
                fence = None
            prev = line
            continue
        fence_match = FENCE_RE.match(line)
        prev_is_text = bool(prev.strip()) and not prev.startswith((" ", "\t", ">", "|")) and not LIST_ITEM_RE.match(prev)
        # Python-Markdown не даёт спискам, таблицам и блокам кода прерывать абзац —
        # добавляем пустую строку, как будто её поставил автор.
        needs_gap = (
            (prev_is_text and INTERRUPTING_ITEM_RE.match(line))
            or (prev.strip() and line.startswith("|") and not prev.startswith("|"))
            or (prev.strip() and fence_match and not fence_match.group(1))
        )
        if needs_gap:
            out.append("")
        if fence_match:
            fence = fence_match.group(2)
        out.append(line)
        prev = line
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Сбор файлов
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _h1(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("# "):
            return line[2:].strip().replace("`", "").replace("**", "")
    return None


def _is_text(path: Path) -> bool:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return False
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def _files(directory: Path) -> list[Path]:
    found = []
    for path in sorted(directory.rglob("*")):
        parts = path.relative_to(directory).parts
        if any(p in SKIP_DIRS or p.startswith(".") for p in parts):
            continue
        if path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES:
            found.append(path)
    return found


def _page_for(path: Path, base: Path, single_lesson_md: bool) -> Page | None:
    rel = _rel(path)
    inner = path.relative_to(base).as_posix()
    short = inner.removeprefix("practice/")
    suffix = path.suffix.lower()

    if suffix == ".md":
        top_level = path.parent == base
        if top_level and base != LESSONS_DIR:
            label = "Урок" if single_lesson_md else (_h1(path) or path.stem)
            return Page(path, rel, label, "lesson")
        if path.name.lower() == "readme.md" and path.parent.name == "practice":
            return Page(path, rel, "Практика", "practice")
        return Page(path, rel, _h1(path) or short, "markdown")
    if suffix in IMAGE_SUFFIXES:
        return Page(path, f"{posixpath.splitext(rel)[0]}-{suffix[1:]}.md", short, "image")
    if _is_text(path):
        return Page(path, f"{rel}.md", short, "code")
    return None


def _sort_key(page: Page) -> tuple:
    order = {"lesson": 0, "practice": 1, "markdown": 2, "code": 3, "image": 4}[page.kind]
    rel = page.source.relative_to(ROOT)
    return order, len(rel.parts), rel.as_posix()


def collect(base: Path) -> list[Page]:
    files = _files(base) if base != LESSONS_DIR else [p for p in sorted(base.iterdir()) if p.is_file()]
    top_md = [p for p in files if p.suffix.lower() == ".md" and p.parent == base]
    pages = [page for p in files if (page := _page_for(p, base, len(top_md) == 1))]
    return sorted(pages, key=_sort_key)


# ---------------------------------------------------------------------------
# Генерация страниц
# ---------------------------------------------------------------------------

LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)\s]+)((?:\s+\"[^\"]*\")?)\)")
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _front_matter(kind: str, source: Path, hide_toc: bool = False) -> str:
    icon = ICONS["python"] if kind == "code" and source.suffix == ".py" else ICONS[kind]
    lines = ["---", f"icon: {icon}"]
    if hide_toc:
        lines += ["hide:", "  - toc"]
    return "\n".join([*lines, "---", "", ""])


def _rewrite_links(text: str, page: Page, by_source: dict[Path, Page]) -> str:
    """Ссылки на .py, папки и COURSE_PLAN.md ведут на сгенерированные страницы."""

    def replace(m: re.Match[str]) -> str:
        bang, label, target, title = m.groups()
        if bang or target.startswith(("#", "/")) or SCHEME_RE.match(target):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        resolved = (page.source.parent / unquote(path)).resolve()
        if resolved.is_dir():
            resolved = next(
                (resolved / n for n in ("README.md", "index.md") if (resolved / n).exists()),
                resolved,
            )
        linked = by_source.get(resolved)
        if linked is None or linked.kind == "image":
            return m.group(0)
        new = posixpath.relpath(linked.doc_path, posixpath.dirname(page.doc_path) or ".")
        return f"[{label}]({new}{sep}{anchor}{title})"

    out, fence = [], None
    for line in text.split("\n"):
        if fence:
            fence = None if _closes_fence(line, fence) else fence
        elif m := FENCE_RE.match(line):
            fence = m.group(2)
        else:
            line = LINK_RE.sub(replace, line)
        out.append(line)
    return "\n".join(out)


def render_markdown(page: Page, by_source: dict[Path, Page]) -> str:
    text = page.source.read_text(encoding="utf-8", errors="replace")
    text = _rewrite_links(github_to_python_markdown(text), page, by_source)
    if text.startswith("---"):  # у файла уже есть свой front matter
        return text
    return _front_matter(page.kind, page.source) + text


def render_code(page: Page) -> str:
    code = page.source.read_text(encoding="utf-8")
    rel = _rel(page.source)
    lang = CODE_LANGS.get(page.source.suffix.lower(), "text")
    longest = max((len(m) for m in re.findall(r"`{3,}", code)), default=0)
    fence = "`" * max(3, longest + 1)
    parts = [_front_matter("code", page.source, hide_toc=True), f"# {page.source.name}\n", f"Файл: `{rel}`\n"]
    if page.source.suffix == ".py":
        cmd = f"uv run python -m pytest {rel} -q" if page.source.name.startswith("test_") else f"uv run python {rel}"
        parts.append(f"Запуск: `{cmd}`\n")
    if code.strip():
        parts.append(f'{fence}{lang} linenums="1"\n{code.rstrip()}\n{fence}\n')
    else:
        parts.append("*Файл пустой.*\n")
    return "\n".join(parts)


def render_image(page: Page) -> str:
    rel = _rel(page.source)
    image = posixpath.relpath(rel, posixpath.dirname(page.doc_path))
    return _front_matter("image", page.source, hide_toc=True) + f"# {page.source.name}\n\nФайл: `{rel}`\n\n![{page.source.name}]({image})\n"


# ---------------------------------------------------------------------------
# Меню
# ---------------------------------------------------------------------------

MODULE_RE = re.compile(r"^## Модуль (\d+)\.\s*(.+)$", re.MULTILINE)
LESSON_DIR_RE = re.compile(r"^(\d+)-(\d+)-")
LESSON_H1_RE = re.compile(r"^Урок\s+(\d+\.\d+)\.?\s*(.+)$")


def _module_titles() -> dict[int, str]:
    if not COURSE_PLAN.exists():
        return {}
    text = COURSE_PLAN.read_text(encoding="utf-8")
    return {int(n): re.sub(r"\s*\(.*\)\s*$", "", t).strip() for n, t in MODULE_RE.findall(text)}


def _lesson_label(lesson_dir: Path, pages: list[Page]) -> str:
    lesson = next((p for p in pages if p.kind == "lesson"), None)
    h1 = _h1(lesson.source) if lesson else None
    if h1 and (m := LESSON_H1_RE.match(h1)):
        return f"{m.group(1)} {m.group(2).split(':')[0].strip()}"
    return h1 or lesson_dir.name


def _nav_item(depth: int, label: str, path: str | None = None) -> str:
    label = label.replace("[", "\\[").replace("]", "\\]")
    return "    " * depth + (f"* [{label}]({path})" if path else f"* {label}")


def main() -> None:
    pages: list[Page] = []
    nav: list[str] = []

    if COURSE_PLAN.exists():
        plan = Page(COURSE_PLAN, "index.md", "План курса", "plan")
        pages.append(plan)
        nav.append(_nav_item(0, plan.label, plan.doc_path))

    if LESSONS_DIR.is_dir():
        loose = collect(LESSONS_DIR)
        pages += loose
        nav += [_nav_item(0, p.label, p.doc_path) for p in loose]

        modules = _module_titles()
        current_module: str | None = None
        for lesson_dir in sorted(LESSONS_DIR.iterdir()):
            if not lesson_dir.is_dir() or lesson_dir.name in SKIP_DIRS or lesson_dir.name.startswith("."):
                continue
            lesson_pages = collect(lesson_dir)
            if not lesson_pages:
                continue
            pages += lesson_pages
            if m := LESSON_DIR_RE.match(lesson_dir.name):
                n = int(m.group(1))
                module = f"Модуль {n}. {modules[n]}" if n in modules else f"Модуль {n}"
            else:
                module = "Другое"
            if module != current_module:
                nav.append(_nav_item(0, module))
                current_module = module
            nav.append(_nav_item(1, _lesson_label(lesson_dir, lesson_pages)))
            nav += [_nav_item(2, p.label, p.doc_path) for p in lesson_pages]

    by_source = {p.source.resolve(): p for p in pages}
    for page in pages:
        if page.kind == "code":
            content = render_code(page)
        elif page.kind == "image":
            content = render_image(page)
            with mkdocs_gen_files.open(_rel(page.source), "wb") as f:
                f.write(page.source.read_bytes())
        else:
            content = render_markdown(page, by_source)
        with mkdocs_gen_files.open(page.doc_path, "w") as f:
            f.write(content)

    with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
        f.write("\n".join(nav) + "\n")


main()
