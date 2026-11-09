# Практика к уроку 0.2 — собираем окружение на Mac

> Твой Mac: **M4 Pro, 24 ГБ**, macOS 15. Уже есть: Homebrew, git, Docker, Ollama (+ старая `llama3`).
> Не хватает: `uv`, Jupyter, API-ключей.

Все команды выполняются из корня проекта:

```bash
cd ~/Desktop/projects/ai-engineering
```

Я уже положил в корень `.gitignore` и `.env.example`, их создавать не нужно.

---

## Шаг 1. Установить uv

```bash
brew install uv
uv --version
```

> Если терминал пишет `command not found: uv`, открой новое окно терминала.

---

## Шаг 2. Превратить папку в Python-проект

```bash
uv init --bare --python 3.12     # создаёт pyproject.toml (и больше ничего лишнего)
uv python pin 3.12               # создаёт .python-version
```

Почему 3.12, а не твой системный 3.13? 3.12 — самая «беспроблемная» версия для ML-библиотек, которые появятся дальше (torch, docling и т.п.). uv сам скачает нужный Python, системный не трогается.

---

## Шаг 3. Установить библиотеки

```bash
# основные
uv add openai anthropic python-dotenv rich

# только для разработки (Jupyter)
uv add --dev ipykernel jupyterlab
```

Что ставим и зачем:

| Пакет | Зачем |
|---|---|
| `openai` | SDK для OpenAI **и всех OpenAI-совместимых API** (OpenRouter, Gemini, Ollama) |
| `anthropic` | SDK для Claude (свой формат API) |
| `python-dotenv` | читает ключи из `.env` |
| `rich` | красивый вывод в терминале (таблицы, цвета, markdown) |
| `ipykernel`, `jupyterlab` | Jupyter-блокноты и запуск ячеек в редакторе |

🔍 **Мини-эксперимент:** открой `pyproject.toml` и `uv.lock`. Сравни их длину:

```bash
wc -l pyproject.toml uv.lock
```

Получится «список покупок» на ~15 строк и «чек» на сотни строк: это транзитивные зависимости, которые подтянули наши 4 пакета.

---

## Шаг 4. Git

```bash
git init
git status
```

✅ Проверь: в списке `git status` **не должно быть** `.env` и `.venv/`. Должны быть `.gitignore`, `.env.example`, `pyproject.toml`, `uv.lock`, `.python-version`, `COURSE_PLAN.md`, `lessons/`, `notebooklm/`.

---

## Шаг 5. API-ключи

Для курса **хватит одного облачного провайдера**. Рекомендация:

### 🥇 OpenRouter (основной): один ключ, сотни моделей

1. Зайди на <https://openrouter.ai>, зарегистрируйся.
2. **Credits** → пополни на **$5–10**. Этого надолго хватит: учебные запросы стоят доли цента.
3. **Keys** → **Create Key**. Сразу задай **credit limit** на ключ (например $5). Это страховка от бага с бесконечным циклом.
4. Скопируй ключ (`sk-or-...`). Он показывается **один раз**.

### 🥈 Google Gemini (бесплатный запасной вариант)

1. <https://aistudio.google.com> → **Get API key** → Create.
2. На бесплатном тарифе есть лимиты по числу запросов, а запросы могут использоваться для улучшения моделей Google. **Рабочие и личные данные туда не отправляем.**

### OpenAI / Anthropic (по желанию)

Не обязательно: их модели доступны и через OpenRouter. Свой ключ Anthropic пригодится, когда будем изучать её родной API (prompt caching и т.п.).

### Записываем ключи

```bash
cp .env.example .env
zed .env        # или open -e .env
```

Вставь ключ после `=` без кавычек и пробелов:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx
```

⚠️ Ключ **не пишем** ни в код, ни в `.env.example`, ни в чат (в том числе мне 🙂).

---

## Шаг 6. Ollama: локальные модели

Ollama у тебя уже стоит. Если он не запущен:

```bash
open -a Ollama
```

Скачиваем модели:

```bash
ollama pull qwen3.5:4b             # ~4 ГБ — быстрая модель для экспериментов
ollama pull qwen3-embedding:0.6b   # ~0.6 ГБ — эмбеддинги, понадобится в модуле 4
```

По желанию (скачай, пока читаешь следующий урок):

```bash
ollama pull qwen3.5:9b             # ~7.6 ГБ — основная локальная модель для RAG
ollama rm llama3                   # старая модель, освободит 4.7 ГБ
```

> Имена моделей быстро меняются. Актуальный каталог: <https://ollama.com/library>.

Первый разговор:

```bash
ollama run qwen3.5:4b "Объясни в двух предложениях, что такое токен в LLM"
```

> Если модель сначала выводит «размышления», это нормально: это reasoning-модель. Разберём в уроке 1.6.

🔍 **Мини-эксперимент 1. Сколько памяти занимает модель:**

```bash
ollama ps        # пока модель загружена (≈5 минут после запроса)
```

Сравни колонку SIZE с размером файла из `ollama list`. Почему в памяти модель весит больше, чем на диске? (Подсказка: в уроке было про «запас на обработку текста».)

🔍 **Мини-эксперимент 2. «Универсальная розетка».** Отправь запрос локальной модели в **формате OpenAI**:

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:4b",
    "messages": [{"role": "user", "content": "Привет! Ответь одним словом: столица Узбекистана?"}]
  }' | python3 -m json.tool
```

Посмотри на структуру ответа: `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens`. Ровно такой же формат вернёт OpenAI или OpenRouter. В уроке 0.3 будем работать с ним из Python.

---

## Шаг 7. Jupyter: два способа

### Способ А. Jupyter Lab в браузере

```bash
uv run jupyter lab
```

Создай блокнот в `lessons/00-02-environment-setup/practice/`, в первой ячейке выполни:

```python
import sys
sys.executable   # путь должен вести в .../ai-engineering/.venv/...
```

Остановить сервер: `Ctrl+C` в терминале.

### Способ Б. Ячейки прямо в Zed (рекомендую попробовать)

Создай файл `lessons/00-02-environment-setup/practice/scratch.py`:

```python
# %%
import sys
print(sys.executable)

# %%
x = 21 * 2
x
```

Поставь курсор в ячейку → палитра команд (`cmd-shift-p`) → **`repl: run`**. Если Zed спросит ядро, выбери то, что из `.venv`.

---

## Шаг 8. Финальная проверка

```bash
uv run python lessons/00-02-environment-setup/practice/check_env.py --ping
```

Ожидаемый финал:

```text
🎉 Окружение готово! (предупреждений: N)
```

`--ping` проверяет ключи **бесплатно**: он только запрашивает информацию о ключе или список моделей, без генерации.

---

## Шаг 9. Первый коммит

```bash
git status                 # ещё раз убедись, что .env НЕТ в списке
git add -A
git commit -m "Lessons 0.1–0.2: course plan and environment setup"
```

---

## 🧯 Если что-то пошло не так

| Симптом | Решение |
|---|---|
| `command not found: uv` | Новое окно терминала; проверь `brew list uv` |
| `Ollama не отвечает` / `could not connect` | `open -a Ollama`, подожди 5 сек |
| `Скрипт запущен НЕ в виртуальном окружении` | Запускай через `uv run python ...`, а не `python3 ...` |
| `Python 3.13` вместо 3.12 в проверке | `uv python pin 3.12 && uv sync` |
| Ключ отклонён (HTTP 401/403) | Ключ скопирован не целиком / с пробелом / в кавычках с лишним символом |
| `SSL: CERTIFICATE_VERIFY_FAILED` | `uv add certifi`; если не помогло — покажи мне вывод |
| Мало места на диске | `ollama rm <модель>`; список — `ollama list` |

---

## ✅ Готово, когда

- [ ] `uv --version` работает
- [ ] В корне есть `pyproject.toml`, `uv.lock`, `.python-version`, `.venv/`
- [ ] `.env` заполнен и **не виден** в `git status`
- [ ] `ollama run qwen3.5:4b` отвечает
- [ ] Запрос через `curl` в OpenAI-формате к Ollama работает
- [ ] Jupyter Lab или REPL в Zed запускает ячейку
- [ ] `check_env.py --ping` → 🎉
- [ ] Первый коммит сделан
- [ ] Можешь ответить на вопросы в конце `0.2-environment-setup.md`
