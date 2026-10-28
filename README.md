# Channel Guard — детектор ухода в сторонние каналы

Python · scikit-learn · PyTorch · Hugging Face Transformers · Streamlit

**Облегчённая версия.** Сохранены код, 640 синтетических сообщений, обученные baseline-модели, метрики четырёх выполненных экспериментов и ноутбук Colab. Веса и токенизаторы ruBERT не включены; для запуска ruBERT выполните обучение по инструкции ниже. Проверка сообщений через baseline доступна после установки requirements.txt. PyTorch и Transformers нужны только для ruBERT.

Учебный ML-проект для русскоязычных сообщений: классифицирует предложение продолжить взаимодействие вне платформы, в том числе при маскировке контактов. Есть сравнение моделей, разметка и проверка устойчивости. Готовый классификатор не заменяет решение модератора.

## Что внутри

- 640 синтетических сообщений, 40 независимых семейств шаблонов, поровну классов 0/1. Все имеют статус pending; человеческая проверка не заявляется.
- Baseline: символьный TF-IDF (2–5-граммы) + логистическая регрессия.
- Дообучение cointegrated/rubert-tiny2; можно выбрать DeepPavlov/rubert-base-cased через --model.
- Обфускации: похожие символы, транслитерация, пробелы внутри слов, невидимые символы, маскировка точек и @.
- Пять аугментаций каждого обучающего сообщения; исходные validation/test не расширяются.
- Интерфейс проверки сообщений и ручной разметки CSV.
- Обмен с LLM через JSONL: инструкции, импорт предварительных меток, журнал обоснований. Автоматических платных запросов нет.

**Статус запусков и фактические метрики:** `reports/RESULTS.md`. Нельзя переносить результаты синтетического набора на реальные сообщения. Синтетические шаблоны и стресс-преобразования не представляют все способы обхода.

## Быстрый запуск на Windows

Установите Python 3.11 или 3.12, распакуйте архив, откройте PowerShell в папке `channel_guard`:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Откройте адрес, показанный Streamlit (обычно http://localhost:8501). Baseline уже лежит в `artifacts/baseline`. Если загрузка сохраненного joblib несовместима с вашей версией scikit-learn, переобучите его следующей командой. Загружайте joblib только из доверенных источников.

```powershell
.\.venv\Scripts\python.exe -m guard.experiment --data data/demo.csv --out artifacts/baseline
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m guard.experiment --out artifacts/baseline
python -m streamlit run app.py
```

## Обучение и сравнение

Команды ниже выполняйте в активированном окружении из корня проекта. В Windows можно заменять python на `.\.venv\Scripts\python.exe`.

```bash
# Восстановить исходные синтетические данные
python -m guard.generate
# Baseline без аугментаций
python -m guard.experiment --out artifacts/baseline
# Baseline с аугментациями
python -m guard.experiment --augment --out artifacts/baseline_augmented
# Установить зависимости Transformer
python -m pip install -r requirements-transformer.txt
# ruBERT-tiny2 без аугментаций
python -m guard.experiment --backend transformer --epochs 3 --batch-size 16 --out artifacts/rubert_tiny2
# ruBERT-tiny2 с аугментациями (дольше)
python -m guard.experiment --backend transformer --augment --epochs 3 --batch-size 16 --out artifacts/rubert_tiny2_augmented
# Собрать сравнительный отчет по выполненным запускам
python -m guard.report
```

Загрузка предобученной модели требует интернета. CPU поддерживается; GPU ускоряет обучение. Для DeepPavlov/rubert-base-cased используйте `--model DeepPavlov/rubert-base-cased` и отдельную папку вывода. Для первого запуска на GPU есть `train_colab.ipynb`: откройте его в Google Colab и загрузите архив проекта. Длительность и доступность GPU зависят от среды.

## Как устроена оценка

Разбиение: 448 train / 96 validation / 96 test, seed=42. Шаблоны группируются целиком до аугментаций, чтобы варианты одной фразы не попадали в разные части. На собственном корпусе группируйте по диалогу/источнику и дополнительно удаляйте близкие дубликаты.

Один и тот же split используется всеми моделями. Лучший checkpoint Transformer выбирается по validation macro-F1 при пороге 0,5. Затем единый рабочий порог подбирается по validation macro-F1; при равенстве выбирается ближайший к 0,5. Тест не используется для выбора порога или checkpoint. Macro-F1 учитывает оба класса, чтобы не поощрять тривиальный ответ «все сообщения — нарушение». В отчете отдельно показан F1 положительного класса — именно он соответствует полю F1 в резюме.

Каждый вариант модели сохраняет:

- `metrics.json`: precision, recall, F1, macro-F1, матрицы ошибок [true 0/1 × predicted 0/1], объемы, hash корпуса, id частей, порог и seed;
- `errors.json`: ошибки на исходном test;
- `settings.json`: рабочий порог и backend;
- baseline.joblib либо папку model с весами и токенизатором, inference.json с лимитом токенов.

Стресс-тест преобразует одну и ту же test-выборку каждым способом отдельно, сохраняя метки. Изменения читаемости и смысла нужно проверять вручную. Оценки на таких автоматических атаках — диагностические, а не доказательство устойчивости. Ограничение длины Transformer может обрезать контакт в конце длинного сообщения.

## Собственный корпус и ручная проверка

Схема CSV: `id, group_id, text, label, source, review_status`. Метки: 0 — нет предложения перейти; 1 — есть. Статусы: pending / approved / rejected. Записи rejected исключаются из обучения всегда; pending доступны только в демонстрационном режиме без --reviewed-only. Источник может быть real_anonymized, llm_synthetic и т. п. Описание и пограничные примеры — в `ANNOTATION.md`.

Загрузите CSV во вкладку «Разметка», проверьте записи, скачайте в `data/reviewed.csv`. Исходный файл не меняется автоматически.

```bash
python -m guard.experiment --data data/reviewed.csv --reviewed-only --out artifacts/reviewed_baseline
```

Для честного проекта в резюме сначала соберите независимые реальные или разнообразные LLM-примеры, проверьте их вручную и повторите все эксперименты. 640 строк здесь — шаблонный демонстрационный корпус, а не 640 независимых наблюдений.

## Проверить одну фразу

```bash
python -m guard.predict "Напиши мне в тг: @example_user"
python -m guard.predict "Не пишите в Telegram, отвечайте здесь"
python -m guard.predict "Перейдем в другой чат" --model-dir artifacts/rubert_tiny2
```

Число score — оценка модели, не калиброванная вероятность. После повторного обучения ruBERT для Transformer в интерфейсе укажите `artifacts/rubert_tiny2` в поле папки модели.

## Проверки

```bash
python -m unittest discover -s tests -v
```

В этой сборке выполнены 7 модульных проверок, проверка интерфейса Streamlit и цикл экспорта/импорта LLM-разметки на тестовых ответах. Проверяются разбиение без пересечения семейств, roundtrip CSV, фильтр проверки человеком, дубликаты, сохранение меток/групп при аугментациях и выбор порога. Версии среды фактического запуска — `reports/environment.json`.

## Источники

- https://huggingface.co/cointegrated/rubert-tiny2
- https://huggingface.co/DeepPavlov/rubert-base-cased
- https://huggingface.co/docs/transformers/main_classes/trainer
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

Код проекта — MIT. Зависимости и исходные модели имеют собственные лицензии.
