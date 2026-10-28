# Как запустить облегчённую версию

В этой сборке нет тяжёлых весов ruBERT. Код обучения и результаты экспериментов сохранены.

1. Установите Python 3.11 или 3.12.
2. Распакуйте архив и откройте PowerShell в папке channel_guard.
3. Выполните:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Откройте показанный адрес (обычно http://localhost:8501).
Выбранная по умолчанию модель artifacts/baseline уже обучена.
Если появляется ошибка несовместимости сохранённой модели, переобучите её:

```powershell
.\.venv\Scripts\python.exe -m guard.experiment --out artifacts/baseline
```

Полное описание проекта, обучение ruBERT и запуск в Google Colab — в README.md.
