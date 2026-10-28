"""Create a truthful report only from completed experiments."""
import json
import importlib.metadata
import platform
from pathlib import Path
from .text import ATTACKS


def main():
    reports=[]
    for path in sorted(Path('artifacts').glob('*/metrics.json')):
        r=json.loads(path.read_text(encoding='utf-8')); reports.append((path.parent.name,r))
    lines=['# Фактические результаты', '',
        'Корпус: 640 синтетических шаблонных сообщений; ручная проверка не выполнена. '
        'Разбиение по 40 семействам шаблонов, seed=42. Это демонстрационный эксперимент, '
        'а не оценка качества на реальном трафике.', '',
        '| Модель | Аугментации | F1 (класс 1) | Macro-F1 | Precision | Recall |',
        '|---|---|---:|---:|---:|---:|']
    for name,r in reports:
        m=r['metrics']['clean']
        lines.append(f"| {name} | {'да' if r['augmented'] else 'нет'} | {m['f1']:.3f} | {m['macro_f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} |")
    lines+=['','## Устойчивость: F1 положительного класса', '',
        '| Модель | '+ ' | '.join(ATTACKS)+' |','|---|'+'---:|'*len(ATTACKS)]
    for name,r in reports:
        lines.append('| '+name+' | '+' | '.join(f"{r['metrics'][a]['f1']:.3f}" for a in ATTACKS)+' |')
    lines+=['','Числа выше получены реальными запусками. Невыполненные эксперименты в таблицу не добавляются. '
        'Сравнивайте их с матрицами ошибок и dummy_all_positive в metrics.json: '
        'F1=0,667 на сбалансированном тесте возможен у модели, которая помечает все как класс 1. '
        'Поэтому высокий F1 сам по себе не доказывает полезность.', '',
        'Тест содержит всего 6 семейств шаблонов, поэтому оценка нестабильна. Аугментации не гарантируют улучшения: итоговый порог выбран на исходной validation. '
        'Каждая атака оценивается отдельно; распределения оценок могут меняться.', '',
        '## Формулировка для резюме на текущем этапе', '',
        'Разработала учебный детектор ухода в сторонние каналы: подготовила 640 синтетических сообщений, '
        'обучила baseline на TF-IDF и логистической регрессии, дообучила ruBERT-tiny2 с аугментациями и без них, '
        'реализовала интерфейс ручной разметки и проверку пяти типов обфускации. '
        'Фактически завершенные запуски и их метрики приведены выше. '
        'Ручную разметку и результаты на реальных данных пока не заявляю.', '',
        'Для итогового пункта «дообучила ruBERT, F1=X против baseline Y» '
        'используйте только сопоставимые завершенные эксперименты на проверенном корпусе.']
    Path('reports').mkdir(exist_ok=True)
    Path('reports/RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    versions={}
    for name in ['numpy','scikit-learn','joblib','torch','transformers','accelerate','streamlit']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    Path('reports/environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions},indent=2),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__': main()
