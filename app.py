import pandas as pd
import streamlit as st
from guard.predict import load_predictor
from guard.text import ATTACKS, transform

st.set_page_config(page_title='Channel Guard', page_icon='🔎', layout='wide')
st.title('Детектор ухода в сторонние каналы')
st.caption('Учебный прототип. Оценка модели — сигнал для проверки модератором.')
check, review = st.tabs(['Проверить сообщение', 'Разметка'])
with check:
    directory = st.text_input('Папка модели', 'artifacts/baseline')
    text = st.text_area('Сообщение', 'Давай обсудим в Telegram: @example_user')
    if st.button('Проверить', type='primary'):
        if not text.strip():
            st.warning('Введите сообщение')
        else:
            try:
                predict, threshold = load_predictor(directory)
                texts = [text] + [transform(text, attack) for attack in ATTACKS]
                scores = predict(texts)
                st.metric('Оценка модели', f'{scores[0]:.1%}')
                st.write('Возможен уход с площадки' if scores[0] >= threshold else 'Уход с площадки не обнаружен')
                st.caption(f'Порог {threshold:.3f}. Это не калиброванная вероятность.')
                st.dataframe(pd.DataFrame({'Вариант':['original']+list(ATTACKS),'Текст':texts,'Оценка':scores}), hide_index=True)
            except (OSError, ValueError, ImportError) as exc:
                st.error(f'Не удалось загрузить модель: {exc}. Сначала выполните обучение по README.')
with review:
    st.write('Исправьте label (0/1), поставьте approved после проверки и скачайте CSV.')
    file = st.file_uploader('CSV: id, group_id, text, label, source, review_status', type='csv')
    if file:
        frame = pd.read_csv(file, dtype={'id':str, 'group_id':str})
        required = {'id','group_id','text','label','source','review_status'}
        if not required.issubset(frame.columns):
            st.error('Не хватает колонок: '+', '.join(sorted(required-set(frame.columns))))
        else:
            edited = st.data_editor(frame, hide_index=True, disabled=['id','group_id','source'],
                column_config={'label': st.column_config.SelectboxColumn(options=[0,1], required=True),
                    'review_status': st.column_config.SelectboxColumn(options=['pending','approved','rejected'], required=True)})
            st.download_button('Скачать разметку', edited.to_csv(index=False).encode('utf-8-sig'), 'reviewed.csv', 'text/csv')
