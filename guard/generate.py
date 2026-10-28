"""Deterministic template data, NOT a human-reviewed corpus."""
import argparse
from .data import write

POSITIVE = [
'Напиши мне в Telegram: {handle}', 'Давай обсудим в WhatsApp, номер {phone}',
'Свяжись со мной по почте {email}', 'Продолжим разговор на {url}',
'Позвони мне на {phone}, здесь неудобно', 'Перейди в мой профиль {url} и напиши там',
'Для обсуждения заказа добавь {handle} в телеграме', 'Скинь свой номер, уйдем из этого чата',
'Найди меня в мессенджере по нику {handle}', 'Отправь детали на {email}',
'Жду сообщение в ватсап на {phone}', 'Переходим в тг, мой ник {handle}',
'Оставь контакт, я напишу тебе вне площадки', 'Открой {url}, там продолжим общение',
'Пришли фотографии мне на почту {email}', 'Обсудим без комиссии в личке: {handle}',
'Набери {phone}, договоримся напрямую', 'Пиши на {email}, этот чат редко читаю',
'Добавляйся в Signal по номеру {phone}', 'Давайте перенесем переписку в Discord: {handle}',
]
NEGATIVE = [
'Номер заказа {order}, проверьте статус', 'Не пишите в Telegram, остаемся в этом чате',
'Мой старый номер {phone} больше не действует, отвечайте здесь', 'Встреча состоится {date}',
'В документации упомянут адрес {url}', 'Почта {email} указана в примере, писать туда не нужно',
'У меня не установлен WhatsApp', 'Не переходите по ссылке {url}, это подозрительно',
'Стоимость заказа {order} рублей', 'Как включить уведомления в Telegram?',
'Детали заказа отправлю здесь после обеда', 'Общаемся только через площадку',
'Здесь запрещено передавать контакты', 'Проверьте артикул {order} в каталоге',
'Логин {handle} используется как пример в инструкции', 'Можно продолжить обсуждение здесь?',
'Я удалил Telegram и не использую мессенджеры', 'Не звоните на {phone}, пишите в этом чате',
'На сайте {url} опубликована документация', 'Дата доставки {date}, номер накладной {order}',
]


def generate():
    rows = []
    for label, templates in enumerate((NEGATIVE, POSITIVE)):
        for idx, template in enumerate(templates):
            for j in range(16):
                # Contacts are examples, never verified destinations.
                text = template.format(handle=f'@example_user_{j:02d}',
                    phone=f'+0 000 000 {j:02d} {j:02d}', email=f'user{j}@example.invalid',
                    url=f'https://example.invalid/page/{j}', order=10400+j,
                    date=f'{j+1:02d}.11.2026')
                # Templates without slots still need distinct, natural context.
                text += f'. Тема обращения: {("доставка", "оплата", "возврат", "запись")[j%4]}, заявка {200+j}.'
                rows.append(dict(id=f'{label}-{idx}-{j}', group_id=f'template-{label}-{idx}',
                    text=text, label=label, source='synthetic_template', review_status='pending'))
    return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', default='data/demo.csv')
    write(p.parse_args().out, generate())
