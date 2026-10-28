import re
import unicodedata

# Conservative normalization; original text remains available for modeling.
HOMOGLYPHS = str.maketrans({'а':'a','е':'e','о':'o','р':'p','с':'c','у':'y','х':'x','к':'k','м':'m','т':'t','в':'b','н':'h'})
TRANSLIT = dict(zip('абвгдеёзийклмнопрстуфхцчшщъыьэюя',
                    ['a','b','v','g','d','e','yo','z','i','y','k','l','m','n','o','p','r','s','t','u','f','kh','ts','ch','sh','sch','','y','','e','yu','ya']))


def normalize(text):
    text = unicodedata.normalize('NFKC', str(text)).lower()
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    text = re.sub(r'\s*\[точка\]\s*', '.', text)
    text = re.sub(r'\s*\[собака\]\s*', '@', text)
    return re.sub(r'\s+', ' ', text).strip()


def transform(text, kind):
    if kind == 'homoglyphs':
        return text.lower().translate(HOMOGLYPHS)
    if kind == 'translit':
        return ''.join(TRANSLIT.get(c, c) for c in text.lower())
    if kind == 'spaces':
        # Space long tokens, phone digits and domain components.
        return re.sub(r'\w{5,}', lambda m: ' '.join(m[0]), text)
    if kind == 'zero_width':
        return '\u200b'.join(text)
    if kind == 'masked_links':
        return text.replace('.', ' [точка] ').replace('@', ' [собака] ')
    raise ValueError(kind)


ATTACKS = ('homoglyphs', 'translit', 'spaces', 'zero_width', 'masked_links')


def augment(rows):
    out = list(rows)
    for row in rows:
        for attack in ATTACKS:
            out.append({**row, 'text': transform(row['text'], attack), 'augmentation': attack})
    return out
