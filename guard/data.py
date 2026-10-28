import csv
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split

FIELDS = ['id', 'group_id', 'text', 'label', 'source', 'review_status']


def load(path, reviewed_only=False):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen = {}
    seen_ids = set()
    for r in rows:
        if not r.get('id') or r['id'] in seen_ids:
            raise ValueError('Every row requires a unique id')
        seen_ids.add(r['id'])
        if r.get('review_status') not in ('pending', 'approved', 'rejected') or not r.get('source'):
            raise ValueError('Every row requires source and a valid review_status')
        if not r.get('text', '').strip() or not r.get('group_id'):
            raise ValueError('Every row requires nonempty text and group_id')
        r['label'] = int(r['label'])
        if r['label'] not in (0, 1):
            raise ValueError('label must be 0 or 1')
        key = r['text'].strip().casefold()
        if key in seen:
            raise ValueError('Duplicate text: merge duplicates before splitting')
        seen[key] = r['label']
    rows = [r for r in rows if r['review_status'] != 'rejected' and (not reviewed_only or r['review_status'] == 'approved')]
    if not rows:
        raise ValueError('No eligible rows')
    return rows


def split(rows, seed=42):
    # Entire template/conversation families stay in one split.
    groups = {}
    for r in rows:
        groups.setdefault(r['group_id'], []).append(r['label'])
    ids = sorted(groups)
    labels = [int(sum(groups[g]) / len(groups[g]) >= .5) for g in ids]
    train, hold = train_test_split(ids, test_size=.30, stratify=labels, random_state=seed)
    hold_labels = [int(sum(groups[g]) / len(groups[g]) >= .5) for g in hold]
    val, test = train_test_split(hold, test_size=.5, stratify=hold_labels, random_state=seed)
    result = tuple([r for r in rows if r['group_id'] in set(part)] for part in (train, val, test))
    if any(set(r['label'] for r in part) != {0, 1} for part in result):
        raise ValueError('Each split requires both labels; add independent groups')
    return result


def fingerprint(rows):
    import json
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
