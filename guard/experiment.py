import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, f1_score
from .data import load, split, fingerprint
from .text import augment, transform, ATTACKS


def metrics(y, p, threshold):
    pred = np.asarray(p) >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(precision=float(precision), recall=float(recall), f1=float(f1), macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(), n=len(y))


def select_threshold(y, p):
    return float(max(np.linspace(.05, .95, 91), key=lambda t: (metrics(y, p, t)['macro_f1'], -abs(t-.5))))


def evaluate(test, predict, threshold):
    y = [r['label'] for r in test]
    results = {'clean': metrics(y, predict([r['text'] for r in test]), threshold)}
    for attack in ATTACKS:
        results[attack] = metrics(y, predict([transform(r['text'], attack) for r in test]), threshold)
    return results


def run(args):
    rows = load(args.data, args.reviewed_only)
    train, val, test = split(rows, args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_rows = augment(train) if args.augment else train
    texts, y = [r['text'] for r in train_rows], [r['label'] for r in train_rows]
    if args.backend == 'baseline':
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        model = Pipeline([('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
                          ('clf', LogisticRegression(C=3, max_iter=1000, class_weight='balanced', random_state=args.seed))])
        model.fit(texts, y)
        def predict(batch):
            return model.predict_proba(batch)[:, 1]
        joblib.dump(model, out / 'baseline.joblib')
    else:
        from .transformer import fit
        predict = fit(train_rows, val, args, out)
    threshold = select_threshold([r['label'] for r in val], predict([r['text'] for r in val]))
    scores = evaluate(test, predict, threshold)
    report = dict(backend=args.backend, model=args.model if args.backend == 'transformer' else 'char-tfidf-logreg',
        training_config=dict(epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, learning_rate=2e-5) if args.backend == 'transformer' else dict(C=3, ngram_range=[2,5]), seed=args.seed, augmented=args.augment, threshold=threshold, dataset_sha256=fingerprint(rows),
        source_counts={s: sum(r['source'] == s for r in rows) for s in sorted({r['source'] for r in rows})},
        reviewed_only=args.reviewed_only, counts=dict(total=len(rows), train=len(train), validation=len(val), test=len(test), train_after_augmentation=len(train_rows)),
        split_ids={name: [r['id'] for r in part] for name, part in zip(('train','validation','test'), (train,val,test))},
        threshold_objective='validation_macro_f1', dummy_all_positive=metrics([r['label'] for r in test], np.ones(len(test)), .5), metrics=scores, limitations='Synthetic template results do not estimate real-world quality. Attacks require manual semantic review.')
    (out / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'settings.json').write_text(json.dumps(dict(threshold=threshold, backend=args.backend)), encoding='utf-8')
    probabilities = predict([r['text'] for r in test])
    errors = [{**r, 'score': float(p), 'prediction': int(p >= threshold)} for r, p in zip(test, probabilities) if int(p >= threshold) != r['label']]
    (out / 'errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'threshold': threshold, 'metrics': scores}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/demo.csv'); p.add_argument('--out', default='artifacts/baseline')
    p.add_argument('--backend', choices=['baseline','transformer'], default='baseline')
    p.add_argument('--model', default='cointegrated/rubert-tiny2')
    p.add_argument('--augment', action='store_true'); p.add_argument('--reviewed-only', action='store_true')
    p.add_argument('--seed', type=int, default=42); p.add_argument('--epochs', type=float, default=3)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--max-length', type=int, default=192)
    run(p.parse_args())

if __name__ == '__main__':
    main()
