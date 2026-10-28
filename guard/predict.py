import argparse
import json
from pathlib import Path


def load_predictor(directory):
    root = Path(directory)
    settings = json.loads((root / 'settings.json').read_text())
    if settings['backend'] == 'baseline':
        import joblib
        model = joblib.load(root / 'baseline.joblib')
        predict = lambda texts: model.predict_proba(texts)[:, 1]
    else:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from .transformer import infer
        tokenizer = AutoTokenizer.from_pretrained(root / 'model')
        model = AutoModelForSequenceClassification.from_pretrained(root / 'model')
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        max_length = json.loads((root / 'inference.json').read_text())['max_length']
        predict = lambda texts: infer(model, tokenizer, texts, max_length)
    return predict, settings['threshold']


def main():
    p = argparse.ArgumentParser(); p.add_argument('text'); p.add_argument('--model-dir', default='artifacts/baseline')
    args = p.parse_args(); predict, threshold = load_predictor(args.model_dir)
    score = float(predict([args.text])[0])
    print(json.dumps(dict(score=score, threshold=threshold, off_platform=score >= threshold), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
