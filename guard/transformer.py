"""Loaded only when the transformer backend is requested."""
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer, set_seed)
from .experiment import metrics


class Messages(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.encoding = tokenizer([r['text'] for r in rows], truncation=True, max_length=max_length)
        self.labels = [r['label'] for r in rows]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {**{k: v[i] for k, v in self.encoding.items()}, 'labels': self.labels[i]}


def fit(train, val, args, out):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2,
        id2label={0:'ON_PLATFORM',1:'OFF_PLATFORM'}, label2id={'ON_PLATFORM':0,'OFF_PLATFORM':1})
    def compute(pred):
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        result = metrics(labels, probs, .5)
        return {k: result[k] for k in ('precision','recall','f1','macro_f1')}
    training = TrainingArguments(output_dir=str(out / 'checkpoints'),
        num_train_epochs=args.epochs, learning_rate=2e-5, weight_decay=.01,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='macro_f1', greater_is_better=True, save_total_limit=1,
        seed=args.seed, data_seed=args.seed, report_to='none', fp16=False)
    trainer = Trainer(model=model, args=training, processing_class=tokenizer,
        train_dataset=Messages(train, tokenizer, args.max_length),
        eval_dataset=Messages(val, tokenizer, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute)
    trainer.train()
    trainer.save_model(str(out / 'model')); tokenizer.save_pretrained(str(out / 'model'))
    import json
    (out / 'inference.json').write_text(json.dumps({'max_length': args.max_length}), encoding='utf-8')
    return lambda texts: infer(model, tokenizer, texts, args.max_length, args.batch_size)


def infer(model, tokenizer, texts, max_length=192, batch_size=8):
    model.eval(); result = []; device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                               max_length=max_length, return_tensors='pt').to(device)
            result.extend(model(**inputs).logits.softmax(-1)[:,1].cpu().tolist())
    return np.asarray(result)
