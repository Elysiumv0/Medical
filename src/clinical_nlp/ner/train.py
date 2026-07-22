"""
Training script cho NER (Phase 4). Fine-tune PhoBERT hoặc ViHealthBERT.

Usage:
    pip install transformers datasets seqeval torch accelerate
    python3 src/clinical_nlp/ner/train.py --model_name vinai/phobert-base
    python3 src/clinical_nlp/ner/train.py --model_name demdecuong/vihealthbert-base-word
"""

import json
import argparse
import numpy as np

try:
    from alignment import char_entities_to_bio_labels
    from constants import LABEL_LIST, LABEL2ID, ID2LABEL
except ImportError:
    from clinical_nlp.ner.alignment import char_entities_to_bio_labels
    from clinical_nlp.ner.constants import LABEL_LIST, LABEL2ID, ID2LABEL

def compute_metrics_fn(eval_pred):
    """Entity-level F1 bằng seqeval — chuẩn cho NER."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for p, l in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    print(classification_report(true_labels, true_predictions, zero_division=0))

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "per_type": {k: v for k, v in report.items() if k not in ("micro avg", "macro avg", "weighted avg")},
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vinai/phobert-base")
    parser.add_argument("--train_path", default="data/ner/combined_train_augmented.jsonl")
    parser.add_argument("--dev_path", default="data/ner/combined_dev.jsonl")
    parser.add_argument("--output_dir", default="ner_model_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
    )
    from datasets import load_dataset
    import torch

    print(f"Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL_LIST),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    print("Loading data using datasets library...")
    dataset = load_dataset("json", data_files={
        "train": args.train_path,
        "validation": args.dev_path
    })

    def tokenize_and_align(examples):
        texts = examples["text"]
        entities_list = examples["entities"]
        all_labels, all_input_ids, all_attention_masks = [], [], []
        for text, entities in zip(texts, entities_list):
            enc = char_entities_to_bio_labels(text, entities, tokenizer)
            all_input_ids.append(enc["input_ids"])
            all_attention_masks.append(enc["attention_mask"])
            all_labels.append(enc["labels"])
        return {"input_ids": all_input_ids, "attention_mask": all_attention_masks, "labels": all_labels}

    print("Tokenizing + aligning BIO labels in parallel...")
    tokenized_datasets = dataset.map(
        tokenize_and_align,
        batched=True,
        batch_size=1000,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics_fn,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("Bắt đầu training...")
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    print("\n=== Đánh giá cuối cùng trên dev set ===")
    print(trainer.evaluate())

if __name__ == "__main__":
    main()