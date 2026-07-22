import json
import argparse
import sys

try:
    from predict import predict_entities
    from evaluate import evaluate_ner
    from constants import LABEL_LIST, LABEL2ID, ID2LABEL
except ImportError:
    from clinical_nlp.ner.predict import predict_entities
    from clinical_nlp.ner.evaluate import evaluate_ner
    from clinical_nlp.ner.constants import LABEL_LIST, LABEL2ID, ID2LABEL


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="ner_model_output/final")
    parser.add_argument("--test_path", default="data/ner/combined_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Chỉ test N sample đầu (0 = all)")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    records = load_jsonl(args.test_path)
    if args.max_samples > 0:
        records = records[:args.max_samples]
    print(f"Test set: {len(records)} câu\n")

    correct, total = 0, 0
    errors = []

    for i, rec in enumerate(records):
        text = rec["text"]
        gold = rec["entities"]

        pred = predict_entities(text, model, tokenizer, ID2LABEL)

        # Exact match
        gold_set = {(e["text"], e["type"]) for e in gold}
        pred_set = {(e["text"], e["type"]) for e in pred}
        matches = len(gold_set & pred_set)
        correct += matches
        total += len(gold_set)

        if matches < len(gold_set):
            missed = gold_set - pred_set
            extra = pred_set - gold_set
            if len(errors) < 20:
                errors.append({
                    "idx": i,
                    "text": text[:100],
                    "missed": list(missed),
                    "extra": list(extra),
                })

        if i < 10 or i % 200 == 0:
            print(f"[{i}/{len(records)}] '{text[:60]}...' → {len(gold)} gold, {len(pred)} pred, matched={matches}")

    recall = correct / total * 100 if total > 0 else 0
    print(f"EXACT MATCH RECALL: {correct}/{total} = {recall:.1f}%")

    if errors:
        print(f"\nTOP ERRORS (first {len(errors)})")
        for e in errors:
            print(f"  [{e['idx']}] {e['text']}")
            if e["missed"]:
                print(f"    BỎ SÓT: {e['missed']}")
            if e["extra"]:
                print(f"    DƯ: {e['extra']}")


if __name__ == "__main__":
    main()
