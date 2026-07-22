import json
import argparse
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    from predict import predict_entities
    from constants import ID2LABEL
except ImportError:
    from clinical_nlp.ner.predict import predict_entities
    from clinical_nlp.ner.constants import ID2LABEL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", required=True)
    parser.add_argument("test_path", required=True)
    parser.add_argument("max_samples", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()
    print(f"Tokenizer: {type(tok).__name__}, is_fast={tok.is_fast}")

    with open(args.test_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if args.max_samples > 0:
        records = records[:args.max_samples]

    print(f"Test set: {len(records)} câu\n")

    correct, total = 0, 0
    errors = []

    for i, rec in enumerate(records):
        text = rec["text"]
        gold = rec["entities"]
        pred = predict_entities(text, model, tok, ID2LABEL)

        gold_set = {(e["text"], e["type"]) for e in gold}
        pred_set = {(e["text"], e["type"]) for e in pred}
        matches = len(gold_set & pred_set)
        correct += matches
        total += len(gold_set)

        if matches < len(gold_set):
            missed = gold_set - pred_set
            extra = pred_set - gold_set
            if len(errors) < 30:
                errors.append({
                    "idx": i,
                    "text": text[:120],
                    "missed": sorted(missed),
                    "extra": sorted(extra),
                })

        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(records)}] running recall={correct/total*100:.1f}%...")

    recall = correct / total * 100 if total > 0 else 0
    print(f"EXACT MATCH RECALL: {correct}/{total} = {recall:.1f}%")

    if errors:
        print(f"\n TOP ERRORS ({len(errors)} shown) ")
        for e in errors:
            print(f"  [{e['idx']}] {e['text']}")
            if e["missed"]:
                print(f"    BỎ SÓT: {e['missed']}")
            if e["extra"]:
                print(f"    DƯ: {e['extra']}")
            print()


if __name__ == "__main__":
    main()
