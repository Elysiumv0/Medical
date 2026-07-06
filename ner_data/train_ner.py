"""
Training script cho NER (Phase 4). Fine-tune PhoBERT hoặc ViHealthBERT trên
bộ combined_train/dev/test.jsonl đã chuẩn bị.

CẦN CHẠY Ở MÁY CÓ INTERNET (tải model từ HuggingFace) — sandbox này chặn
huggingface.co nên không test được với model thật, chỉ verify logic bằng
mock tokenizer (xem test_alignment_logic.py đi kèm).

Cài đặt:
    pip install transformers datasets seqeval torch accelerate

Chạy:
    python3 train_ner.py --model_name vinai/phobert-base
    python3 train_ner.py --model_name demdecuong/vihealthbert-base-word  # nếu có
"""
import json
import argparse
import numpy as np
import os


LABEL_LIST = [
    "O",
    "B-THUỐC", "I-THUỐC",
    "B-BỆNH", "I-BỆNH",
    "B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG",
    "B-THÔNG_TIN_BỆNH_NHÂN", "I-THÔNG_TIN_BỆNH_NHÂN",
    "B-KẾT_QUẢ_XÉT_NGHIỆM", "I-KẾT_QUẢ_XÉT_NGHIỆM",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def load_jsonl(path):
    """Load JSONL, tự động tìm trong mọi ngóc ngách của Colab."""
    base_name = os.path.basename(path)

    # Danh sách các đường dẫn tiềm năng, bao gồm cả các folder lồng nhau
    possible_paths = [
        path,                                             # 1. Đúng như args truyền vào
        os.path.join("ner_data", "converted_data", base_name), # 2. Cấu trúc chuẩn notebook
        os.path.join("converted_data", base_name),        # 3. Folder converted_data ở root
        base_name,                                        # 4. Ngay tại root /content/
        os.path.join("ner_data", base_name),              # 5. Trong folder ner_data/
    ]

    for p in possible_paths:
        try:
            with open(p, encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Could not find {base_name} in any of the searched locations: {possible_paths}")


def char_entities_to_bio_labels(text: str, entities: list, tokenizer, max_length=256):
    """
    PHOBERT-SPECIFIC BIO alignment. PhoBERT Python tokenizer không hỗ trợ
    return_offsets_mapping / word_ids() / pre_tokenize — phải tokenize thủ công.

    Cách làm:
    1. Regex segment text → words + char offsets
    2. Encode từng word riêng → subword tokens
    3. Map entity char-span → word range → subword BIO tags

    Trả về: dict với input_ids, attention_mask, labels (list int theo LABEL2ID)
    """
    import re

    entity_types = {"THUỐC", "BỆNH", "TRIỆU_CHỨNG", "THÔNG_TIN_BỆNH_NHÂN", "KẾT_QUẢ_XÉT_NGHIỆM"}

    # ---- Bước 1: Segment text thành words + track char offsets ----
    # Regex tách: số bullet (1., 2.), dấu câu (: ; . ,), và từ thông thường
    words = []  # (word_str, char_start, char_end)
    for m in re.finditer(r'\d+\.|[:;,.?!]|\S+', text):
        words.append((m.group(), m.start(), m.end()))

    # ---- Bước 2: Per-word encode → subword tokens + ánh xạ subword→word ----
    all_input_ids = []
    subword_to_word = []  # sub_idx → (word_idx, char_start, char_end)
    for wid, (w_str, w_cs, w_ce) in enumerate(words):
        sub_ids = tokenizer.encode(w_str, add_special_tokens=False)
        all_input_ids.extend(sub_ids)
        for _ in sub_ids:
            subword_to_word.append((wid, w_cs, w_ce))

    # Truncate to max_length
    all_input_ids = all_input_ids[:max_length]
    subword_to_word = subword_to_word[:max_length]

    # ---- Bước 3: Init labels all O ----
    labels = [LABEL2ID["O"]] * len(all_input_ids)

    # ---- Bước 4: Map entity char-span → subword BIO tags ----
    for entity in entities:
        e_start, e_end = entity["position"]
        e_type = entity["type"]
        if e_type not in entity_types:
            continue

        first_token_in_entity = True
        for sub_i, (wid, w_cs, w_ce) in enumerate(subword_to_word):
            # overlap: word span khớp với entity char span?
            if w_cs < e_end and w_ce > e_start:
                if first_token_in_entity:
                    labels[sub_i] = LABEL2ID[f"B-{e_type}"]
                    first_token_in_entity = False
                else:
                    labels[sub_i] = LABEL2ID[f"I-{e_type}"]

    # ---- Bước 5: Bọc special tokens và attention_mask ----
    input_ids = [tokenizer.bos_token_id] + all_input_ids + [tokenizer.eos_token_id]
    labels = [LABEL2ID["O"]] + labels + [LABEL2ID["O"]]
    attention_mask = [1] * len(input_ids)

    encoding = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    return encoding


def prepare_dataset(records, tokenizer):
    all_encodings = []
    for rec in records:
        enc = char_entities_to_bio_labels(rec["text"], rec["entities"], tokenizer)
        all_encodings.append(enc)
    return all_encodings


def compute_metrics_fn(eval_pred):
    """Entity-level F1 bằng seqeval — ĐÚNG chuẩn cho bài toán NER, không dùng
    accuracy thô (sẽ bị inflate bởi số lượng lớn nhãn 'O')."""
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
    parser.add_argument("--train_path", default="combined_train.jsonl")
    parser.add_argument("--dev_path", default="combined_dev.jsonl")
    parser.add_argument("--output_dir", default="ner_model_output")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification,
        TrainingArguments, Trainer, DataCollatorForTokenClassification,
    )
    import torch

    print(f"Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name, num_labels=len(LABEL_LIST),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )

    print("Loading data...")
    train_records = load_jsonl(args.train_path)
    dev_records = load_jsonl(args.dev_path)

    print(f"Train: {len(train_records)} câu, Dev: {len(dev_records)} câu")
    print("Tokenizing + aligning BIO labels (có thể mất vài phút)...")

    train_encodings = prepare_dataset(train_records, tokenizer)
    dev_encodings = prepare_dataset(dev_records, tokenizer)

    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
            return item

    train_dataset = NERDataset(train_encodings)
    dev_dataset = NERDataset(dev_encodings)

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
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics_fn,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("Bắt đầu training...")
    trainer.train()

    print("\nLưu model cuối cùng...")
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")

    print("\n=== Đánh giá cuối cùng trên dev set ===")
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
