"""
Chạy sau khi train_ner.py đã xong, model lưu ở output_dir/final.

Làm 2 việc:
1. Eval trên Test set (chưa từng đụng tới khi train/tune) — xác nhận Dev
   không phải số liệu may mắn.
2. Sanity test bằng CHÍNH 18 entity mẫu trong đề bài gốc — đây là dữ liệu
   gần với văn phong đề thi thật nhất, và cho biết cụ thể model dự đoán gì
   sai ở đâu (đặc biệt để xem pattern false positive của THUỐC).
"""
import json
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from train_ner import char_entities_to_bio_labels, LABEL2ID, ID2LABEL


# ─── 18 entity mẫu CHÍNH THỨC từ đề bài gốc ───
SAMPLE_TEXT = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ. 1. amlodipine 10 mg "
    "po daily 2. aspirin 81 mg po daily 3. metoprolol succinate xl 50 mg po "
    "daily 4. guaifenesin ml po q6h:prn điều trị ho 5. nystatin oral "
    "suspension 5 ml po qid:prn điều trị đau nhức 6. acetaminophen 325-650 mg "
    "po q6h:prn điều trị sốt đau 7. pravastatin 40 mg po daily 8. docusate "
    "sodium 100 mg po bid điều trị táo bón 9. senna 8.6 mg po bid:prn điều "
    "trị táo bón 10. clonazepam 0.5 mg po qam:prn điều trị lo âu 11. "
    "clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ"
)

EXPECTED_ENTITIES = [
    {"text": "amlodipine 10 mg po daily", "type": "THUỐC"},
    {"text": "aspirin 81 mg po daily", "type": "THUỐC"},
    {"text": "metoprolol succinate xl 50 mg po daily", "type": "THUỐC"},
    {"text": "guaifenesin ml po q6h:prn", "type": "THUỐC"},
    {"text": "ho", "type": "TRIỆU_CHỨNG"},
    {"text": "nystatin oral suspension 5 ml po qid:prn", "type": "THUỐC"},
    {"text": "đau nhức", "type": "TRIỆU_CHỨNG"},
    {"text": "acetaminophen 325-650 mg po q6h:prn", "type": "THUỐC"},
    {"text": "sốt đau", "type": "TRIỆU_CHỨNG"},
    {"text": "pravastatin 40 mg po daily", "type": "THUỐC"},
    {"text": "docusate sodium 100 mg po bid", "type": "THUỐC"},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG"},
    {"text": "senna 8.6 mg po bid:prn", "type": "THUỐC"},
    {"text": "táo bón", "type": "TRIỆU_CHỨNG"},
    {"text": "clonazepam 0.5 mg po qam:prn", "type": "THUỐC"},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG"},
    {"text": "clonazepam 1.5 mg po qhs", "type": "THUỐC"},
    {"text": "lo âu", "type": "TRIỆU_CHỨNG"},
    {"text": "mất ngủ", "type": "TRIỆU_CHỨNG"},
]


# ─── Hàm predict — dùng manual word segmentation (giống train_ner.py) ───

def predict_entities(text, model, tokenizer, id2label):
    """
    Chạy model, merge subword tokens → entities theo word segmentation thủ công.
    PhoBERT Python tokenizer không có offset_mapping nên dùng regex word split.
    """
    # Bước 1: Segment text thành words (giống char_entities_to_bio_labels)
    words = []
    for m in re.finditer(r'\d+\.|[:;,.?!]|\S+', text):
        words.append((m.group(), m.start(), m.end()))

    # Bước 2: Encode từng word riêng để có subword → word mapping
    all_input_ids = [tokenizer.bos_token_id]
    subword_to_word = [(-1, 0, 0)]  # BOS token
    for wid, (w_str, w_cs, w_ce) in enumerate(words):
        sub_ids = tokenizer.encode(w_str, add_special_tokens=False)
        all_input_ids.extend(sub_ids)
        for _ in sub_ids:
            subword_to_word.append((wid, w_cs, w_ce))
    all_input_ids.append(tokenizer.eos_token_id)
    subword_to_word.append((-1, 0, 0))  # EOS token

    # Bước 3: Predict
    input_tensor = torch.tensor([all_input_ids])
    attn = torch.ones_like(input_tensor)
    with torch.no_grad():
        outputs = model(input_tensor, attention_mask=attn)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    # Bước 4: Merge subword BIO → word-level entities
    # Bullet numbers & strong punctuation are NATURAL boundaries —
    # don't merge across them even if model predicts I- tags.
    ENTITY_BREAK_TOKENS = re.compile(r'^\d+\.(?!\d)$|^[;:]$')
    entities = []
    current_words = []
    current_type = None

    for pred_id, (wid, w_cs, w_ce) in zip(predictions, subword_to_word):
        if wid == -1:
            # BOS/EOS — flush entity
            if current_words:
                full_start = current_words[0][1]
                full_end = current_words[-1][2]
                entities.append({
                    "text": text[full_start:full_end],
                    "type": current_type,
                    "position": [full_start, full_end]
                })
                current_words = []
                current_type = None
            continue

        word_str = text[w_cs:w_ce]
        label = id2label[pred_id]
        is_break = bool(ENTITY_BREAK_TOKENS.match(word_str))

        # Force break: bullet numbers & strong punctuation ALWAYS cut entity
        # and are NEVER part of any entity themselves
        if is_break:
            # Flush current entity first
            if current_words:
                full_start = current_words[0][1]
                full_end = current_words[-1][2]
                entities.append({
                    "text": text[full_start:full_end],
                    "type": current_type,
                    "position": [full_start, full_end]
                })
                current_words = []
                current_type = None
            # Skip this token entirely — it cannot start a new entity
            continue

        if label.startswith("B-"):
            if current_words:
                full_start = current_words[0][1]
                full_end = current_words[-1][2]
                entities.append({
                    "text": text[full_start:full_end],
                    "type": current_type,
                    "position": [full_start, full_end]
                })
            current_words = [(wid, w_cs, w_ce)]
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_words.append((wid, w_cs, w_ce))
        else:
            if current_words:
                full_start = current_words[0][1]
                full_end = current_words[-1][2]
                entities.append({
                    "text": text[full_start:full_end],
                    "type": current_type,
                    "position": [full_start, full_end]
                })
                current_words = []
                current_type = None

    # Entity cuối cùng
    if current_words:
        full_start = current_words[0][1]
        full_end = current_words[-1][2]
        entities.append({
            "text": text[full_start:full_end],
            "type": current_type,
            "position": [full_start, full_end]
        })

    return entities


# ─── Load model ───

def load_model(output_dir="ner_model_output/final"):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForTokenClassification.from_pretrained(output_dir)
    model.eval()
    return tokenizer, model


# ═══════════════════════════════════════════════════════════════════
# TEST 1: SANITY CHECK 18 ENTITY MẪU
# ═══════════════════════════════════════════════════════════════════

def sanity_check(tokenizer, model):
    print("=" * 70)
    print("SANITY TEST: 18 entity mẫu từ đề bài gốc")
    print("=" * 70)

    predicted = predict_entities(SAMPLE_TEXT, model, tokenizer, ID2LABEL)

    # Gom bằng dict: key = (text_lower, type) để so sánh dễ dàng
    expected_lookup = {}
    for e in EXPECTED_ENTITIES:
        key = (e["text"].lower(), e["type"])
        expected_lookup.setdefault(key, 0)
        expected_lookup[key] += 1

    predicted_lookup = {}
    for e in predicted:
        key = (e["text"].lower().strip(), e["type"])
        predicted_lookup.setdefault(key, 0)
        predicted_lookup[key] += 1

    print(f"\nExpected: {len(EXPECTED_ENTITIES)} entities")
    print(f"Predicted: {sum(predicted_lookup.values())} entities\n")

    hits, misses, fps = [], [], []

    # Khớp chính xác
    for key, exp_count in expected_lookup.items():
        pred_count = predicted_lookup.get(key, 0)
        if pred_count >= exp_count:
            for _ in range(exp_count):
                hits.append(key)
            if pred_count > exp_count:
                for _ in range(pred_count - exp_count):
                    fps.append(("extra duplicate", key))
        elif pred_count == 0:
            for _ in range(exp_count):
                misses.append(("not found at all", key))
        else:  # pred_count < exp_count
            for _ in range(pred_count):
                hits.append(key)
            for _ in range(exp_count - pred_count):
                misses.append(("missing some occurrences", key))

    # Predicted thừa (không có trong expected)
    for key, pred_count in predicted_lookup.items():
        if key not in expected_lookup:
            for _ in range(pred_count):
                fps.append(("FALSE POSITIVE", key))

    print(f"  KHỚP CHÍNH XÁC: {len(hits)}/{len(EXPECTED_ENTITIES)}")
    print(f"  MISS (không tìm thấy): {len(misses)}")
    print(f"  THỪA (false positive): {len(fps)}")
    print()

    if misses:
        print("─" * 70)
        print("CHI TIẾT MISS:")
        for reason, (text, etype) in misses:
            print(f"  ❌ [{reason}] {etype}: \"{text}\"")
        print()

    if fps:
        print("─" * 70)
        print("CHI TIẾT THỪA (FALSE POSITIVE) — ĐÂY LÀ MANH MỐI LỖI THUỐC:")
        for reason, (text, etype) in fps:
            print(f"  ⚠️  [{reason}] model đoán {etype}: \"{text}\"")
        print()

    print("─" * 70)
    print("DANH SÁCH MODEL DỰ ĐOÁN (đầy đủ):")
    for e in predicted:
        print(f"  [{e['type']}] \"{e['text']}\"")
    print()

    acc = len(hits) / len(EXPECTED_ENTITIES) if EXPECTED_ENTITIES else 0
    print(f"Sanity Accuracy: {acc:.1%} ({len(hits)}/{len(EXPECTED_ENTITIES)})")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: EVAL TRÊN TEST SET
# ═══════════════════════════════════════════════════════════════════

def eval_test_set(tokenizer, model, test_path="combined_test.jsonl"):
    from train_ner import load_jsonl

    print("\n" + "=" * 70)
    print("TEST SET EVAL")
    print("=" * 70)

    records = load_jsonl(test_path)
    print(f"Test set: {len(records)} câu")

    tp_total, fp_total, fn_total = {}, {}, {}

    for rec in records:
        text = rec["text"]
        gold_entities = rec["entities"]

        # Predict
        pred_entities = predict_entities(text, model, tokenizer, ID2LABEL)

        # Normalize để so sánh: key = (text lọc whitespace, type)
        def norm(e):
            return (re.sub(r'\s+', ' ', e["text"] or '').strip().lower(), e["type"])

        gold_set = {norm(e) for e in gold_entities if e["type"] in {"BỆNH", "THUỐC", "TRIỆU_CHỨNG", "THÔNG_TIN_BỆNH_NHÂN", "KẾT_QUẢ_XÉT_NGHIỆM"}}
        pred_set = {norm(e) for e in pred_entities}

        for key in gold_set:
            tp_total.setdefault(key[1], {"tp": 0, "fn": 0, "fp": 0})
            if key in pred_set:
                tp_total[key[1]]["tp"] += 1
            else:
                tp_total[key[1]]["fn"] += 1

        for key in pred_set:
            tp_total.setdefault(key[1], {"tp": 0, "fn": 0, "fp": 0})
            if key not in gold_set:
                tp_total[key[1]]["fp"] += 1

    print("\n=== Test Set Per-Type Stats ===")
    total_tp, total_fn, total_fp = 0, 0, 0
    for etype in sorted(tp_total.keys()):
        d = tp_total[etype]
        total_tp += d["tp"]
        total_fn += d["fn"]
        total_fp += d["fp"]
        prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) > 0 else 0
        rec = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"  {etype:25s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"(tp={d['tp']}, fn={d['fn']}, fp={d['fp']})")

    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0

    print(f"\n  {'TỔNG':25s}  P={total_prec:.3f}  R={total_rec:.3f}  F1={total_f1:.3f}  "
          f"(tp={total_tp}, fn={total_fn}, fp={total_fp})")
    print(f"  Số liệu Dev tham chiếu: P=0.805 R=0.858 F1=0.831")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    # Tự động tìm file
    model_path = os.path.join("ner_model_output", "final")
    if not os.path.exists(model_path):
        print(f"⚠️  Không tìm thấy {model_path}, thử các vị trí khác...")
        for alt in ["ner_model_output/final", "../ner_model_output/final", "/content/ner_model_output/final"]:
            if os.path.exists(alt):
                model_path = alt
                break

    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()

    # ===== SANITY TEST =====
    sanity_check(tokenizer, model)

    # ===== TEST SET EVAL =====
    test_files = [f for f in ["combined_test.jsonl", "ner_data/combined_test.jsonl",
                               "converted_data/combined_test.jsonl",
                               "ner_data/converted_data/combined_test.jsonl",
                               "/content/combined_test.jsonl",
                               "/content/ner_data/converted_data/combined_test.jsonl"]
                  if os.path.exists(f)]
    if test_files:
        eval_test_set(tokenizer, model, test_path=test_files[0])
    else:
        print("\n⚠️  Không tìm thấy file test set (.jsonl).")
        print("   Upload combined_test.jsonl vào cùng thư mục với script này rồi chạy lại.")