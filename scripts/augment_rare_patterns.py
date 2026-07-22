"""
Targeted data augmentation — tập trung chính xác vào 2 bug đã phát hiện
từ Phase 4 evaluation, không sinh data chung chung.

Bug 1: Model không biết format "danh sách đánh số" trong EHR thật
       → sinh câu có pattern "1. thuốc_a... 2. thuốc_b..."
Bug 2: 2 entity liền kề cùng type bị gộp thành 1
       → sinh câu có "triệu_chứng_a triệu_chứng_b" không dấu phẩy

Cộng thêm: tăng cường nhãn hiếm (THUỐC, KẾT_QUẢ_XÉT_NGHIỆM)
để giảm imbalance.

Usage:
    python3 scripts/augment_rare_patterns.py --out data/ner/synthetic_rare_patterns.jsonl
"""

import json
import random
import argparse
from datetime import datetime

# ── Templates cho Bug 1: danh sách đánh số ──────────────────────────
# Mỗi template có placeholder {numbered_list} sẽ được thay bằng chuỗi
# "1. thuốc_a liều_dùng. 2. thuốc_b liều_dùng. ..."

NUMBERED_LIST_TEMPLATES = [
    "Thuốc trước khi nhập viện: {numbered_list}",
    "Đơn thuốc hiện tại: {numbered_list}",
    "Thuốc đang dùng: {numbered_list}",
    "Bệnh nhân đang sử dụng các thuốc sau: {numbered_list}",
    "Toa thuốc xuất viện: {numbered_list}",
    "Thuốc điều trị ngoại trú: {numbered_list}",
]

# Danh sách thuốc + liều (đa dạng pattern liều lượng)
MED_POOL = [
    ("amlodipine", "5 mg", "1 viên/ngày"),
    ("aspirin", "81 mg", "1 viên/ngày"),
    ("pravastatin", "40 mg", "1 viên/ngày tối"),
    ("metformin", "500 mg", "2 viên/ngày"),
    ("senna", "8.6 mg", "2 viên/ngày"),
    ("atorvastatin", "20 mg", "1 viên/ngày tối"),
    ("bisoprolol", "5 mg", "1 viên/ngày"),
    ("omeprazole", "20 mg", "1 viên/ngày sáng"),
    ("furosemide", "40 mg", "1 viên/ngày"),
    ("losartan", "50 mg", "1 viên/ngày"),
    ("gliclazide", "80 mg", "2 viên/ngày"),
    ("insulin glargine", "10 IU", "tiêm dưới da buổi tối"),
    ("paracetamol", "500 mg", "khi cần, tối đa 4 viên/ngày"),
    ("levothyroxine", "100 mcg", "1 viên/ngày sáng"),
    ("warfarin", "5 mg", "1 viên/ngày, theo INR"),
]

TRIỆU_CHỨNG_POOL = [
    # Các cặp triệu chứng liền kề thường gặp trong EHR thật
    ("lo âu", "mất ngủ"),
    ("đau đầu", "chóng mặt"),
    ("buồn nôn", "nôn"),
    ("mệt mỏi", "chán ăn"),
    ("ho khan", "khó thở"),
    ("đau bụng", "tiêu chảy"),
    ("sốt nhẹ", "ớn lạnh"),
    ("đau ngực", "hồi hộp"),
    ("đau lưng", "tê bì chân"),
    ("khó nuốt", "đau họng"),
]

KẾT_QUẢ_XÉT_NGHIỆM_POOL = [
    ("Glucose", "7.2 mmol/L", "đói"),
    ("HbA1c", "7.8%", ""),
    ("Creatinin", "98 μmol/L", ""),
    ("eGFR", "72 mL/min/1.73m²", ""),
    ("ALT", "45 U/L", ""),
    ("AST", "38 U/L", ""),
    ("Cholesterol toàn phần", "5.8 mmol/L", ""),
    ("HDL-C", "1.1 mmol/L", ""),
    ("LDL-C", "3.6 mmol/L", ""),
    ("Triglycerid", "2.4 mmol/L", ""),
    ("TSH", "3.2 mIU/L", ""),
    ("FT4", "14.5 pmol/L", ""),
    ("CRP", "12 mg/L", ""),
    ("Hemoglobin", "128 g/L", ""),
    ("Bạch cầu", "9.2 G/L", ""),
    ("Tiểu cầu", "245 G/L", ""),
    ("INR", "1.2", ""),
    ("Natri", "138 mmol/L", ""),
    ("Kali", "4.1 mmol/L", ""),
]

GENDERS = ["nam", "nữ"]
AGES = ["45", "52", "63", "71", "38", "59", "67", "80", "34", "55"]


def generate_numbered_list(num_items=3):
    """Sinh chuỗi '1. thuốc_a liều_dùng. 2. thuốc_b...'"""
    selected = random.sample(MED_POOL, min(num_items, len(MED_POOL)))
    parts = []
    for i, (name, dose, freq) in enumerate(selected, 1):
        freq_str = f", {freq}" if freq else ""
        parts.append(f"{i}. {name} {dose}{freq_str}")
    return ". ".join(parts) + ".", selected


def generate_adjacent_symptoms():
    pattern = random.choice(TRIỆU_CHỨNG_POOL)
    templates = [
        f"Bệnh nhân than {pattern[0]} {pattern[1]} từ 1 tuần nay",
        f"Triệu chứng {pattern[0]} {pattern[1]} xuất hiện sau khi ngưng thuốc",
        f"BN {pattern[0]} {pattern[1]} ngày càng tăng",
        f"Bệnh nhân có biểu hiện {pattern[0]} {pattern[1]} kéo dài",
        f"Khám thấy {pattern[0]} {pattern[1]} không rõ nguyên nhân",
        f"Triệu chứng chính: {pattern[0]} {pattern[1]}",
    ]
    return random.choice(templates), list(pattern)


def generate_lab_sentence(num_results=3):
    selected = random.sample(KẾT_QUẢ_XÉT_NGHIỆM_POOL, min(num_results, len(KẾT_QUẢ_XÉT_NGHIỆM_POOL)))
    items = []
    for name, value, context in selected:
        ctx = f" ({context})" if context else ""
        items.append(f"{name}{ctx}: {value}")

    templates = [
        "Kết quả xét nghiệm: {results}.",
        "XN máu: {results}.",
        "Xét nghiệm sinh hóa: {results}.",
        "Lab: {results}.",
    ]
    text = random.choice(templates).format(results=", ".join(items))

    entities = []
    for name, value, context in selected:
        # Entity text phải khớp CHÍNH XÁC với những gì xuất hiện trong câu,
        # bao gồm cả context nếu có
        ctx = f" ({context})" if context else ""
        if ctx:
            # Format trong text: "Glucose (đói): 7.2 mmol/L"
            search_str = f"{name}{ctx}: {value}"
        else:
            # Format: "Creatinin: 98 μmol/L"
            search_str = f"{name}: {value}"

        if search_str in text:
            start = text.index(search_str)
            end = start + len(search_str)
        elif name in text:
            # Fallback: tên đứng riêng lẻ (không nên xảy ra với template hiện tại)
            search_str = name
            start = text.index(name)
            end = start + len(name)
        else:
            continue  # item này không xuất hiện trong text, bỏ qua

        entities.append({
            "text": search_str,
            "type": "KẾT_QUẢ_XÉT_NGHIỆM",
            "position": [start, end],
        })
        assert text[start:end] == search_str, f"Lab span mismatch: '{text[start:end]}' != '{search_str}'"

    return text, entities


def generate_patient_info_sentence():
    age = random.choice(AGES)
    gender = random.choice(GENDERS)
    templates = [
        f"BN {gender} {age} tuổi vào viện",
        f"Bệnh nhân {gender}, {age} tuổi, tiền sử THA",
        f"Tiếp nhận BN {gender} {age} tuổi từ tuyến dưới",
        f"Bệnh nhân {gender} {age} tuổi nhập viện ngày {random.randint(1,28)}/{random.randint(1,12)}/{random.randint(2024,2026)}",
    ]
    text = random.choice(templates)
    entities = []
    # Thêm cả "BN" hoặc "Bệnh nhân" + giới + tuổi vào 1 entity THÔNG_TIN_BỆNH_NHÂN
    # Convention: giống format EHR thật — BN nam 45 tuổi là 1 cụm thông tin
    patterns = [
        f"BN {gender} {age} tuổi",
        f"Bệnh nhân {gender}, {age} tuổi",
        f"BN {gender} {age} tuổi",
        f"Bệnh nhân {gender} {age} tuổi",
    ]
    for pat in patterns:
        if pat in text:
            start = text.index(pat)
            end = start + len(pat)
            entities.append({"text": pat, "type": "THÔNG_TIN_BỆNH_NHÂN", "position": [start, end]})
            assert text[start:end] == pat, f"Span mismatch: '{text[start:end]}' != '{pat}'"
            break
    else:
        # Fallback nếu không khớp pattern nào
        if gender in text:
            start = text.index(gender)
            entities.append({"text": gender, "type": "THÔNG_TIN_BỆNH_NHÂN", "position": [start, start + len(gender)]})
        if age in text:
            start = text.index(age)
            entities.append({"text": age, "type": "THÔNG_TIN_BỆNH_NHÂN", "position": [start, start + len(age)]})
    return text, entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/ner/synthetic_rare_patterns.jsonl")
    parser.add_argument("--n_numbered_list", type=int, default=40, help="Số câu format danh sách đánh số")
    parser.add_argument("--n_adjacent_symptoms", type=int, default=30, help="Số câu 2 triệu chứng liền kề")
    parser.add_argument("--n_lab", type=int, default=30, help="Số câu kết quả xét nghiệm")
    parser.add_argument("--n_patient_info", type=int, default=20, help="Số câu thông tin BN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    records = []

    # ── Bug 1: danh sách đánh số ──────────────────────────────────
    for _ in range(args.n_numbered_list):
        numbered, selected_meds = generate_numbered_list(random.randint(2, 5))
        template = random.choice(NUMBERED_LIST_TEMPLATES)
        text = template.format(numbered_list=numbered)
        entities = []
        for med_name, dose, freq in selected_meds:
            # Convention: entity THUỐC = tên + liều + ", tần_suất" (khớp ground truth)
            search_str = f"{med_name} {dose}, {freq}"
            if search_str in text:
                start = text.index(search_str)
                end = start + len(search_str)
            elif f"{med_name} {dose}" in text:
                search_str = f"{med_name} {dose}"
                start = text.index(search_str)
                end = start + len(search_str)
            else:
                search_str = med_name
                start = text.index(search_str)
                end = start + len(search_str)
            entities.append({
                "text": search_str,
                "type": "THUỐC",
                "position": [start, end],
            })
            # Assert: text span khớp đúng entity text
            assert text[start:end] == search_str, f"Span mismatch: '{text[start:end]}' != '{search_str}' in text: {text}"
        records.append({"text": text, "entities": entities, "source": "augment_numbered_list"})

    # ── Bug 2: 2 triệu chứng liền kề ──────────────────────────────
    for _ in range(args.n_adjacent_symptoms):
        text, (sym1, sym2) = generate_adjacent_symptoms()
        entities = []
        # Dùng tìm kiếm tuần tự từ trái sang phải để tránh trùng vị trí
        # khi sym1 là substring của sym2 hoặc ngược lại (vd "đau" vs "đau đầu")
        pos = 0
        for sym in [sym1, sym2]:
            idx = text.find(sym, pos)
            if idx >= 0:
                entities.append({"text": sym, "type": "TRIỆU_CHỨNG", "position": [idx, idx + len(sym)]})
                pos = idx + len(sym)  # tiếp tục tìm từ sau entity này
            else:
                raise ValueError(f"'{sym}' not found (searching from pos {pos}) in: {text}")
        records.append({"text": text, "entities": entities, "source": "augment_adjacent_symptoms"})

    # ── Nhãn hiếm: lab ────────────────────────────────────────────
    for _ in range(args.n_lab):
        text, entities = generate_lab_sentence(random.randint(2, 4))
        records.append({"text": text, "entities": entities, "source": "augment_lab"})

    # ── Nhãn hiếm: patient_info ────────────────────────────────────
    for _ in range(args.n_patient_info):
        text, entities = generate_patient_info_sentence()
        records.append({"text": text, "entities": entities, "source": "augment_patient_info"})

    # Shuffle để tránh order bias
    random.shuffle(records)

    # ── VALIDATE TOÀN BỘ: assert text[start:end] == entity["text"] ──────
    errors = []
    for i, rec in enumerate(records):
        for ent in rec["entities"]:
            span = rec["text"][ent["position"][0]:ent["position"][1]]
            if span != ent["text"]:
                errors.append(f"Record {i}: span '{span}' != entity.text '{ent['text']}' in text '{rec['text']}'")
    if errors:
        print(f"\n❌ VALIDATION FAILED — {len(errors)} span mismatches:")
        for e in errors[:20]:
            print(f"  {e}")
        raise SystemExit(1)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    label_counts = {}
    for rec in records:
        for ent in rec["entities"]:
            label_counts[ent["type"]] = label_counts.get(ent["type"], 0) + 1

    print(f"Generated {len(records)} sentences → {args.out}")
    print(f"Entity distribution: {label_counts}")
    print(f"  Bug 1 (numbered list): {args.n_numbered_list} câu")
    print(f"  Bug 2 (adjacent symptoms): {args.n_adjacent_symptoms} câu")
    print(f"  Rare label - lab: {args.n_lab} câu, patient_info: {args.n_patient_info} câu")
    print(f"✅ All span assertions passed")


if __name__ == "__main__":
    main()