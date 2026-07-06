"""
Verify hàm char_entities_to_bio_labels() — logic quan trọng nhất của
train_ner.py — bằng PHOBERT TOKENIZER THẬT. Hàm mới dùng manual word
segmentation + per-word encode để map entity → subword BIO tags,
vì PhoBERT Python tokenizer không hỗ trợ offset_mapping / word_ids().
"""
from train_ner import char_entities_to_bio_labels, LABEL2ID, ID2LABEL
from transformers import AutoTokenizer


def decode_labels(labels, input_ids, tokenizer):
    """In ra để mắt kiểm tra: subword nào được gán nhãn gì."""
    for i, (label_id, token_id) in enumerate(zip(labels, input_ids)):
        label = ID2LABEL[label_id]
        if label != "O":
            token_text = tokenizer.decode([token_id])
            print(f"  sub[{i}] '{token_text}' -> {label}")


def test_simple_case():
    print("=== Test 1: Case đơn giản, entity ngắn ===")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    text = "Bệnh nhân bị sốt và ho"
    entities = [
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [12, 15]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [19, 21]},
    ]
    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=20)
    print(f"  input_ids ({len(enc['input_ids'])}): {enc['input_ids']}")
    print(f"  labels   ({len(enc['labels'])}):  {enc['labels']}")
    decode_labels(enc["labels"], enc["input_ids"], tokenizer)

    non_o_labels = [ID2LABEL[l] for l in enc["labels"] if l != LABEL2ID["O"]]
    assert any(l.startswith("B-TRIỆU_CHỨNG") for l in non_o_labels), "Thiếu B- label!"
    assert any(l.startswith("B-TRIỆU_CHỨNG") for l in non_o_labels if not l.startswith("I-")), "OK"
    print("  ✅ PASS\n")


def test_multiword_entity():
    print("=== Test 2: Entity nhiều từ ===")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    text = "Chẩn đoán rung nhĩ đáp ứng thất nhanh hôm nay"
    # Entity: "rung nhĩ đáp ứng thất nhanh"
    entities = [{"text": "rung nhĩ đáp ứng thất nhanh", "type": "BỆNH", "position": [10, 37]}]

    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=30)
    decode_labels(enc["labels"], enc["input_ids"], tokenizer)

    labels_str = [ID2LABEL[l] for l in enc["labels"]]
    b_count = sum(1 for l in labels_str if l == "B-BỆNH")
    i_count = sum(1 for l in labels_str if l == "I-BỆNH")
    print(f"  Số B-BỆNH: {b_count} (phải đúng 1), số I-BỆNH: {i_count} (phải >=1)")
    assert b_count == 1, f"Phải có ĐÚNG 1 B- cho 1 entity, có {b_count}"
    assert i_count >= 1, "Entity nhiều từ phải có ít nhất 1 I- label"
    print("  ✅ PASS\n")


def test_medication_entity():
    print("=== Test 3: Entity thuốc (BPE split điển hình) ===")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    text = "Dùng thuốc metoprolol hôm nay"
    entities = [{"text": "metoprolol", "type": "THUỐC", "position": [11, 21]}]

    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=20)
    decode_labels(enc["labels"], enc["input_ids"], tokenizer)

    labels_str = [ID2LABEL[l] for l in enc["labels"]]
    b_count = sum(1 for l in labels_str if l == "B-THUỐC")
    i_count = sum(1 for l in labels_str if l == "I-THUỐC")
    print(f"  Số B-THUỐC: {b_count} (phải 1), số I-THUỐC: {i_count} (phải >=0)")
    assert b_count == 1, f"Phải có B-THUỐC, có {b_count}"
    print("  ✅ PASS\n")


def test_no_entity():
    print("=== Test 4: Câu không có entity ===")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    text = "Không có gì đặc biệt"
    entities = []
    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=10)
    labels_str = [ID2LABEL[l] for l in enc["labels"]]
    assert all(l == "O" for l in labels_str), "Tất cả phải là O"
    print(f"  labels: {labels_str}")
    print("  ✅ PASS\n")


if __name__ == "__main__":
    print("Tải PhoBERT tokenizer (cần internet lần đầu)...\n")
    test_simple_case()
    test_multiword_entity()
    test_medication_entity()
    test_no_entity()
    print("=== TẤT CẢ TESTS PASSED ===")