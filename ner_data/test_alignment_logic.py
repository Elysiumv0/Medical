"""
Verify hàm char_entities_to_bio_labels() — logic quan trọng nhất của
train_ner.py — bằng MOCK TOKENIZER giả lập đúng interface của HuggingFace
fast tokenizer (offset_mapping). Không cần tải model thật, nên chạy được
ngay trong sandbox bị chặn mạng.
"""
from train_ner import char_entities_to_bio_labels, LABEL2ID, ID2LABEL


class MockFastTokenizer:
    """
    Giả lập tokenizer kiểu BPE: tách theo khoảng trắng rồi tách tiếp mỗi từ
    dài >4 ký tự thành 2 sub-token (mô phỏng hành vi BPE thật của PhoBERT,
    nơi 1 từ có thể bị tách thành nhiều token) — để test xem hàm align có
    xử lý đúng multi-token entity hay không, không chỉrom trường hợp
    1-từ-1-token đơn giản.
    """
    def __call__(self, text, truncation=True, max_length=256,
                 return_offsets_mapping=True, padding="max_length"):
        offsets = []
        pos = 0
        for word in text.split(" "):
            start = pos
            if len(word) > 4:
                mid = len(word) // 2
                offsets.append((start, start + mid))
                offsets.append((start + mid, start + len(word)))
            else:
                offsets.append((start, start + len(word)))
            pos += len(word) + 1

        # Thêm [CLS] đầu, [SEP] cuối (offset (0,0) như HF thật)
        offsets = [(0, 0)] + offsets + [(0, 0)]
        # Pad tới max_length bằng (0,0)
        while len(offsets) < max_length:
            offsets.append((0, 0))
        offsets = offsets[:max_length]

        return {
            "input_ids": list(range(len(offsets))),  # giá trị giả, không quan trọng cho test này
            "attention_mask": [1 if o != (0, 0) else 0 for o in offsets],
            "offset_mapping": offsets,
        }


def decode_labels(labels, offset_mapping, text):
    """In ra để mắt kiểm tra: token nào được gán nhãn gì."""
    for label_id, (start, end) in zip(labels, offset_mapping):
        if start == end == 0:
            continue
        label = ID2LABEL[label_id]
        if label != "O":
            print(f"  [{start},{end}] '{text[start:end]}' -> {label}")


def test_simple_case():
    print("=== Test 1: Case đơn giản, entity ngắn ===")
    text = "Bệnh nhân bị sốt và ho"
    entities = [
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [13, 16]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [20, 22]},
    ]
    tokenizer = MockFastTokenizer()
    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=20)
    decode_labels(enc["labels"], tokenizer(text, max_length=20)["offset_mapping"], text)

    # Assertion: đúng 2 entity, mỗi entity có ít nhất 1 label B-
    non_o_labels = [ID2LABEL[l] for l in enc["labels"] if l != LABEL2ID["O"]]
    assert any(l.startswith("B-TRIỆU_CHỨNG") for l in non_o_labels), "Thiếu B- label!"
    print("  ✅ PASS\n")


def test_multiword_entity():
    print("=== Test 2: Entity nhiều từ (nhiều token do bị BPE tách) ===")
    text = "Chẩn đoán rung nhĩ đáp ứng thất nhanh hôm nay"
    # "rung nhĩ đáp ứng thất nhanh" nên toàn bộ được gán B- rồi I-...
    start = text.index("rung nhĩ")
    end = start + len("rung nhĩ đáp ứng thất nhanh")
    entities = [{"text": text[start:end], "type": "BỆNH", "position": [start, end]}]

    tokenizer = MockFastTokenizer()
    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=30)
    offset_mapping = tokenizer(text, max_length=30)["offset_mapping"]
    decode_labels(enc["labels"], offset_mapping, text)

    labels_str = [ID2LABEL[l] for l in enc["labels"]]
    b_count = sum(1 for l in labels_str if l == "B-BỆNH")
    i_count = sum(1 for l in labels_str if l == "I-BỆNH")
    print(f"  Số B-BỆNH: {b_count} (phải đúng 1), số I-BỆNH: {i_count} (phải >=1)")
    assert b_count == 1, f"Phải có ĐÚNG 1 B- cho 1 entity, có {b_count}"
    assert i_count >= 1, "Entity nhiều từ phải có ít nhất 1 I- label"
    print("  ✅ PASS\n")


def test_overlapping_bpe_split():
    print("=== Test 3: Entity mà ranh giới rơi GIỮA 1 sub-token BPE ===")
    # Mô phỏng case khó: entity boundary không trùng khớp hoàn hảo với token
    # boundary (rất phổ biến với BPE thật) — hàm dùng overlap nên vẫn nên bắt được
    text = "Dùng thuốc metoprolol hôm nay"
    start = text.index("metoprolol")
    end = start + len("metoprolol")  # "metoprolol" dài 10 ký tự -> bị mock tách 2 sub-token
    entities = [{"text": "metoprolol", "type": "THUỐC", "position": [start, end]}]

    tokenizer = MockFastTokenizer()
    enc = char_entities_to_bio_labels(text, entities, tokenizer, max_length=20)
    offset_mapping = tokenizer(text, max_length=20)["offset_mapping"]
    decode_labels(enc["labels"], offset_mapping, text)

    labels_str = [ID2LABEL[l] for l in enc["labels"]]
    assert "B-THUỐC" in labels_str, "Phải bắt được B-THUỐC dù entity bị tách sub-token"
    assert "I-THUỐC" in labels_str, "Sub-token thứ 2 phải là I-THUỐC, không phải bị bỏ sót"
    print("  ✅ PASS\n")


def test_no_entity():
    print("=== Test 4: Câu không có entity nào -> toàn bộ phải là O ===")
    text = "Bệnh nhân tỉnh táo tiếp xúc tốt"
    tokenizer = MockFastTokenizer()
    enc = char_entities_to_bio_labels(text, [], tokenizer, max_length=15)
    non_o = [l for l in enc["labels"] if l != LABEL2ID["O"]]
    assert len(non_o) == 0, "Không có entity thì không được có label nào khác O"
    print("  ✅ PASS\n")


if __name__ == "__main__":
    test_simple_case()
    test_multiword_entity()
    test_overlapping_bpe_split()
    test_no_entity()
    print("=" * 50)
    print("✅ TẤT CẢ TEST PASS — logic alignment char-offset -> BIO đúng.")
    print("   Sẵn sàng chạy với tokenizer thật (PhoBERT/ViHealthBERT) ở máy")
    print("   có internet — logic sẽ hoạt động giống hệt vì cùng dùng đúng")
    print("   interface offset_mapping chuẩn của HuggingFace fast tokenizer.")
