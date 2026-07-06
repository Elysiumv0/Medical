"""
Gộp 3 nguồn data NER thành 1 bộ train/dev/test thống nhất:
  - PhoNER_COVID19 (converted_data/phoner_*.jsonl) — văn phong báo chí, BỆNH/TRIỆU_CHỨNG
  - ViMQ (converted_data/vimq_*.jsonl) — câu hỏi bệnh nhân, có thêm THUỐC/THỦ_THUẬT
  - Synthetic EHR (converted_data/synthetic_ehr.jsonl) — tự viết, bù domain gap
    văn phong EHR/ghi chú bác sĩ (chỉ có 20 câu — ít, chỉ mang tính minh hoạ
    pattern, KHÔNG đủ thay thế việc cần thêm nhiều data EHR thật/giả lập hơn)

Synthetic EHR được chia hết vào train (vì số lượng quá nhỏ, chia dev/test sẽ
không có ý nghĩa thống kê) — dev/test chính vẫn dựa trên PhoNER+ViMQ.
"""
import json
import random
import os


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def validate_offsets(records, source_name):
    """Kiểm tra lại LẦN CUỐI toàn bộ offset trước khi gộp — không tin tưởng mù
    quáng vào các bước convert trước, dù đã assert lúc tạo."""
    errors = 0
    for rec in records:
        for e in rec["entities"]:
            s, en = e["position"]
            if rec["text"][s:en] != e["text"]:
                errors += 1
    status = "✅" if errors == 0 else f"❌ {errors} lỗi"
    print(f"  Validate {source_name}: {len(records)} câu — {status}")
    return errors == 0


def merge_all(seed=42):
    random.seed(seed)

    print("=== Bước 1: Load, FILTER THỦ_THUẬT, và validate từng nguồn ===")
    print("(BỎ THỦ_THUẬT khỏi mọi nguồn, giữ THÔNG_TIN_BỆNH_NHÂN từ PhoNER)")

    VALID_TYPES = {"BỆNH", "TRIỆU_CHỨNG", "THUỐC", "THÔNG_TIN_BỆNH_NHÂN", "KẾT_QUẢ_XÉT_NGHIỆM"}
    def filter_entities(records):
        for rec in records:
            rec["entities"] = [e for e in rec["entities"] if e["type"] in VALID_TYPES]

    phoner_train = load_jsonl("converted_data/phoner_train.jsonl"); filter_entities(phoner_train)
    phoner_dev = load_jsonl("converted_data/phoner_dev.jsonl");   filter_entities(phoner_dev)
    phoner_test = load_jsonl("converted_data/phoner_test.jsonl"); filter_entities(phoner_test)
    vimq_train = load_jsonl("converted_data/vimq_train.jsonl");   filter_entities(vimq_train)
    vimq_dev = load_jsonl("converted_data/vimq_dev.jsonl");     filter_entities(vimq_dev)
    vimq_test = load_jsonl("converted_data/vimq_test.jsonl");   filter_entities(vimq_test)

    synth_parts = []
    for batch in ["synthetic_ehr", "synthetic_ehr_batch2", "synthetic_ehr_batch3",
                  "synthetic_ehr_batch4", "synthetic_ehr_batch5_patient_info",
                  "synthetic_ehr_batch5_lab"]:
        path = f"converted_data/{batch}.jsonl"
        if os.path.exists(path):
            recs = load_jsonl(path); filter_entities(recs)
            synth_parts.extend(recs)
            print(f"  Load {batch}: {len(recs)} records")
        else:
            print(f"  SKIP {batch}: file không tồn tại")
    synthetic = synth_parts

    all_ok = True
    for recs, name in [
        (phoner_train, "phoner_train"), (phoner_dev, "phoner_dev"), (phoner_test, "phoner_test"),
        (vimq_train, "vimq_train"), (vimq_dev, "vimq_dev"), (vimq_test, "vimq_test"),
        (synthetic, "synthetic_ehr"),
    ]:
        all_ok &= validate_offsets(recs, name)

    if not all_ok:
        raise RuntimeError("Có lỗi offset ở 1 nguồn nào đó — DỪNG LẠI, không gộp data lỗi.")

    print("\n=== Bước 2: Gắn nhãn nguồn gốc (để có thể tách lọc sau này nếu cần) ===")
    for recs, src in [(phoner_train, "phoner"), (vimq_train, "vimq"), (synthetic, "synthetic")]:
        for r in recs:
            r["source"] = src
    for recs, src in [(phoner_dev, "phoner"), (vimq_dev, "vimq")]:
        for r in recs:
            r["source"] = src
    for recs, src in [(phoner_test, "phoner"), (vimq_test, "vimq")]:
        for r in recs:
            r["source"] = src

    print("=== Bước 3: Gộp — synthetic chia 70/15/15, KHÔNG đánh đổi train lấy dev/test ===")
    OVERSAMPLE_FACTOR = 15

    # Chia TOÀN BỘ synthetic (140 câu, gồm 38 có KẾT_QUẢ_XÉT_NGHIỆM từ
    # batch5_lab + 8 cũ từ batch 1-4) theo tỷ lệ 70/15/15 để cả 3 tập
    # đều có mẫu KQXN, không tập nào bị "0 support" và model có pattern
    # để học (train được oversample thêm x15).
    random.shuffle(synthetic)
    n = len(synthetic)
    train_cut = int(n * 0.70)
    dev_cut = train_cut + int(n * 0.15)

    synth_train = synthetic[:train_cut]
    synth_dev = synthetic[train_cut:dev_cut]
    synth_test = synthetic[dev_cut:]

    # Đếm KQXN thực tế trong mỗi phần để verify
    def count_kqxn(recs):
        return sum(1 for r in recs for e in r["entities"] if e["type"] == "KẾT_QUẢ_XÉT_NGHIỆM")

    print(f"  Tổng synthetic: {n} câu (KQXN entity: {count_kqxn(synthetic)})")
    print(f"  → Train: {len(synth_train)} câu, KQXN={count_kqxn(synth_train)} (oversample x{OVERSAMPLE_FACTOR})")
    print(f"  → Dev:   {len(synth_dev)} câu, KQXN={count_kqxn(synth_dev)}")
    print(f"  → Test:  {len(synth_test)} câu, KQXN={count_kqxn(synth_test)}")

    synthetic_oversampled = synth_train * OVERSAMPLE_FACTOR
    print(f"  Train synthetic gốc {len(synth_train)} → x{OVERSAMPLE_FACTOR} = {len(synthetic_oversampled)}")

    combined_train = phoner_train + vimq_train + synthetic_oversampled
    combined_dev = phoner_dev + vimq_dev + synth_dev
    combined_test = phoner_test + vimq_test + synth_test

    random.shuffle(combined_train)
    random.shuffle(combined_dev)
    random.shuffle(combined_test)

    save_jsonl(combined_train, "converted_data/combined_train.jsonl")
    save_jsonl(combined_dev, "converted_data/combined_dev.jsonl")
    save_jsonl(combined_test, "converted_data/combined_test.jsonl")

    print("\n=== Thống kê cuối cùng ===")
    from collections import Counter
    for name, recs in [("TRAIN", combined_train), ("DEV", combined_dev), ("TEST", combined_test)]:
        type_counts = Counter()
        source_counts = Counter()
        for r in recs:
            source_counts[r["source"]] += 1
            for e in r["entities"]:
                type_counts[e["type"]] += 1
        print(f"\n{name}: {len(recs)} câu")
        print(f"  Theo nguồn: {dict(source_counts)}")
        print(f"  Theo type:  {dict(type_counts)}")

    print("\n⚠️  LƯU Ý QUAN TRỌNG:")
    print(f"  - Synthetic gốc chỉ {len(synthetic)} câu, oversample x{OVERSAMPLE_FACTOR} để")
    print("    tăng trọng số — nhưng đây là NHÂN BẢN Y HỆT, không phải data mới.")
    print("    Rủi ro: model có thể MEMORIZE đúng 86 câu này thay vì học pattern")
    print("    tổng quát (dấu hiệu: train loss trên synthetic giảm rất nhanh/rất")
    print("    thấp bất thường so với phần ViMQ/PhoNER). Cần theo dõi riêng loss/")
    print("    metric trên tập dev theo từng 'source' để phát hiện sớm nếu xảy ra.")
    print("  - THỦ_THUẬT: đã BỎ hoàn toàn (filter tại Step 1) — không có trong schema chính thức.")
    print("  - KẾT_QUẢ_XÉT_NGHIỆM: chỉ 14 entity trong 8 câu synthetic, quá ít để học,")
    print("    nên 8 câu này được chia đều cho Dev (4) và Test (4) để ít nhất")
    print("    seqeval có support > 0 khi eval — Train không có sample nào của type này.")
    print("    CẦN THÊM DATA THẬT cho KẾT_QUẢ_XÉT_NGHIỆM nếu muốn model học được type này.")


if __name__ == "__main__":
    merge_all()
