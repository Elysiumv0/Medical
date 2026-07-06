"""
Convert PhoNER_COVID19 (định dạng CoNLL, mức syllable) sang format training
cho bài toán clinical NER của bạn.

Quyết định thiết kế:
1. Dùng bản SYLLABLE (không phải WORD) — vì syllable gần với chuỗi ký tự gốc
   hơn nhiều (không có dấu "_" nối từ ghép), nên tái tạo char offset chính xác
   hơn và ít rủi ro lệch offset khi ghép lại thành câu.
2. CHỈ giữ nhãn SYMPTOM_AND_DISEASE (map thành 2 type: TRIỆU_CHỨNG hoặc BỆNH —
   PhoNER không phân biệt 2 loại này, cần xử lý riêng, xem ghi chú bên dưới).
   NAY GIỮ THÊM AGE, GENDER, JOB — map thành THÔNG_TIN_BỆNH_NHÂN (đề bài chính
   thức có type này, không phải THỦ_THUẬT như giả định cũ).
   KHÔNG lấy NAME, LOCATION, PATIENT_ID — 3 loại này là PII định danh cá nhân,
   khác bản chất với "thông tin bệnh nhân" kiểu tuổi/giới/nghề, và hầu như
   không xuất hiện trong EHR đã ẩn danh — rủi ro nhiễu cao hơn lợi ích.
3. Bỏ hoàn toàn các câu chỉ toàn nhãn không liên quan (DATE, NAME, LOCATION...)
   để tránh dạy model học "mọi câu đều toàn O" một cách lệch lạc.

LƯU Ý QUAN TRỌNG chưa giải quyết được ở bước này:
PhoNER gộp chung TRIỆU_CHỨNG và BỆNH vào 1 nhãn SYMPTOM_AND_DISEASE, trong khi
schema của bạn tách riêng 2 loại. Không thể tự động phân tách chính xác 100%
chỉ từ nhãn có sẵn — cần 1 trong 2 hướng:
  (a) Gán tạm toàn bộ thành TRIỆU_CHỨNG (an toàn hơn, vì đa số case COVID là
      triệu chứng, ít bệnh nền), rồi review tay sau
  (b) Dùng heuristic từ điển nhỏ (danh sách tên bệnh mạn tính phổ biến) để
      tách case rõ ràng là BỆNH (như "suy thận mạn", "tiểu đường"...)
Code dưới đây làm (b) với 1 danh sách khởi điểm nhỏ, CẦN BẠN REVIEW LẠI.
"""
import json
import re

# Danh sách khởi điểm để phân biệt BỆNH (mạn tính, chẩn đoán) vs TRIỆU_CHỨNG
# (biểu hiện tạm thời) — dựa trên kiến thức y khoa phổ thông, CẦN REVIEW.
DISEASE_KEYWORDS = [
    "suy thận", "tiểu đường", "đái tháo đường", "cao huyết áp", "tăng huyết áp",
    "suy tim", "ung thư", "viêm gan", "xơ gan", "hen phế quản", "hen suyễn",
    "copd", "phổi tắc nghẽn", "suy gan", "viêm phổi", "lao phổi", "gút",
    "rung nhĩ", "nhồi máu cơ tim", "đột quỵ", "tai biến", "parkinson",
    "động kinh", "trầm cảm", "tâm thần phân liệt",
    # Bổ sung sau khi audit thực tế trên PhoNER_COVID19 (không phải từ 100 file
    # test cuộc thi — đây là data nguồn ngoài, hợp lệ để hoàn thiện heuristic):
    "thoái hoá", "thoái hóa", "thiếu máu cục bộ", "thiếu máu",
    "rối loạn tiền đình", "bệnh tim thiếu máu cục bộ",
]


def is_disease_not_symptom(entity_text: str) -> bool:
    text_lower = entity_text.lower()
    return any(kw in text_lower for kw in DISEASE_KEYWORDS)


def parse_conll_syllable(filepath: str):
    """Đọc file CoNLL, trả về list câu, mỗi câu là list (token, tag)."""
    sentences = []
    current = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token, tag = parts[0], parts[-1]
                    current.append((token, tag))
    if current:
        sentences.append(current)
    return sentences


def reconstruct_text_and_spans(tokens_tags):
    """
    Ghép lại câu từ list token (mức syllable) thành 1 chuỗi text liên tục,
    đồng thời tính char offset chính xác cho từng entity.

    QUAN TRỌNG: PhoNER (giống hầu hết corpus NER) đã tokenize sẵn, nghĩa là
    khoảng trắng gốc giữa các từ đã bị chuẩn hoá về đúng 1 dấu cách — không
    còn giữ multiple-space hay không-có-space gốc của văn bản thô ban đầu.
    Điều này CHẤP NHẬN ĐƯỢC cho mục đích tạo training data (model học pattern
    ngôn ngữ, không cần offset khớp 1 văn bản cụ thể nào), nhưng KHÔNG dùng
    cách ghép này cho dữ liệu thật cần nộp bài (100 file test) — ở đó phải
    lấy offset trực tiếp từ văn bản gốc, không tái tạo qua tokenized form.
    """
    text_parts = []
    entities = []
    current_pos = 0
    current_entity_start = None
    current_entity_tag = None
    current_entity_tokens = []

    for i, (token, tag) in enumerate(tokens_tags):
        if text_parts:
            text_parts.append(" ")
            current_pos += 1

        token_start = current_pos
        text_parts.append(token)
        current_pos += len(token)

        if tag.startswith("B-"):
            if current_entity_start is not None:
                entities.append((current_entity_start, current_pos - len(token) - 1,
                                  current_entity_tag, " ".join(current_entity_tokens)))
            current_entity_start = token_start
            current_entity_tag = tag[2:]
            current_entity_tokens = [token]
        elif tag.startswith("I-") and current_entity_start is not None:
            current_entity_tokens.append(token)
        else:  # O
            if current_entity_start is not None:
                entities.append((current_entity_start, current_pos - len(token) - 1,
                                  current_entity_tag, " ".join(current_entity_tokens)))
                current_entity_start = None
                current_entity_tokens = []

    if current_entity_start is not None:
        entities.append((current_entity_start, current_pos, current_entity_tag,
                          " ".join(current_entity_tokens)))

    full_text = "".join(text_parts)
    return full_text, entities


def convert_phoner_to_target_schema(conll_path: str, output_path: str):
    sentences = parse_conll_syllable(conll_path)
    print(f"Đọc được {len(sentences)} câu từ {conll_path}")

    output_records = []
    skipped_no_entity = 0
    disease_count, symptom_count, patient_info_count = 0, 0, 0

    for tokens_tags in sentences:
        full_text, entities = reconstruct_text_and_spans(tokens_tags)

        relevant_entities = []
        for start, end, tag, entity_text in entities:
            # Verify offset đúng — sanity check bắt buộc
            assert full_text[start:end] == entity_text, (
                f"Offset lệch! '{full_text[start:end]}' != '{entity_text}'"
            )

            if tag == "SYMPTOM_AND_DISEASE":
                mapped_type = "BỆNH" if is_disease_not_symptom(entity_text) else "TRIỆU_CHỨNG"
                if mapped_type == "BỆNH":
                    disease_count += 1
                else:
                    symptom_count += 1
            elif tag in ("AGE", "GENDER", "JOB"):
                mapped_type = "THÔNG_TIN_BỆNH_NHÂN"
                patient_info_count += 1
            else:
                continue  # bỏ DATE, NAME, LOCATION, PATIENT_ID, ...

            relevant_entities.append({
                "text": entity_text, "type": mapped_type, "position": [start, end]
            })

        if not relevant_entities:
            skipped_no_entity += 1
            continue  # bỏ câu không có entity liên quan — tránh dạy toàn "O"

        output_records.append({"text": full_text, "entities": relevant_entities})

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Giữ lại {len(output_records)} câu có entity liên quan "
          f"(bỏ {skipped_no_entity} câu không liên quan)")
    print(f"  -> BỆNH: {disease_count}, TRIỆU_CHỨNG: {symptom_count}, "
          f"THÔNG_TIN_BỆNH_NHÂN: {patient_info_count}")
    print(f"  (Lưu ý: phân loại BỆNH/TRIỆU_CHỨNG dựa trên keyword heuristic, CẦN REVIEW TAY)")
    print(f"Đã lưu: {output_path}")
    return output_records


if __name__ == "__main__":
    import os
    os.makedirs("converted_data", exist_ok=True)

    for split in ["train", "dev", "test"]:
        convert_phoner_to_target_schema(
            f"PhoNER_COVID19/data/syllable/{split}_syllable.conll",
            f"converted_data/phoner_{split}.jsonl",
        )
