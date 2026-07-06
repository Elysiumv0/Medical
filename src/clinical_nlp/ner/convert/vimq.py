"""
Convert ViMQ (data/{train,dev,test}.json) sang format training cùng chuẩn với
convert_phoner.py.

Điểm khác biệt quan trọng so với PhoNER:
1. ViMQ dùng OFFSET THEO TỪ (word index, word-segment kiểu VnCoreNLP với "_"
   nối từ ghép), không phải offset ký tự — seq_label = [start_word, end_word
   (inclusive), type].
2. Có 3 loại nhãn: SYMPTOM_AND_DISEASE (cần tách BỆNH/TRIỆU_CHỨNG bằng heuristic
   giống convert_phoner.py), medical_procedure (map sang KẾT_QUẢ_XÉT_NGHIỆM/thủ
   thuật — schema đề bài chưa xác nhận rõ có type này không, tạm giữ riêng để
   quyết định sau), drug (map thẳng THUỐC — không cần heuristic, tin cậy cao).

Cách xử lý offset: vì "_" thay thế đúng 1 dấu cách trong từ ghép (không đổi độ
dài chuỗi), có thể tính char offset trên chuỗi gốc (có "_") rồi thay "_" -> " "
ở bước cuối cùng — độ dài không đổi nên offset vẫn đúng sau khi thay.
"""
import json
from convert_phoner import is_disease_not_symptom  # tái dùng heuristic đã có


def word_spans_to_char_spans(sentence: str):
    """Trả về list (char_start, char_end) cho từng 'từ' (token cách nhau bởi
    khoảng trắng, có thể chứa '_' bên trong cho từ ghép)."""
    spans = []
    pos = 0
    for word in sentence.split(" "):
        start = pos
        end = pos + len(word)
        spans.append((start, end))
        pos = end + 1  # +1 cho dấu cách phân tách
    return spans


def convert_vimq_to_target_schema(json_path: str, output_path: str):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Đọc được {len(data)} câu từ {json_path}")

    output_records = []
    skipped_no_entity = 0
    type_counts = {"BỆNH": 0, "TRIỆU_CHỨNG": 0, "THUỐC": 0, "THỦ_THUẬT": 0}

    for rec in data:
        sentence = rec["sentence"]
        word_spans = word_spans_to_char_spans(sentence)

        relevant_entities = []
        for start_w, end_w, tag in rec["seq_label"]:
            if end_w >= len(word_spans):
                continue  # bảo vệ khỏi lỗi index nếu có (không nên xảy ra, nhưng an toàn)

            char_start = word_spans[start_w][0]
            char_end = word_spans[end_w][1]
            entity_text_raw = sentence[char_start:char_end]  # còn chứa "_"

            # Verify offset đúng TRÊN CHUỖI GỐC (có "_") trước khi biến đổi
            assert sentence[char_start:char_end] == entity_text_raw

            if tag == "drug":
                mapped_type = "THUỐC"
            elif tag == "medical_procedure":
                mapped_type = "THỦ_THUẬT"
            elif tag == "SYMPTOM_AND_DISEASE":
                mapped_type = "BỆNH" if is_disease_not_symptom(entity_text_raw.replace("_", " ")) else "TRIỆU_CHỨNG"
            else:
                continue

            type_counts[mapped_type] += 1
            relevant_entities.append({
                "char_start": char_start, "char_end": char_end,
                "type": mapped_type, "text_raw": entity_text_raw,
            })

        if not relevant_entities:
            skipped_no_entity += 1
            continue

        # Thay "_" -> " " ở CẢ câu lẫn entity text — độ dài không đổi (1-1),
        # nên char_start/char_end tính ở trên VẪN ĐÚNG trên chuỗi đã thay.
        final_text = sentence.replace("_", " ")

        final_entities = []
        for e in relevant_entities:
            s, en = e["char_start"], e["char_end"]
            final_entity_text = final_text[s:en]
            # Sanity check bắt buộc: offset trên chuỗi đã thay "_" phải khớp
            assert final_entity_text == e["text_raw"].replace("_", " "), (
                f"Offset lệch sau khi thay underscore! "
                f"'{final_entity_text}' != '{e['text_raw'].replace('_', ' ')}'"
            )
            final_entities.append({
                "text": final_entity_text, "type": e["type"], "position": [s, en]
            })

        output_records.append({"text": final_text, "entities": final_entities})

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Giữ lại {len(output_records)} câu có entity liên quan "
          f"(bỏ {skipped_no_entity} câu không liên quan)")
    print(f"  -> {type_counts}")
    print(f"Đã lưu: {output_path}")
    return output_records


if __name__ == "__main__":
    import os
    os.makedirs("converted_data", exist_ok=True)

    for split in ["train", "dev", "test"]:
        convert_vimq_to_target_schema(
            f"ViMQ/data/{split}.json",
            f"converted_data/vimq_{split}.jsonl",
        )
