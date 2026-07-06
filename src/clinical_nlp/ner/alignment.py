from clinical_nlp.ner.constants import LABEL2ID


def char_entities_to_bio_labels(text: str, entities: list, tokenizer, max_length=256):
    """
    ĐÂY LÀ HÀM QUAN TRỌNG NHẤT — xử lý đúng cạm bẫy offset đã cảnh báo từ đầu.

    Dùng tokenizer FAST (return_offsets_mapping=True) để lấy char-span của
    từng token, rồi map ngược sang nhãn BIO dựa trên overlap giữa token-span
    và entity-span theo CHAR OFFSET — không dựa vào việc đếm từ/tách theo
    khoảng trắng (cách đó sẽ sai với BPE subword tokenization).

    Trả về: input_ids, attention_mask, labels (list số nguyên theo LABEL2ID)
    """
    encoding = tokenizer(
        text, truncation=True, max_length=max_length,
        return_offsets_mapping=True, padding="max_length",
    )
    offset_mapping = encoding["offset_mapping"]
    labels = [LABEL2ID["O"]] * len(offset_mapping)

    for entity in entities:
        e_start, e_end = entity["position"]
        e_type = entity["type"]
        if e_type not in {"THUỐC", "BỆNH", "TRIỆU_CHỨNG", "THÔNG_TIN_BỆNH_NHÂN", "KẾT_QUẢ_XÉT_NGHIỆM"}:
            continue  # bỏ qua type lạ không có trong LABEL_LIST, tránh crash

        first_token_in_entity = True
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end == 0:
                continue  # special token ([CLS], [SEP], padding) — offset (0,0)

            # Token được coi là thuộc entity nếu có overlap với entity span
            if tok_start < e_end and tok_end > e_start:
                if first_token_in_entity:
                    labels[i] = LABEL2ID[f"B-{e_type}"]
                    first_token_in_entity = False
                else:
                    labels[i] = LABEL2ID[f"I-{e_type}"]

    encoding["labels"] = labels
    encoding.pop("offset_mapping")  # không cần giữ lại cho training, chỉ cần cho bước này
    return encoding