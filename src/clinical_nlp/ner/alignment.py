try:
    from clinical_nlp.ner.constants import LABEL2ID
except ImportError:
    from constants import LABEL2ID  # fallback khi chạy phẳng trên Colab


def _stripped_text(text):
    """Remove all whitespace from text, preserving only non-space chars."""
    return ''.join(c for c in text if not c.isspace())


def _stripped_to_orig(text, stripped_pos):
    """Convert position in stripped (space-free) text to position in original text."""
    count = 0
    for i, c in enumerate(text):
        if not c.isspace():
            if count == stripped_pos:
                return i
            count += 1
    return len(text)


def _build_slow_offset_mapping(text, tokens, max_length):
    """
    Build char-level offset mapping for a slow tokenizer's token list.

    Handles PhoBERT @@ continuation markers: groups consecutive @@ tokens
    into one merged word, then maps each sub-token to its position in the
    original text by matching against a space-stripped version of the text.

    Returns list of (start, end) tuples aligned with tokens (including
    [CLS] at index 0 and [SEP] at the end).
    """
    # 1) Merge consecutive @@ tokens into groups
    groups = []  # each group: list of (raw_token, stripped_chars)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.endswith("@@"):
            group = []
            while i < len(tokens) and tokens[i].endswith("@@"):
                group.append((tokens[i], tokens[i][:-2]))  # strip trailing @@
                i += 1
            groups.append(group)
            continue
        # Strip any other BPE markers (▁ for SentencePiece, ## for BERT)
        clean = tok.replace("▁", "").replace("##", "").replace("@@", "")
        groups.append([(tok, clean)])
        i += 1

    # 2) Build stripped text and find each group in it
    text_stripped = _stripped_text(text)
    text_stripped_lower = text_stripped.lower()

    offset_mapping = [(0, 0)]  # [CLS]
    scan_pos = 0

    for group in groups:
        combined_stripped = ''.join(ch for _, ch in group)
        if not combined_stripped:
            for _ in group:
                offset_mapping.append((0, 0))
            continue

        found = text_stripped_lower.find(combined_stripped.lower(), scan_pos)
        if found < 0:
            # Fallback: try each sub-token individually
            for _, ch in group:
                f = text_stripped_lower.find(ch.lower(), scan_pos)
                if f >= 0:
                    orig_start = _stripped_to_orig(text, f)
                    orig_end = _stripped_to_orig(text, f + len(ch))
                    offset_mapping.append((orig_start, orig_end))
                    scan_pos = f + len(ch)
                else:
                    offset_mapping.append((0, 0))
            continue

        scan_pos = found + len(combined_stripped)

        # Distribute offsets across sub-tokens in the group
        cum = _stripped_to_orig(text, found)
        for _, ch in group:
            offset_mapping.append((cum, cum + len(ch)))
            cum += len(ch)

    offset_mapping.append((0, 0))  # [SEP]

    # Pad to max_length
    while len(offset_mapping) < max_length:
        offset_mapping.append((0, 0))
    return offset_mapping[:max_length]


def char_entities_to_bio_labels(text: str, entities: list, tokenizer, max_length=256):
    """
    Map char-level entity spans → BIO token labels.

    Dùng tokenizer FAST (return_offsets_mapping=True) nếu có.
    Fallback sang slow path — merge BPE tokens và align qua stripped text
    (bỏ qua whitespace khi match ký tự).
    """
    # ── Thử fast path (BPE subword tokenizer có Rust backend) ──
    try:
        encoding = tokenizer(
            text, truncation=True, max_length=max_length,
            return_offsets_mapping=True, padding="max_length",
        )
        offset_mapping = encoding["offset_mapping"]
    except (KeyError, TypeError):
        # ── Slow path: tokenize + build offset thủ công ──
        raw_tokens = tokenizer.tokenize(text)
        raw_tokens = raw_tokens[:max_length - 2]  # chừa slot [CLS] + [SEP]

        input_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(raw_tokens) + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        pad_len = max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len

        offset_mapping = _build_slow_offset_mapping(text, raw_tokens, max_length)

        encoding = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,
        }

    labels = [LABEL2ID["O"]] * len(offset_mapping)

    for entity in entities:
        e_start, e_end = entity["position"]
        e_type = entity["type"]
        if e_type not in {"THUỐC", "BỆNH", "TRIỆU_CHỨNG", "THÔNG_TIN_BỆNH_NHÂN", "KẾT_QUẢ_XÉT_NGHIỆM"}:
            continue

        first_token_in_entity = True
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start == tok_end == 0:
                continue

            # Token overlaps with entity span
            if tok_start < e_end and tok_end > e_start:
                if first_token_in_entity:
                    labels[i] = LABEL2ID[f"B-{e_type}"]
                    first_token_in_entity = False
                else:
                    labels[i] = LABEL2ID[f"I-{e_type}"]

    encoding["labels"] = labels
    encoding.pop("offset_mapping", None)
    return encoding