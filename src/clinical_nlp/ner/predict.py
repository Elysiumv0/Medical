import torch


try:
    from constants import ID2LABEL, LABEL2ID
except ImportError:
    from clinical_nlp.ner.constants import ID2LABEL, LABEL2ID


def _stripped(tokens):
    """Merge @@ continuation tokens and return list of (stripped_word, num_tokens_consumed)."""
    out = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.endswith("@@"):
            combined = tok[:-2]
            i += 1
            while i < len(tokens) and tokens[i].endswith("@@"):
                combined += tokens[i][:-2]
                i += 1
            out.append((combined, 1 + sum(1 for t in tokens[i-len(combined.replace('@@','')):i] if t.endswith('@@')) + (len(combined) - len(tok[:-2])) // 2))
            # Simplify: just record combined word, count = how many raw tokens merged
            continue
        out.append((tok.replace('@@', ''), 1))
        i += 1
    return out


def _merge_continuation_tokens(tokens):
    """Merge @@ continuation tokens into groups. Returns list of groups,
    each group is a list of (original_token_index, raw_token, stripped_char).
    """
    groups = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.endswith("@@"):
            # Start a merged group: collect all consecutive @@ tokens
            group = []
            while i < len(tokens) and tokens[i].endswith("@@"):
                group.append((i, tokens[i], tokens[i][:-2]))  # strip trailing @@
                i += 1
            groups.append(group)
            continue
        groups.append([(i, tok, tok.replace('@@', ''))])
        i += 1
    return groups


def _build_offset_mapping_slow(text, tokens):
    """
    Build char-level offsets for each token by reconstructing text from
    stripped tokens and matching against the original text (skipping spaces).

    Strategy:
    1. Merge @@ continuation tokens into groups (số@@ + t@@ → group of 2 tokens forming "sốt")
    2. Scan original text char-by-char, skipping whitespace
    3. Match each group's combined stripped string to the stripped text
    4. Distribute offsets within the group: first sub-token gets [start, start+len(sub1)],
       second gets [start+len(sub1), start+len(sub1)+len(sub2)], etc.
    """
    groups = _merge_continuation_tokens(tokens)

    # Build stripped version of original text (no spaces)
    stripped_text = ''.join(c for c in text if not c.isspace())
    stripped_lower = stripped_text.lower()

    offsets = [(0, 0)]  # [CLS]

    # Build stripped concatenation of all groups
    stripped_groups = [''.join(ch for _, _, ch in g) for g in groups]

    # Map each group to original text offsets by scanning stripped_text
    scan_pos = 0  # position in stripped_text
    for g_idx, group in enumerate(groups):
        target = stripped_groups[g_idx]
        if not target:
            for _ in group:
                offsets.append((0, 0))
            continue

        # Find target in stripped_text from scan_pos
        found = stripped_lower.find(target.lower(), scan_pos)
        if found < 0:
            for _ in group:
                offsets.append((0, 0))
            continue

        scan_pos = found + len(target)

        # Convert stripped position to original text position
        # Map: stripped_pos = count of non-space chars before that point in original text
        orig_start = _stripped_to_orig(text, found)
        orig_end = _stripped_to_orig(text, found + len(target))

        # Distribute sub-offsets within the group
        cum = orig_start
        for _, _, sub_stripped in group:
            sub_len = len(sub_stripped)
            offsets.append((cum, cum + sub_len))
            cum += sub_len

    offsets.append((0, 0))  # [SEP]
    return offsets


def _stripped_to_orig(text, stripped_pos):
    """Convert position in stripped (no-space) text to position in original text."""
    count = 0
    for i, c in enumerate(text):
        if not c.isspace():
            if count == stripped_pos:
                return i
            count += 1
    return len(text)  # past end


def predict_entities(text, model, tokenizer, id2label=None, max_length=256):
    """Predict entities from clinical text."""
    if id2label is None:
        id2label = ID2LABEL

    # Tokenize text
    raw_tokens = tokenizer.tokenize(text)[:max_length - 2]
    input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.convert_tokens_to_ids(raw_tokens)
        + [tokenizer.sep_token_id]
    )
    attention_mask = [1] * len(input_ids)

    # Build offset mapping
    offset_mapping = _build_offset_mapping_slow(text, raw_tokens)

    # Inference
    input_tensor = torch.tensor([input_ids])
    attention_tensor = torch.tensor([attention_mask])
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, attention_mask=attention_tensor)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()

    # Align lengths
    o_len = len(offset_mapping)
    p_len = len(predictions)
    if p_len < o_len:
        predictions.extend([LABEL2ID["O"]] * (o_len - p_len))
    predictions = predictions[:o_len]

    return _bio_to_entities(text, predictions, offset_mapping, id2label)


def _bio_to_entities(text, predictions, offset_mapping, id2label):
    """Convert BIO label sequence back to entity list."""
    entities = []
    current_words = []
    current_type = None

    for pred_id, (start, end) in zip(predictions, offset_mapping):
        if start == end == 0:
            # finalize entity even across special tokens
            if current_words:
                s0 = current_words[0][0]
                e0 = current_words[-1][1]
                entities.append({
                    "text": text[s0:e0],
                    "type": current_type,
                    "position": [s0, e0],
                })
                current_words = []
                current_type = None
            continue

        label = id2label[pred_id]

        if label.startswith("B-"):
            if current_words:
                s0 = current_words[0][0]
                e0 = current_words[-1][1]
                entities.append({
                    "text": text[s0:e0],
                    "type": current_type,
                    "position": [s0, e0],
                })
            current_words = [(start, end)]
            current_type = label[2:]
        elif label.startswith("I-"):
            if current_words:
                # Robust logic: if we are inside an entity, continue it
                # regardless of whether the label type matches.
                # This handles "model wobble" (e.g., B-X -> I-Y).
                current_words.append((start, end))
            else:
                # Orphan I- label with no preceding B- -> ignore
                pass
        else:
            if current_words:
                s0 = current_words[0][0]
                e0 = current_words[-1][1]
                entities.append({
                    "text": text[s0:e0],
                    "type": current_type,
                    "position": [s0, e0],
                })
            current_words = []
            current_type = None

    if current_words:
        s0 = current_words[0][0]
        e0 = current_words[-1][1]
        entities.append({
            "text": text[s0:e0],
            "type": current_type,
            "position": [s0, e0],
        })

    return entities