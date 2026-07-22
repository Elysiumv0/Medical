import torch


try:
    from constants import ID2LABEL, LABEL2ID
except ImportError:
    from clinical_nlp.ner.constants import ID2LABEL, LABEL2ID


def _stripped(tokens):
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
            continue
        out.append((tok.replace('@@', ''), 1))
        i += 1
    return out


def _merge_continuation_tokens(tokens):
    groups = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.endswith("@@"):
            group = []
            while i < len(tokens) and tokens[i].endswith("@@"):
                group.append((i, tokens[i], tokens[i][:-2]))
                i += 1
            groups.append(group)
            continue
        groups.append([(i, tok, tok.replace('@@', ''))])
        i += 1
    return groups

def _build_offset_mapping_slow(text, tokens):
    groups = _merge_continuation_tokens(tokens)
    stripped_text = ''.join(c for c in text if not c.isspace())
    stripped_lower = stripped_text.lower()
    offsets = [(0, 0)]
    stripped_groups = [''.join(ch for _, _, ch in g) for g in groups]
    scan_pos = 0
    for g_idx, group in enumerate(groups):
        target = stripped_groups[g_idx]
        if not target:
            for _ in group:
                offsets.append((0, 0))
            continue
        found = stripped_lower.find(target.lower(), scan_pos)
        if found < 0:
            for _ in group:
                offsets.append((0, 0))
            continue
        scan_pos = found + len(target)
        orig_start = _stripped_to_orig(text, found)
        orig_end = _stripped_to_orig(text, found + len(target))
        cum = orig_start
        for _, _, sub_stripped in group:
            sub_len = len(sub_stripped)
            offsets.append((cum, cum + sub_len))
            cum += sub_len
    offsets.append((0, 0)) 
    return offsets

def _stripped_to_orig(text, stripped_pos):
    count = 0
    for i, c in enumerate(text):
        if not c.isspace():
            if count == stripped_pos:
                return i
            count += 1
    return len(text)

def predict_entities(text, model, tokenizer, id2label=None, max_length=256):
    if id2label is None:
        id2label = ID2LABEL
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
