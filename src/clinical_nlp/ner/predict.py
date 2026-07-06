import torch
import re
from clinical_nlp.ner.constants import ID2LABEL


def predict_entities(text, model, tokenizer, id2label=ID2LABEL):
    ENTITY_BREAK_TOKENS = re.compile(r'^\d+\.(?!\d)$|^[;:]$')
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, return_offsets_mapping=True)
    offset_mapping = inputs["offset_mapping"][0].tolist()
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    entities, current_words, current_type = [], [], None
    for pred_id, (start, end) in zip(predictions, offset_mapping):
        if start == end == 0:
            continue
        word_str, label = text[start:end], id2label[pred_id]
        if bool(ENTITY_BREAK_TOKENS.match(word_str)):
            if current_words:
                fs, fe = current_words[0][0], current_words[-1][1]
                entities.append({"text": text[fs:fe], "type": current_type, "position": [fs, fe]})
                current_words, current_type = [], None
            continue
        if label.startswith("B-"):
            if current_words:
                fs, fe = current_words[0][0], current_words[-1][1]
                entities.append({"text": text[fs:fe], "type": current_type, "position": [fs, fe]})
            current_words, current_type = [(start, end)], label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_words.append((start, end))
        else:
            if current_words:
                fs, fe = current_words[0][0], current_words[-1][1]
                entities.append({"text": text[fs:fe], "type": current_type, "position": [fs, fe]})
                current_words, current_type = [], None
    if current_words:
        fs, fe = current_words[0][0], current_words[-1][1]
        entities.append({"text": text[fs:fe], "type": current_type, "position": [fs, fe]})
    return entities