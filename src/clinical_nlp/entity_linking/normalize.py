import re
import unicodedata
MEDICAL_ABBREVIATIONS = {
    "po": "per oral",
    "bid": "bis in die",
    "tid": "ter in die",
    "qid": "quater in die",
    "qam": "quaque ante meridiem",
    "qhs": "quaque hora somni",
    "prn": "pro re nata",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "spo2": "oxygen saturation",
    "ra": "room air",
    "osh": "outside hospital",
    "ef": "ejection fraction",
    "ct": "computed tomography",
}


def strip_diacritics(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def basic_normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Bỏ dấu câu không mang nghĩa (giữ lại dấu gạch nối trong liều thuốc như "325-650")
    text = re.sub(r"[.,;:!?()\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_matching(text: str, expand_abbrev: bool = False) -> str:
    text = basic_normalize(text)
    if expand_abbrev:
        tokens = text.split()
        tokens = [MEDICAL_ABBREVIATIONS.get(t, t) for t in tokens]
        text = " ".join(tokens)
    return text


def extract_dose_info(text: str):
    dose_match = re.search(r"(\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?)?)\s*(mg|mcg|g|ml|iu)\b", text.lower())
    route_match = re.search(r"\b(po|iv|im|sc|sl|pr)\b", text.lower())
    freq_match = re.search(r"\b(daily|bid|tid|qid|qam|qhs|prn|q\d+h)\b", text.lower())

    return {
        "dose": dose_match.group(1) if dose_match else None,
        "unit": dose_match.group(2) if dose_match else None,
        "route": route_match.group(1) if route_match else None,
        "freq": freq_match.group(1) if freq_match else None,
    }
