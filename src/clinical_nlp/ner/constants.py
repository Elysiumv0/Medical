"""
NER label constants — loaded from configs/ner_labels.yaml (single source of truth).

Usage:
    from clinical_nlp.ner.constants import LABEL_LIST, LABEL2ID, ID2LABEL
"""
from pathlib import Path
import yaml


_config_path = Path(__file__).resolve().parents[4] / "configs" / "ner_labels.yaml"
if not _config_path.exists():
    raise FileNotFoundError(
        f"NER labels config not found at {_config_path}. "
        "Expected configs/ner_labels.yaml in project root."
    )

with open(_config_path, encoding="utf-8") as f:
    _config = yaml.safe_load(f)

LABEL_LIST = _config["labels"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}