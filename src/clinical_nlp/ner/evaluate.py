from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from clinical_nlp.ner.constants import ID2LABEL


def evaluate_ner(predictions, labels):
    true_predictions = [[ID2LABEL[p] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[ID2LABEL[l] for p, l in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "per_type": {k: v for k, v in report.items() if k not in ("micro avg", "macro avg", "weighted avg")},
    }