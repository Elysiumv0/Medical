import random
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class EvalReport:
    total: int
    exact_or_high_conf: int      # match ở stage exact/exact_no_diacritic
    approximate: int              # match ở stage fuzzy/embedding
    unmatched: int
    by_stage: dict

def leave_k_out_eval(dict_df, matcher_cls, k: int = 30, n_rounds: int = 5,
                      random_seed: int = 42, **matcher_kwargs) -> EvalReport:
    random.seed(random_seed)
    all_stage_counts = {}
    total_correct, total_cases = 0, 0

    for round_idx in range(n_rounds):
        sample_idx = random.sample(range(len(dict_df)), min(k, len(dict_df)))
        held_out = dict_df.iloc[sample_idx].reset_index(drop=True)
        remaining = dict_df.drop(dict_df.index[sample_idx]).reset_index(drop=True)

        matcher = matcher_cls(remaining, **matcher_kwargs)

        for _, row in held_out.iterrows():
            # Dùng tên gốc (chưa chuẩn hoá) làm "input giả lập" — giống tình huống
            # thực tế nhận 1 span text mới trong văn bản
            query_field = "norm_name_vi" if "norm_name_vi" in row else "norm_str"
            results = matcher.match(row[query_field])
            total_cases += 1
            stage = results[0].stage if results else "unmatched"
            all_stage_counts[stage] = all_stage_counts.get(stage, 0) + 1

            correct_code_field = "code" if "code" in row else "RXCUI"
            if results and results[0].code == row[correct_code_field]:
                total_correct += 1

    recall = total_correct / total_cases if total_cases else 0
    print(f"Leave-{k}-out, {n_rounds} rounds ({total_cases} cases total)")
    print(f"Recall: {recall:.1%}")
    print("Phân bố theo stage:")
    for stage, count in sorted(all_stage_counts.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {count} ({count/total_cases:.1%})")

    return recall, all_stage_counts


def audit_matches(entities: List[dict], matcher, gold_lookup: dict = None):
    report = {"exact": [], "high_conf_fuzzy": [], "low_conf_fuzzy_or_embed": [], "unmatched": []}

    for ent in entities:
        results = matcher.match(ent["text"])
        if not results:
            report["unmatched"].append(ent["text"])
            continue

        top = results[0]
        if top.stage in ("exact", "exact_no_diacritic", "exact_specific"):
            report["exact"].append((ent["text"], top.code))
        elif top.confidence >= 0.75:
            report["high_conf_fuzzy"].append((ent["text"], top.code, top.stage, top.confidence))
        else:
            report["low_conf_fuzzy_or_embed"].append((ent["text"], top.code, top.stage, top.confidence))

    total = len(entities)
    print(f"Tổng entity: {total}")
    print(f"  Exact match:              {len(report['exact'])} ({len(report['exact'])/total:.1%})")
    print(f"  High-confidence fuzzy:    {len(report['high_conf_fuzzy'])} ({len(report['high_conf_fuzzy'])/total:.1%})")
    print(f"  Low-confidence (CẦN REVIEW TAY): {len(report['low_conf_fuzzy_or_embed'])} ({len(report['low_conf_fuzzy_or_embed'])/total:.1%})")
    print(f"  Unmatched:                {len(report['unmatched'])} ({len(report['unmatched'])/total:.1%})")

    if report["low_conf_fuzzy_or_embed"]:
        for text, code, stage, conf in report["low_conf_fuzzy_or_embed"][:20]:
            print(f"   '{text}' -> {code} (stage={stage}, conf={conf:.2f})")

    if gold_lookup:
        correct = sum(1 for ent in entities
                      if gold_lookup.get(ent["text"]) == (matcher.match(ent["text"]) or [None])[0])
        print(f"\nAccuracy: {correct/total:.1%}")

    return report
