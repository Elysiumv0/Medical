"""
Đánh giá matcher trên bộ paraphrase test full (179 case).

Dùng matcher_vectorized (rapidfuzz cdist, không embedding, top_k=1).

Cách chạy:
    cd entity_linking   # hoặc clinical-nlp
    python3 run_paraphrase_eval.py

Kết quả in ra:
  - Jaccard (exact code match only)
  - Phân bố theo stage
  - Danh sách case sai (để audit, KHÔNG để vá tay)
"""
import sys
import pandas as pd
from dictionary import load_icd10
from paraphrase_test_set_full import PARAPHRASE_TEST_FULL
from matcher_vectorized import DiseaseMatcher


def eval_paraphrase(dict_df, test_set):
    matcher = DiseaseMatcher(dict_df)

    exact = 0
    chapter_ok = 0
    wrong = 0
    unmatched = 0
    by_stage = {}
    wrong_cases = []
    unmatched_cases = []

    for input_text, expected, canonical in test_set:
        results = matcher.match(input_text)
        if results:
            r = results[0]
            got = r.code.replace('*', '').replace('†', '')
            stage = r.stage
            by_stage[stage] = by_stage.get(stage, 0) + 1

            if got == expected:
                exact += 1
            elif got[:3] == expected[:3]:
                chapter_ok += 1
            else:
                wrong += 1
                wrong_cases.append({
                    'input': input_text,
                    'expected': expected,
                    'canonical': canonical,
                    'got': got,
                    'matched': r.matched_string[:60],
                    'stage': stage,
                    'confidence': r.confidence,
                })
        else:
            unmatched += 1
            unmatched_cases.append({
                'input': input_text,
                'expected': expected,
                'canonical': canonical,
            })

    total = len(test_set)
    jaccard = exact / total

    print(f"\n{'='*60}")
    print(f"=== Paraphrase Eval ({total} cases, dict {len(dict_df):,} entries) ===")
    print(f"Jaccard (exact code match): {jaccard:.3f}")
    print(f"Exact:     {exact:3d}/{total}")
    print(f"Chapter OK:{chapter_ok:3d}/{total}  (right chapter, wrong 4th digit)")
    print(f"Wrong:     {wrong:3d}/{total}  (wrong chapter)")
    print(f"Unmatched: {unmatched:3d}/{total}")
    print(f"\nBy stage: {dict(sorted(by_stage.items()))}")

    # Show wrong cases (first 20)
    if wrong_cases:
        print(f"\n--- Wrong cases ({len(wrong_cases)} total, showing first 20) ---")
        for w in wrong_cases[:20]:
            print(f"  '{w['input']}' → got={w['got']} exp={w['expected']} "
                  f"({w['stage']}, conf={w['confidence']:.2f})")

    if unmatched_cases:
        print(f"\n--- Unmatched cases ({len(unmatched_cases)} total, showing first 10) ---")
        for u in unmatched_cases[:10]:
            print(f"  '{u['input']}' exp={u['expected']}")

    return {
        'jaccard': jaccard,
        'exact': exact,
        'chapter_ok': chapter_ok,
        'wrong': wrong,
        'unmatched': unmatched,
        'by_stage': by_stage,
        'wrong_cases': wrong_cases,
        'unmatched_cases': unmatched_cases,
    }


if __name__ == "__main__":
    # === CONFIG: chỉnh 3 dòng này cho máy bạn ===
    EXCEL_PATH = (
        "ICD_23_8_2021_132741591570640911_a21a50083a.xlsx"
    )
    CODE_COL = "Mã"
    NAME_COL = "Tên bệnh"
    HEADER_ROW = 4
    # ============================================

    import os, sys
    from collections import Counter
    from normalize import basic_normalize

    # Resolve path (current dir → ~/Downloads → WSL /mnt/c/...)
    import platform
    candidates = [
        EXCEL_PATH,
        os.path.expanduser(f"~/Downloads/{EXCEL_PATH}"),
    ]
    if "microsoft" in platform.release().lower():
        candidates.append(
            f"/mnt/c/Users/{os.environ.get('USER', 'ADMIN')}/Downloads/{EXCEL_PATH}"
        )
    found = False
    for p in candidates:
        if os.path.exists(p):
            EXCEL_PATH = p; found = True; break
    if not found:
        print(f"ERROR: File not found. Looked in:")
        for p in candidates: print(f"  {p}")
        print("Adjust EXCEL_PATH in run_paraphrase_eval.py")
        sys.exit(1)

    print(f"Loading: {EXCEL_PATH}")
    icd_df = load_icd10(excel_path=EXCEL_PATH, code_col=CODE_COL,
                        name_col=NAME_COL, header_row=HEADER_ROW)
    print(f"Dict: {len(icd_df):,} entries, {icd_df['code'].nunique()} unique codes")
    print(f"Columns: {icd_df.columns.tolist()}")
    print(f"Sample:\n{icd_df.head(5).to_string()}")

    # === Sanity: check NULL trên TOÀN BỘ dict ===
    total_nulls = (icd_df["norm_name_vi"].isna().sum()
                   + (icd_df["norm_name_vi"].astype(str).str.strip() == "").sum())
    star_count = icd_df["is_star"].sum()
    print(f"\nSanity: {total_nulls}/{len(icd_df)} null norm_name_vi (full dict)")
    print(f"Star/dagger codes: {star_count} ({star_count/len(icd_df)*100:.1f}%)")

    if total_nulls > 50:
        print(f"ERROR: {total_nulls} null names — header_row likely wrong. Aborting.")
        sys.exit(1)

    test_set = PARAPHRASE_TEST_FULL
    print(f"\nTest set: {len(test_set)} cases")

    # === Chạy song song 2 nhánh: lọc star vs giữ nguyên ===
    icd_no_star = icd_df[~icd_df["is_star"]].reset_index(drop=True)
    icd_full = icd_df

    results_no_star = eval_paraphrase(icd_no_star, test_set)
    results_with_star = eval_paraphrase(icd_full, test_set)

    print(f"\n{'='*60}")
    print(f"=== COMPARISON ===")
    print(f"  NO-STAR:   {len(icd_no_star):,} entries → Jaccard={results_no_star['jaccard']:.3f}")
    print(f"  WITH-STAR: {len(icd_full):,} entries → Jaccard={results_with_star['jaccard']:.3f}")
    delta = results_no_star['jaccard'] - results_with_star['jaccard']
    print(f"  Delta: {delta:+.3f}  ({'better' if delta > 0 else 'worse or same'} with star removed)")