import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dictionary import load_icd10
from matcher_vectorized import DiseaseMatcher
EXCEL_PATH = "ICD_23_8_2021_132741591570640911_a21a50083a.xlsx"
CODE_COL = "Mã"
NAME_COL = "Tên bệnh"
HEADER_ROW = 4
EMBED_CACHE = "icd10_embeddings.npy" 
from run_paraphrase_eval import PARAPHRASE_TEST_SET as TEST_SET
icd_df = load_icd10(EXCEL_PATH, code_col=CODE_COL, name_col=NAME_COL, header_row=HEADER_ROW)
all_embeddings = np.load(EMBED_CACHE)
print(f"Embeddings: {all_embeddings.shape}")
text_to_emb = {}
for i, row in icd_df.iterrows():
    text = str(row["norm_name_vi"]).strip()
    if text not in text_to_emb:
        text_to_emb[text] = all_embeddings[i]

class CachedEmbedder:
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        result = []
        for t in texts:
            t = t.strip()
            if t in text_to_emb:
                result.append(text_to_emb[t])
            else:
                # text ngoài dict (query paraphrase) — không có trong cache
                # -> trả về zeros (sanity check sẽ fail, luôn reject)
                result.append(np.zeros(all_embeddings.shape[1], dtype="float32"))
        return np.stack(result)

embed_fn = CachedEmbedder()

# Note: sanity check chỉ cần embed cho candidate text (có trong dict)
# nên CachedEmbedder luôn trả đúng vector cho candidate.
# Query text paraphrase thì trả zeros -> cosine = 0 -> luôn dưới threshold -> luôn reject.
#
# CẦN SỬA: query embedding từ model nữa. Nhưng sweep này cần model.
# Tạm thời: dùng BGE-M3 encode cả query + candidate thật.
# Load model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")

def embed_both_query_and_candidates(query_text, candidate_texts):
    """Encode query + candidates trong 1 batch để nhanh."""
    all_texts = [query_text] + list(candidate_texts)
    emb = model.encode(all_texts, normalize_embeddings=True)
    return emb[0], emb[1:]  # query_vec, candidate_vecs

class FullEmbedder:
    def __init__(self, model):
        self.model = model
    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")
embed_fn_full = FullEmbedder(model)
THRESHOLDS = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
for threshold in THRESHOLDS:
    matcher = DiseaseMatcher(icd_df, embed_fn=embed_fn_full, embed_sanity_threshold=threshold)
    t0 = time.time()
    results = []
    for text, expected_code, ref_name in TEST_SET:
        matches = matcher.match(text)
        predicted = matches[0].code if matches else None
        stage = matches[0].stage if matches else "unmatched"
        results.append({
            "input": text, "expected": expected_code,
            "predicted": predicted, "stage": stage,
            "correct": predicted == expected_code,
            "ref": ref_name,
        })
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    stages = {}
    for r in results:
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
    elapsed = time.time() - t0
    # Đếm số case bị sanity reject (có trong fuzzy nhưng bị loại)
    fuzzy_rejected = sum(1 for r in results
                         if r["stage"] == "unmatched" and r["predicted"] is None
                         and any(r["input"] == text for text, _, _ in TEST_SET))

    print(f"threshold={threshold:.2f}")
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")
    print(f"Stages: exact={stages.get('exact',0)}, exact_nodia={stages.get('exact_no_diacritic',0)}, "
          f"token_set={stages.get('token_set_fuzzy',0)}, edit_dist={stages.get('edit_distance_fuzzy',0)}, "
          f"unmatched={stages.get('unmatched',0)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/query)")

    # In các case SAI (chỉ threshold đầu và cuối để không quá dài):
    if threshold in (0.0, 0.35, 0.50):
        wrong = [r for r in results if not r["correct"]]
        print(f"  Wrong cases ({len(wrong)}):")
        for w in wrong[:5]:
            print(f"    '{w['input'][:50]}' -> pred={w['predicted']}, exp={w['expected']} ({w['stage']}, ref={w['ref']})")
        if len(wrong) > 5:
            print(f"{len(wrong)-5} case sai")
    print()
