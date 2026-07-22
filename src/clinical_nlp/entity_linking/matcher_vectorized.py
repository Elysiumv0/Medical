"""
Matcher vectorized cho Entity Linking (ICD-10 / RxNorm).

Thay vòng lặp for _, row in df.iterrows() bằng rapidfuzz.process.cdist
để tăng tốc độ x10-50 lần trên dict lớn.

Nguyên tắc:
  - top_k MẶC ĐỊNH = 1 để tối ưu Jaccard (tránh tăng union vô ích).
  - Tie-break: Khi score bằng nhau, ưu tiên mã ngắn hơn (parent) -> mã dài (child).
  - Sanity Check: Dùng embedding cosine similarity làm VETO (loại bỏ match sai chapter).
"""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from rapidfuzz import fuzz, process as rp
from normalize import basic_normalize, strip_diacritics

@dataclass
class MatchResult:
    code: str
    matched_string: str
    stage: str
    confidence: float
    score: float = None

class DiseaseMatcher:
    def __init__(self, dict_df, embed_fn=None, embed_sanity_threshold=0.35):
        self.dict_df = dict_df.reset_index(drop=True)
        self.embed_fn = embed_fn
        self.embed_sanity_threshold = embed_sanity_threshold
        self._norm_texts = self.dict_df["norm_name_vi"].astype(str).tolist()

        # Build exact-lookup
        self._exact_lookup = {}
        self._no_diacritic_lookup = {}
        for _, row in self.dict_df.iterrows():
            n = str(row["norm_name_vi"] or "").strip()
            c = str(row["code"]).strip()
            if n and n not in self._exact_lookup:
                self._exact_lookup[n] = c
            nd = strip_diacritics(n)
            if nd and nd not in self._no_diacritic_lookup:
                self._no_diacritic_lookup[nd] = c

    def _cosine(self, text_a, text_b) -> Optional[float]:
        if self.embed_fn is None: return None
        ea = np.asarray(self.embed_fn([text_a])).flatten()
        eb = np.asarray(self.embed_fn([text_b])).flatten()
        return float(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8))

    def match(self, span_text: str, top_k: int = 1) -> List[MatchResult]:
        query = basic_normalize(span_text)

        # Stage 1: Exact
        if query in self._exact_lookup:
            return [MatchResult(code=self._exact_lookup[query], matched_string=query, stage="exact", confidence=0.98)]

        # Stage 2: Exact no-diacritic
        query_nodia = strip_diacritics(query)
        if query_nodia in self._no_diacritic_lookup:
            return [MatchResult(code=self._no_diacritic_lookup[query_nodia], matched_string=query, stage="exact_no_diacritic", confidence=0.90)]

        # Stage 3: Token-set fuzzy (VECTORIZED)
        scores = rp.cdist([query], self._norm_texts, scorer=fuzz.token_set_ratio, workers=-1)
        sa = np.asarray(scores[0])

        threshold = 80
        cand_idx = np.where(sa >= threshold)[0]
        if len(cand_idx) == 0:
            threshold = 75
            cand_idx = np.where(sa >= threshold)[0]

        if len(cand_idx) > 0:
            # TIE-BREAKING LOGIC
            # sorted_by_score: (-score, length_of_code, has_dot)
            def get_priority(idx):
                c = str(self.dict_df.iloc[idx]["code"]).replace('*','').replace('†','')
                return (len(c), 0 if '.' not in c else 1)

            # Pure Python sort to avoid numpy inhomogeneous shape error
            candidates = []
            for idx in cand_idx:
                candidates.append({
                    'idx': idx,
                    'score': float(sa[idx]),
                    'priority': get_priority(idx)
                })

            # Sort: score DESC, priority ASC
            candidates.sort(key=lambda x: (-x['score'], x['priority']))

            for cand in candidates[:max(top_k, 3)]:
                idx = cand['idx']
                code = str(self.dict_df.iloc[idx]["code"])
                matched_str = str(self.dict_df.iloc[idx]["norm_name_vi"])
                score_val = cand['score']

                if self.embed_fn is not None:
                    cos = self._cosine(query, matched_str)
                    if cos is not None and cos < self.embed_sanity_threshold:
                        continue

                return [MatchResult(
                    code=code, matched_string=matched_str,
                    stage="token_set_fuzzy",
                    confidence=min(0.80, score_val / 100),
                    score=score_val
                )]

        # Stage 4: Edit-distance fuzzy (VECTORIZED)
        scores_ed = rp.cdist([query], self._norm_texts, scorer=fuzz.ratio, workers=-1)
        ea = np.asarray(scores_ed[0])
        ed_idx = np.where(ea >= 88)[0]
        if len(ed_idx) > 0:
            best = ed_idx[np.argmax(ea[ed_idx])]
            code = str(self.dict_df.iloc[best]["code"])
            matched_str = str(self.dict_df.iloc[best]["norm_name_vi"])
            score_val = float(ea[best])
            return [MatchResult(
                code=code, matched_string=matched_str,
                stage="edit_distance_fuzzy",
                confidence=min(0.70, score_val / 100),
                score=score_val
            )]

        return []