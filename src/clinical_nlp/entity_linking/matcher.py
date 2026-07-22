from dataclasses import dataclass
from typing import Optional, List
from rapidfuzz import fuzz
from normalize import basic_normalize, strip_diacritics, extract_dose_info

@dataclass
class MatchResult:
    code: str
    matched_string: str
    stage: str          # tên tầng đã match được, để audit
    confidence: float    # 0-1, ước lượng độ tin cậy của tầng đó
    score: float = None  # raw score nội bộ (fuzzy ratio, cosine...), để debug


class DiseaseMatcher:

    def __init__(self, dict_df, embed_fn=None, embed_index=None,
                 embed_threshold=0.55, embed_sanity_threshold=0.35):

        self.dict_df = dict_df.reset_index(drop=True)
        self.embed_fn = embed_fn
        self.embed_index = embed_index
        self.embed_threshold = embed_threshold
        self.embed_sanity_threshold = embed_sanity_threshold

        self._exact_lookup = dict(zip(self.dict_df["norm_name_vi"], self.dict_df["code"]))
        self._no_diacritic_lookup = {
            strip_diacritics(k): v for k, v in self._exact_lookup.items()
        }

    def _embedding_search(self, span_text, top_k=5):
        if self.embed_fn is None or self.embed_index is None:
            return []
        query_emb = self.embed_fn([span_text])
        scores, indices = self.embed_index.search(query_emb, top_k)
        return [
            (self.dict_df.iloc[idx]["code"], self.dict_df.iloc[idx]["norm_name_vi"], float(score))
            for score, idx in zip(scores[0], indices[0])
        ]

    def _cosine_sim_to_candidate(self, span_text, candidate_norm_str):
        if self.embed_fn is None:
            return None
        q_emb = self.embed_fn([span_text])
        c_emb = self.embed_fn([candidate_norm_str])
        import numpy as np
        q, c = np.asarray(q_emb).flatten(), np.asarray(c_emb).flatten()
        return float(q @ c / ((q @ q) ** 0.5 * (c @ c) ** 0.5 + 1e-8))

    def match(self, span_text: str, top_k: int = 1) -> List[MatchResult]:
        query = basic_normalize(span_text)

        # Stage 1: exact match — không cần kiểm tra chéo gì, độ tin cậy tuyệt đối
        if query in self._exact_lookup:
            return [MatchResult(
                code=self._exact_lookup[query], matched_string=query,
                stage="exact", confidence=0.98
            )]

        # Stage 2: exact sau khi bỏ dấu
        query_nodia = strip_diacritics(query)
        if query_nodia in self._no_diacritic_lookup:
            return [MatchResult(
                code=self._no_diacritic_lookup[query_nodia], matched_string=query,
                stage="exact_no_diacritic", confidence=0.90
            )]

        # Stage 3: EMBEDDING làm tầng ngữ nghĩa chính.
        # Lý do đổi thứ tự: string-fuzzy đứng trước dễ "tự tin nhầm" theo từ khoá chung
        # chung ("nhiễm trùng", "cao") mà bỏ qua ngữ cảnh — đây chính là nguồn lỗi
        # wrong-chapter 26% đã đo được. Embedding xét toàn bộ ngữ nghĩa câu, ít bị lỗi này hơn.
        embed_results = self._embedding_search(span_text, top_k=top_k)
        strong_embed = [r for r in embed_results if r[2] >= self.embed_threshold]
        if strong_embed:
            return [
                MatchResult(code=c, matched_string=s, stage="embedding", confidence=score, score=score)
                for c, s, score in strong_embed
            ]

        # Stage 4: token-set fuzzy — CHỈ chấp nhận nếu embedding xác nhận
        # candidate không phải hoàn toàn lạc đề. Nếu không có embedder, vẫn chạy
        # nhưng nên coi confidence ở mức thấp hơn, và ưu tiên chạy paraphrase eval
        # thường xuyên để phát hiện sớm case wrong-chapter.
        candidates = []
        for _, row in self.dict_df.iterrows():
            score = fuzz.token_set_ratio(query, row["norm_name_vi"])
            if score >= 88:
                candidates.append((row["code"], row["norm_name_vi"], score))

        if candidates:
            candidates.sort(key=lambda x: -x[2])
            verified = []
            for code, matched_str, score in candidates[:top_k]:
                if self.embed_fn is not None:
                    sanity = self._cosine_sim_to_candidate(span_text, matched_str)
                    if sanity is not None and sanity < self.embed_sanity_threshold:
                        continue  # loại — embedding không đồng thuận, khả năng cao là wrong-chapter
                verified.append(MatchResult(
                    code=code, matched_string=matched_str, stage="token_set_fuzzy",
                    confidence=min(0.85, score / 100), score=score
                ))
            if verified:
                return verified

        # Stage 5: edit-distance fuzzy — áp dụng cùng nguyên tắc kiểm tra chéo
        candidates = []
        for _, row in self.dict_df.iterrows():
            score = fuzz.ratio(query, row["norm_name_vi"])
            if score >= 85:
                candidates.append((row["code"], row["norm_name_vi"], score))

        if candidates:
            candidates.sort(key=lambda x: -x[2])
            verified = []
            for code, matched_str, score in candidates[:top_k]:
                if self.embed_fn is not None:
                    sanity = self._cosine_sim_to_candidate(span_text, matched_str)
                    if sanity is not None and sanity < self.embed_sanity_threshold:
                        continue
                verified.append(MatchResult(
                    code=code, matched_string=matched_str, stage="edit_distance_fuzzy",
                    confidence=min(0.75, score / 100), score=score
                ))
            if verified:
                return verified

        # Stage 6: embedding yếu hơn threshold chính nhưng vẫn còn khá — dùng làm
        # phương án cuối cùng trước khi bỏ cuộc, gắn confidence thấp để bạn biết cần review tay
        weak_embed = [r for r in embed_results if r[2] >= self.embed_sanity_threshold]
        if weak_embed:
            return [
                MatchResult(code=c, matched_string=s, stage="embedding_weak", confidence=score, score=score)
                for c, s, score in weak_embed
            ]

        return []


class DrugMatcher:

    def __init__(self, dict_df):
        self.dict_df = dict_df
        self.specific_df = dict_df[dict_df["TTY"].isin(["SCD", "SBD"])]
        self.ingredient_df = dict_df[dict_df["TTY"].isin(["IN", "PIN"])]

    def match(self, span_text: str) -> List[MatchResult]:
        query = basic_normalize(span_text)
        dose_info = extract_dose_info(span_text)

        # Stage 1: exact match ở mức cụ thể (đúng cả tên + liều + dạng bào chế)
        exact = self.specific_df[self.specific_df["norm_str"] == query]
        if not exact.empty:
            row = exact.iloc[0]
            return [MatchResult(code=row["RXCUI"], matched_string=row["norm_str"],
                                 stage="exact_specific", confidence=0.97)]

        # Stage 2: fuzzy ở mức cụ thể — chỉ nhận ngưỡng cao vì sai liều = sai hẳn RXCUI
        best_score, best_row = 0, None
        for _, row in self.specific_df.iterrows():
            score = fuzz.token_sort_ratio(query, row["norm_str"])
            if score > best_score:
                best_score, best_row = score, row
        if best_score >= 90:
            return [MatchResult(code=best_row["RXCUI"], matched_string=best_row["norm_str"],
                                 stage="fuzzy_specific", confidence=min(0.85, best_score / 100),
                                 score=best_score)]

        # Stage 3: fallback về mức hoạt chất (khi span thiếu liều/dạng bào chế)
        # Lấy từ đầu chuỗi làm tên hoạt chất phỏng đoán (trước số/đơn vị đầu tiên)
        ingredient_guess = query.split()[0] if query else ""
        for _, row in self.ingredient_df.iterrows():
            score = fuzz.ratio(ingredient_guess, row["norm_str"])
            if score >= 90:
                return [MatchResult(code=row["RXCUI"], matched_string=row["norm_str"],
                                     stage="ingredient_fallback", confidence=0.55,
                                     score=score)]

        return []
