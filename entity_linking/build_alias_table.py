"""
Xây alias table từ NGUỒN HỢP LỆ, không tự chế theo từng case trong 100 file test.

2 nguồn dùng ở đây:
1. Rule biến đổi HÌNH THÁI TIẾNG VIỆT có tính hệ thống — áp dụng đồng loạt cho
   TOÀN BỘ dictionary, không riêng cho từ nào — nên không thể coi là hardcode
   case-cụ-thể (khác hẳn việc gõ tay "gout -> gút" chỉ vì thấy nó fail).
2. ICD-10 Alphabetical Index (Volume 3, WHO) — tài liệu chính thức được thiết kế
   ĐÚNG CHO MỤC ĐÍCH NÀY (map hàng vạn thuật ngữ lâm sàng vào mã), tồn tại độc
   lập với bất kỳ test set nào. Cần tự tìm file (thường đi kèm bản ICD-10 đầy đủ
   từ WHO hoặc bản dịch Việt hoá) — code dưới đây có sẵn parser, chỉ cần trỏ path.
"""
import re
import itertools
import pandas as pd
from normalize import basic_normalize


# ============================================================
# NGUỒN 1: Rule biến đổi hình thái — áp dụng cho MỌI entry trong
# dict, không phải danh sách case cụ thể. Đây là các pattern
# NGỮ PHÁP tiếng Việt y khoa đã được ghi nhận rộng rãi (không
# phải quan sát riêng từ 100 file test):
#   - "bệnh X" <-> "X"                  (rụng tiền tố phân loại)
#   - "viêm X" <-> "nhiễm trùng X"       (2 cách diễn đạt viêm nhiễm)
#   - "chứng X" <-> "X"                  (rụng tiền tố hội chứng)
#   - "rối loạn X" <-> "X"               (rụng tiền tố rối loạn)
#   - "type 1/2" <-> "tuýp 1/2" <-> "týp 1/2"  (biến thể phiên âm)
# ============================================================

MORPHOLOGICAL_RULES = [
    (r"^bệnh\s+", ""),
    (r"^chứng\s+", ""),
    (r"^hội chứng\s+", ""),
    (r"^rối loạn\s+", ""),
    (r"\btype\s*(\d)\b", r"tuýp \1"),
    (r"\btuýp\s*(\d)\b", r"type \1"),
    (r"\btýp\s*(\d)\b", r"tuýp \1"),
    (r"^viêm\s+", "nhiễm trùng "),
    (r"^nhiễm trùng\s+", "viêm "),
    (r"^nhiễm khuẩn\s+", "viêm "),
]


def generate_morphological_variants(name_vi: str) -> set:
    """Sinh các biến thể hình thái từ 1 tên bệnh, áp dụng đồng loạt theo rule
    ngữ pháp — KHÔNG phụ thuộc vào việc tên đó có xuất hiện trong test set nào."""
    variants = {basic_normalize(name_vi)}
    for pattern, replacement in MORPHOLOGICAL_RULES:
        new_variants = set()
        for v in variants:
            new_v = re.sub(pattern, replacement, v)
            if new_v != v:
                new_variants.add(new_v)
        variants |= new_variants
    return variants


def build_morphological_alias_table(icd_df: pd.DataFrame) -> pd.DataFrame:
    """Áp dụng rule cho TOÀN BỘ dict — tính hệ thống, không chọn lọc case nào."""
    rows = []
    for _, row in icd_df.iterrows():
        variants = generate_morphological_variants(row["name_vi"])
        for v in variants:
            rows.append({"code": row["code"], "norm_name_vi": v, "source": "morphological_rule"})
    return pd.DataFrame(rows).drop_duplicates()


# ============================================================
# NGUỒN 2: Danh sách từ mượn tiếng Anh/Pháp phổ biến trong y văn
# Việt Nam — đây là các cặp từ ĐÃ ĐƯỢC GHI NHẬN RỘNG RÃI trong
# giảng dạy y khoa Việt Nam (sách giáo khoa, từ điển y khoa),
# không phải quan sát riêng từ 100 file test hay bộ paraphrase.
# CẦN ĐỐI CHIẾU LẠI với ICD-10 Index Volume 3 nếu tìm được, đây
# chỉ là danh sách khởi điểm từ kiến thức y khoa phổ thông.
# ============================================================

COMMON_LOANWORD_PAIRS = [
    ("gout", "gút"),
    ("graves", "basedow"),
    ("stroke", "đột quỵ"),
    ("migraine", "đau nửa đầu"),
    ("hepatitis", "viêm gan"),
    ("cirrhosis", "xơ gan"),
    ("pneumonia", "viêm phổi"),
    ("asthma", "hen phế quản"),
    ("hen suyễn", "hen phế quản"),
    ("diabetes", "đái tháo đường"),
    ("hypertension", "tăng huyết áp"),
    ("anemia", "thiếu máu"),
]


def build_loanword_alias_table(icd_df: pd.DataFrame) -> pd.DataFrame:
    """Với mỗi cặp loanword, nếu tên tiếng Việt chuẩn (vế thứ 2) xuất hiện
    trong dict (dạng contains, kiểm tra thủ công 1 lần khi build, không phải
    runtime), gắn alias vế thứ 1 vào cùng mã đó."""
    rows = []
    norm_names = icd_df["name_vi"].apply(basic_normalize)
    for loanword, vi_term in COMMON_LOANWORD_PAIRS:
        vi_term_norm = basic_normalize(vi_term)
        mask = norm_names.str.contains(vi_term_norm, regex=False)
        matched = icd_df[mask]
        for _, row in matched.iterrows():
            rows.append({"code": row["code"], "norm_name_vi": basic_normalize(loanword),
                         "source": "loanword_pair"})
    return pd.DataFrame(rows).drop_duplicates()


# ============================================================
# NGUỒN 3 (khung sẵn, cần file thật): ICD-10 Alphabetical Index Vol.3
# ============================================================

def parse_icd10_index_volume3(index_file_path: str) -> pd.DataFrame:
    """
    Parser khung cho ICD-10 Alphabetical Index (Volume 3).
    Định dạng file thực tế thay đổi tuỳ nguồn (PDF cần OCR/extract trước,
    hoặc bản đã số hoá dạng text/csv) — sửa lại logic parse bên trong cho
    khớp định dạng file bạn tìm được.

    Cấu trúc điển hình của Index: mỗi dòng là 1 thuật ngữ (có thể lồng nhau
    theo cấp độ thụt đầu dòng) kèm mã ICD-10 ở cuối dòng, ví dụ:
        Rung nhĩ .......................... I48.9
          - kịch phát ...................... I48.0
          - mạn tính ....................... I48.2
    """
    raise NotImplementedError(
        "Cần file ICD-10 Index Volume 3 thật để biết định dạng cụ thể. "
        "Tìm bản PDF/text từ WHO (icd.who.int) hoặc bản Việt hoá đi kèm "
        "Thông tư 06/2026/TT-BYT, sau đó quay lại để mình viết parser khớp "
        "đúng định dạng."
    )


if __name__ == "__main__":
    # Demo với dict giả lập nhỏ để verify rule hoạt động đúng
    fake_dict = pd.DataFrame([
        {"code": "M10", "name_vi": "Bệnh gút"},
        {"code": "E11", "name_vi": "Đái tháo đường type 2"},
        {"code": "J45", "name_vi": "Hen phế quản"},
        {"code": "I63", "name_vi": "Nhồi máu não"},
    ])

    print("=== Test morphological rules ===")
    morph_table = build_morphological_alias_table(fake_dict)
    print(morph_table)

    print("\n=== Test loanword pairs ===")
    loan_table = build_loanword_alias_table(fake_dict)
    print(loan_table)
