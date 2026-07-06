"""
Load dictionary RxNorm và ICD-10 thành cấu trúc dùng chung cho matcher.py.
KHÔNG chứa bất kỳ mapping hardcode nào theo case cụ thể — chỉ load nguyên
dictionary gốc rồi chuẩn hoá cột.
"""
import pandas as pd
from normalize import basic_normalize


def load_rxnorm(rrf_path: str, keep_tty=("SCD", "SBD", "IN", "PIN")) -> pd.DataFrame:
    """
    Đọc file RXNCONSO.RRF (RxNorm Full Release, tải từ UMLS/NLM).
    keep_tty: các Term Type giữ lại.
        - SCD/SBD: mức cụ thể (hoạt chất+liều+dạng bào chế) — ưu tiên match trước
        - IN/PIN: mức hoạt chất (ingredient) — dùng làm fallback khi span thiếu liều
    """
    cols = ["RXCUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "RXAUI", "SAUI",
            "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
    df = pd.read_csv(rrf_path, sep="|", names=cols, index_col=False, dtype=str,
                      usecols=["RXCUI", "LAT", "TTY", "STR", "SUPPRESS"])

    df = df[(df["LAT"] == "ENG") & (df["SUPPRESS"] == "N")]
    df = df[df["TTY"].isin(keep_tty)]
    df = df[["RXCUI", "TTY", "STR"]].drop_duplicates()
    df["norm_str"] = df["STR"].apply(basic_normalize)
    return df.reset_index(drop=True)


def load_icd10(excel_path: str, code_col: str, name_col: str,
               name_col_en: str = None) -> pd.DataFrame:
    """
    Đọc file Excel danh mục ICD-10 (Bộ Y tế). Tên cột code_col/name_col cần
    khớp với cấu trúc file thực tế — kiểm tra bằng df.columns trước khi gọi.
    """
    df = pd.read_excel(excel_path, dtype=str)
    df = df.rename(columns={code_col: "code", name_col: "name_vi"})
    if name_col_en and name_col_en in df.columns:
        df = df.rename(columns={name_col_en: "name_en"})
    else:
        df["name_en"] = None

    df = df.dropna(subset=["code", "name_vi"]).drop_duplicates(subset=["code", "name_vi"])
    df["norm_name_vi"] = df["name_vi"].apply(basic_normalize)
    return df.reset_index(drop=True)


def build_alias_table(icd_df: pd.DataFrame, alias_source_path: str = None) -> pd.DataFrame:
    """
    Mở rộng dictionary bằng bảng đồng nghĩa lấy từ NGUỒN CHÍNH THỐNG bên ngoài
    (VD: bảng đồng nghĩa thuật ngữ y khoa Việt-Anh công khai, MedDRA...).

    QUAN TRỌNG: hàm này KHÔNG được dùng để nhét alias tự chế theo từng case
    xuất hiện trong 100 file test — đó chính là hardcode trá hình (CURATED table
    núp dưới tên khác). Alias phải đến từ nguồn từ điển độc lập, có thể audit được.

    Nếu chưa có nguồn alias xác đáng, để alias_source_path=None và bỏ qua bước này —
    thà thiếu coverage còn hơn tạo overfit ẩn.
    """
    if alias_source_path is None:
        return icd_df

    alias_df = pd.read_csv(alias_source_path)  # cột kỳ vọng: code, alias
    alias_df["norm_name_vi"] = alias_df["alias"].apply(basic_normalize)
    alias_df = alias_df[["code", "norm_name_vi"]]

    combined = pd.concat([
        icd_df[["code", "norm_name_vi"]],
        alias_df
    ]).drop_duplicates()
    return combined
