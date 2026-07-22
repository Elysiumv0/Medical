import pandas as pd
from normalize import basic_normalize
def load_rxnorm(rrf_path: str, keep_tty=("SCD", "SBD", "IN", "PIN")) -> pd.DataFrame:
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
               name_col_en: str = None, header_row: int = None) -> pd.DataFrame:
    df = pd.read_excel(excel_path, engine='openpyxl', dtype=str, header=header_row)
    # Strip whitespace from all column names (Excel merge cells often leave trailing space)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={code_col: "code", name_col: "name_vi"})
    if name_col_en and name_col_en in df.columns:
        df = df.rename(columns={name_col_en: "name_en"})
    else:
        df["name_en"] = None
    df = df.dropna(subset=["code", "name_vi"]).drop_duplicates(subset=["code", "name_vi"])
    df["norm_name_vi"] = df["name_vi"].apply(basic_normalize)
    return df.reset_index(drop=True)


def build_alias_table(icd_df: pd.DataFrame, alias_source_path: str = None) -> pd.DataFrame:
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
