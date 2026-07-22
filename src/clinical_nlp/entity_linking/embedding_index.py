import numpy as np
import pandas as pd


def build_and_save_index(dict_df, save_prefix="icd10_embed", model_name="BAAI/bge-m3"):
    from sentence_transformers import SentenceTransformer
    import faiss

    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)

    texts = dict_df["norm_name_vi"].tolist()
    print(f"Encoding {len(texts)} entries... (có thể mất vài phút với 13K dòng)")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True,
                               batch_size=64)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{save_prefix}.faiss")
    dict_df.to_csv(f"{save_prefix}_meta.csv", index=False)
    print(f"Đã lưu: {save_prefix}.faiss + {save_prefix}_meta.csv")
    return model, index


class CachedEmbedder:
    def __init__(self, model_name="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")


def load_index(save_prefix="icd10_embed"):
    import faiss
    index = faiss.read_index(f"{save_prefix}.faiss")
    meta_df = pd.read_csv(f"{save_prefix}_meta.csv")
    return index, meta_df


if __name__ == "__main__":
    from dictionary import load_icd10

    # SỬA lại cho khớp file + cột thật
    icd_df = load_icd10(excel_path="icd10.xlsx", code_col="Ma", name_col="TenBenh")

    model, index = build_and_save_index(icd_df)
    print("\nXong. Lần sau chỉ cần load_index() để dùng lại, không cần encode lại.")
