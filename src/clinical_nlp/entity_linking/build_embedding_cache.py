"""
Build embedding cache cho ICD-10 dictionary — chạy 1 lần, lưu file .npy,
dùng lại cho sanity check sweep và các lần eval sau.

Cần: pip install sentence-transformers
Model: BAAI/bge-m3 (~2.2GB, hỗ trợ tiếng Việt tốt)
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dictionary import load_icd10

# === CONFIG: sửa nếu cần ===
EXCEL_PATH = "ICD_23_8_2021_132741591570640911_a21a50083a.xlsx"
CODE_COL = "Mã"
NAME_COL = "Tên bệnh"
HEADER_ROW = 4
OUTPUT = "icd10_embeddings.npy"     # file cache
MODEL_NAME = "BAAI/bge-m3"

print("=== Bước 1: Load dictionary ===")
icd_df = load_icd10(EXCEL_PATH, code_col=CODE_COL, name_col=NAME_COL, header_row=HEADER_ROW)
texts = icd_df["norm_name_vi"].astype(str).tolist()
print(f"Loaded {len(texts)} entries")

print(f"\n=== Bước 2: Load model {MODEL_NAME} ===")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded, dim={model.get_sentence_embedding_dimension()}")

print(f"\n=== Bước 3: Encode {len(texts)} entries ===")
t0 = time.time()
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)
elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s ({elapsed/len(texts)*1000:.1f}ms/entry)")

print(f"\n=== Bước 4: Save {OUTPUT} ===")
np.save(OUTPUT, embeddings.astype("float32"))
print(f"Saved {embeddings.shape} to {OUTPUT}")
print("DONE ✅ — có thể chạy sweep_sanity_threshold.py tiếp theo")