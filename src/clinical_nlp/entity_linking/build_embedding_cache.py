import sys, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dictionary import load_icd10
EXCEL_PATH = "ICD_23_8_2021_132741591570640911_a21a50083a.xlsx"
CODE_COL = "Mã"
NAME_COL = "Tên bệnh"
HEADER_ROW = 4
OUTPUT = "icd10_embeddings.npy" 
MODEL_NAME = "BAAI/bge-m3"
icd_df = load_icd10(EXCEL_PATH, code_col=CODE_COL, name_col=NAME_COL, header_row=HEADER_ROW)
texts = icd_df["norm_name_vi"].astype(str).tolist()
print(f"Loaded {len(texts)} entries")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded, dim={model.get_sentence_embedding_dimension()}")
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    batch_size=64,
)
np.save(OUTPUT, embeddings.astype("float32"))
print(f"Saved {embeddings.shape} to {OUTPUT}")
