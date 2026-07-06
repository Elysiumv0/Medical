"""
Ví dụ cách cắm dictionary thật vào. Bạn cần sửa lại đường dẫn file
và tên cột cho khớp với file ICD-10 Excel / RxNorm RRF thực tế của bạn.
"""
from dictionary import load_rxnorm, load_icd10
from matcher import DiseaseMatcher, DrugMatcher
from evaluate import leave_k_out_eval, audit_matches
import json

# ==== BƯỚC 1: Load dictionary thật ====
# Sửa code_col/name_col cho khớp cấu trúc file Excel ICD-10 thực tế của bạn
# (mở file bằng pandas trước, in df.columns ra để biết tên cột chính xác)
icd_df = load_icd10(
    excel_path="path/to/icd10_byt.xlsx",
    code_col="Ma",          # <-- sửa theo cột thật
    name_col="TenBenh",     # <-- sửa theo cột thật
)

rxnorm_df = load_rxnorm(rrf_path="path/to/RXNCONSO.RRF")

# ==== BƯỚC 2: Leave-k-out để đo generalize TRƯỚC KHI chạy trên 100 file test ====
# Đây là bước bắt buộc phải làm và ghi lại con số trung thực trước khi nộp bài
disease_recall, stages = leave_k_out_eval(icd_df, DiseaseMatcher, k=50, n_rounds=10)
print(f"\n>>> Recall generalize của DiseaseMatcher: {disease_recall:.1%}")
print(">>> Nếu số này thấp (<50-60%), đừng vội chạy trên 100 file — quay lại")
print(">>> cải thiện matcher trước (thêm alias từ nguồn độc lập, tune threshold).")

# ==== BƯỚC 3: Chạy trên entity thật từ NER, audit theo confidence ====
disease_matcher = DiseaseMatcher(icd_df)
drug_matcher = DrugMatcher(rxnorm_df)

with open("ner_output_sample.json") as f:
    entities = json.load(f)  # [{'text': ..., 'type': ...}, ...] từ output Phase 4

disease_entities = [e for e in entities if e["type"] in ("BỆNH", "CHẨN_ĐOÁN")]
drug_entities = [e for e in entities if e["type"] == "THUỐC"]

print("\n=== Audit BỆNH ===")
audit_matches(disease_entities, disease_matcher)

# ==== BƯỚC 4: Chỉ giữ candidate confidence đủ cao, review tay phần còn lại ====
# Gợi ý: với case confidence < 0.6, cân nhắc để candidates=[] thay vì đoán bừa,
# vì Jaccard phạt sai = 0 điểm, trong khi để rỗng khi ground truth cũng rỗng
# vẫn được J=1 theo công thức đề bài.
