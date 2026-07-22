from dictionary import load_rxnorm, load_icd10
from matcher import DiseaseMatcher, DrugMatcher
from evaluate import leave_k_out_eval, audit_matches
import json

#BƯỚC 1: Load dictionary thật
# Sửa code_col/name_col cho khớp cấu trúc file Excel ICD-10 thực tế của bạn
icd_df = load_icd10(
    excel_path="path/to/icd10_byt.xlsx",
    code_col="Ma",         
    name_col="TenBenh",    
)
rxnorm_df = load_rxnorm(rrf_path="path/to/RXNCONSO.RRF")

# BƯỚC 2: Leave-k-out để đo generalize TRƯỚC KHI chạy trên 100 file test
# Đây là bước bắt buộc phải làm và ghi lại con số trung thực trước khi nộp bài
disease_recall, stages = leave_k_out_eval(icd_df, DiseaseMatcher, k=50, n_rounds=10)

# BƯỚC 3: Chạy trên entity thật từ NER, audit theo confidence
disease_matcher = DiseaseMatcher(icd_df)
drug_matcher = DrugMatcher(rxnorm_df)

with open("ner_output_sample.json") as f:
    entities = json.load(f)  # [{'text': ..., 'type': ...}, ...] từ output Phase 4

disease_entities = [e for e in entities if e["type"] in ("BỆNH", "CHẨN_ĐOÁN")]
drug_entities = [e for e in entities if e["type"] == "THUỐC"]
audit_matches(disease_entities, disease_matcher)

# BƯỚC 4: Chỉ giữ candidate confidence đủ cao
# với case confidence < 0.6, cân nhắc để candidates=[] thay vì đoán bừa,
# vì Jaccard phạt sai = 0 điểm, trong khi để rỗng khi ground truth cũng rỗng
# vẫn được J=1 theo công thức đề bài.
