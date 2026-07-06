# Clinical NLP Pipeline — NER + Entity Linking
=======

Dự án xử lý hồ sơ bệnh án điện tử (EHR) tiếng Việt, gồm 2 module chính:
- **NER** (Named Entity Recognition): trích xuất thực thể y khoa từ văn bản EHR
- **Entity Linking**: chuẩn hóa tên bệnh về mã ICD-10 và tên thuốc về RxNorm

---

## Cấu trúc thư mục

```
clinical_nlp_project/
├── README.md                          # File này — tổng quan + changelog
├── pipeline_clinical_nlp.md           # Thiết kế pipeline tổng thể
├── NER_Training_Colab.ipynb           # Notebook train NER trên Google Colab
│
├── ner_data/                          # === MODULE NER ===
│   ├── train_ner.py                   # Script training chính (dùng trong Colab)
│   ├── test_alignment_logic.py        # Verify logic token-label alignment
│   ├── merge_ner_datasets.py          # Gộp 3 nguồn → combined train/dev/test
│   ├── convert_phoner.py              # Convert PhoNER_COVID19 .conll → .jsonl
│   ├── convert_vimq.py                # Convert ViMQ → .jsonl
│   ├── generate_synthetic_ehr.py      # Sinh synthetic EHR batch 1
│   ├── generate_synthetic_ehr_batch2.py
│   ├── generate_synthetic_ehr_batch3.py
│   ├── generate_synthetic_ehr_batch4.py
│   ├── generate_synthetic_ehr_batch5_lab.py  # Sinh synthetic KẾT_QUẢ_XÉT_NGHIỆM
│   │
│   └── converted_data/               # Data đã convert & merge — READY TO TRAIN
│       ├── combined_train.jsonl       # 9,569 câu — TRAIN
│       ├── combined_dev.jsonl         # 1,654 câu — DEV
│       ├── combined_test.jsonl        # 2,009 câu — TEST
│       ├── phoner_*.jsonl             # PhoNER source (sau convert)
│       ├── vimq_*.jsonl               # ViMQ source (sau convert)
│       └── synthetic_ehr_batch*.jsonl # Synthetic bổ sung
│
└── entity_linking/                    # === MODULE ENTITY LINKING ===
    ├── HUONG_DAN.md                   # Hướng dẫn chạy chi tiết
    ├── run_pipeline_example.py        # Entry point chính
    ├── run_paraphrase_eval.py         # Eval paraphrase test set
    ├── matcher.py                     # String matcher (exact → token-set → edit)
    ├── matcher_vectorized.py          # Vectorized fuzzy (nhanh hơn)
    ├── dictionary.py                  # ICD-10 dictionary loader
    ├── normalize.py                   # Text normalization
    ├── evaluate.py                    # Leave-k-out eval logic
    ├── build_alias_table.py           # Alias table từ paraphrase
    ├── build_embedding_index.py       # Embedding fallback (BGE-M3)
    ├── paraphrase_test_set.py         # Bộ test paraphrase 50 case
    ├── paraphrase_test_set_extended.py
    ├── paraphrase_test_set_full.py
    └── test_matcher.py                # Unit test matcher
```

---

## Schema NER — 5 entity types

| Type | Mô tả | Ví dụ | Nguồn dữ liệu |
|---|---|---|---|
| `BỆNH` | Tên bệnh/chẩn đoán | *Tăng huyết áp vô căn* | PhoNER + synthetic |
| `TRIỆU_CHỨNG` | Triệu chứng lâm sàng | *đau ngực trái*, *khó thở* | PhoNER + ViMQ |
| `THUỐC` | Tên thuốc/hoạt chất | *amlodipine 5mg* | ViMQ + synthetic |
| `THÔNG_TIN_BỆNH_NHÂN` | Tuổi/giới/nghề nghiệp | *67 tuổi*, *nam*, *nông dân* | PhoNER (AGE/GENDER/JOB) |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Kết quả lab/xét nghiệm | *bạch cầu 15.000/mm3* | Synthetic (chưa có nguồn thật) |

---

## Phân bố dữ liệu NER (combined_*.jsonl)

| Split | Câu | BỆNH | TRIỆU_CHỨNG | THUỐC | THÔNG_TIN_BỆNH_NHÂN | KẾT_QUẢ_XÉT_NGHIỆM |
|---|---|---|---|---|---|---|
| Train | 9,569 | 1,825 | 11,144 | 1,301 | 2,194 | 600 |
| Dev | 1,654 | 283 | 1,854 | 110 | 785 | 10 |
| Test | 2,009 | 406 | 2,172 | 131 | 1,223 | 4 |

### Nguồn gốc từng split:
- **PhoNER_COVID19** (văn phong báo chí): BỆNH + TRIỆU_CHỨNG + THÔNG_TIN_BỆNH_NHÂN (3,416 entity từ AGE/GENDER/JOB gốc)
- **ViMQ** (câu hỏi bệnh nhân): BỆNH + TRIỆU_CHỨNG + THUỐC
- **Synthetic EHR** (giả lập văn phong bác sĩ): đủ 5 type, oversample x15 trong train

---

## Lịch sử xử lý chính (Changelog)

### 2026-07-06: Hoàn thiện schema + chia tập đúng cách

1. **Drop THỦ_THUẬT hoàn toàn** — type này không có trong schema đề bài chính thức,
   chỉ là từ khóa cũ của ViMQ. Filter ở merge_ner_datasets.py Step 1.

2. **Thêm THÔNG_TIN_BỆNH_NHÂN từ PhoNER_COVID19** — phát hiện tag AGE/GENDER/JOB
   trong PhoNER (682+542+205 entity ở train split), convert_phoner.py ban đầu đã bỏ
   qua vì chỉ giữ SYMPTOM_AND_DISEASE.

   - Bug fix: `patient_info_count = 0` bị đặt trong vòng lặp for → log luôn báo 0
     dù entity đã được ghi đúng vào file. Kiểm tra grep trực tiếp file JSONL mới
     phát hiện (kỷ luật "không tin số print"). Fix: chuyển biến ra trước vòng lặp.

3. **KẾT_QUẢ_XÉT_NGHIỆM — viết thêm 30 câu synthetic + chia 70/15/15**:
   - Trước: chỉ 14 entity (8 câu), toàn bộ dồn vào Train → Dev/Test = 0 support
   - Sau: 54 entity (38 câu), Train=600 (oversample), Dev=10, Test=4
   - File: `generate_synthetic_ehr_batch5_lab.py` → `synthetic_ehr_batch5_lab.jsonl`

4. **Merge logic: chia synthetic 70/15/15** thay vì "tách hết KQXN cho dev/test,
   train không có gì". Đảm bảo cả 3 tập đều >0 cho mọi entity type.

### Các commit trước (brief):

- **2026-07-04**: Convert ViMQ (THUỐC/THỦ_THUẬT) + PhoNER (BỆNH/TRIỆU_CHỨNG),
  merge 3 nguồn, tạo synthetic batch 1-4, oversample x15
- **Entity Linking module**: Leave-k-out eval, paraphrase test set 53 case,
  string matcher 35.8% (minh chứng cần embedding fallback)

---

## Cách train trên Google Colab

1. Upload 7 file lên Colab (xem `NER_Training_Colab.ipynb` Cell 0):
   - 3 file combined_*.jsonl
   - train_ner.py, test_alignment_logic.py, merge_ner_datasets.py
   - File entity_types.py (nếu có)

2. Chạy notebook từng cell — sẽ cài PhoBERT-base-v2,
   train ~3-5 epochs, xuất classification report per entity type

3. Theo dõi:
   - F1 trên Dev/Test (đặc biệt KẾT_QUẢ_XÉT_NGHIỆM — support thấp, sẽ F1 thấp)
   - Loss trên synthetic vs PhoNER/ViMQ (cảnh báo memorize do oversample)

---

## Cách chạy Entity Linking

Xem `entity_linking/HUONG_DAN.md` — cần:
- File ICD-10 Excel (tải từ nguồn Bộ Y tế/BHXH)
- RxNorm Full Release (đăng ký UMLS UTS, tải RXNCONSO.RRF)
- `pip install pandas rapidfuzz openpyxl`
- Sửa `code_col`/`name_col` trong `run_pipeline_example.py` khớp file Excel thật
- Chạy `python3 run_pipeline_example.py`

---

## Known Issues & Next Steps

| Issue | Impact | Fix Plan |
|---|---|---|
| Synthetic oversample x15 — model memorize | Train F1 >> Dev F1 trên EHR | Theo dõi loss từng source; giảm oversample nếu gap >15% |
| KẾT_QUẢ_XÉT_NGHIỆM chỉ 10+4 mẫu ở Dev/Test | F1 không ổn định, CI rộng | Cần data thật (lab report de-identified) |
| Entity Linking string-only < 40% | Không dùng được thật tế | Thêm BGE-M3 embedding fallback |
| Chưa có RxNorm data thật | Entity Linking mới test bộ paraphrase ICD-10 | Đăng ký UMLS UTS → tải RxNorm Full Monthly |
