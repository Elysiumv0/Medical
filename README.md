# Clinical NLP Pipeline — NER + Entity Linking

Dự án xử lý hồ sơ bệnh án điện tử (EHR) tiếng Việt, gồm 2 module chính:
- **NER** (Named Entity Recognition): trích xuất thực thể y khoa từ văn bản EHR
- **Entity Linking**: chuẩn hóa tên bệnh về mã ICD-10 và tên thuốc về RxNorm

## Cấu trúc thư mục

```text
Medical/
├── README.md
├── pyproject.toml
├── data/
│   └── ner/
│       ├── combined_train.jsonl
│       ├── combined_dev.jsonl
│       ├── combined_test.jsonl
│       ├── phoner_train.jsonl
│       ├── phoner_dev.jsonl
│       ├── phoner_test.jsonl
│       ├── vimq_train.jsonl
│       ├── vimq_dev.jsonl
│       ├── vimq_test.jsonl
│       └── synthetic_ehr*.jsonl
├── models/
│   └── ner_v2/
└── src/
    └── clinical_nlp/
        ├── __init__.py
        ├── ner/
        │   ├── __init__.py
        │   ├── alignment.py
        │   ├── constants.py
        │   ├── convert/
        │   │   ├── __init__.py
        │   │   ├── phoner.py
        │   │   └── vimq.py
        │   ├── evaluate.py
        │   ├── predict.py
        │   ├── sanity.py
        │   ├── sanity_check.py
        │   └── train.py
        └── entity_linking/
            ├── __init__.py
            ├── alias.py
            ├── build_embedding_cache.py
            ├── dictionary.py
            ├── embedding_index.py
            ├── evaluator.py
            ├── matcher.py
            ├── matcher_vectorized.py
            ├── normalize.py
            ├── pipeline.py
            └── sweep_sanity_threshold.py
```

## Chạy nhanh

### 1) NER

Train:
```bash
python3 -m clinical_nlp.ner.train \
  --train_path data/ner/combined_train.jsonl \
  --dev_path data/ner/combined_dev.jsonl \
  --output_dir models/ner_v2_retrain
```

Sanity check trên test:
```bash
python3 -m clinical_nlp.ner.sanity_check \
  --model_path models/ner_v2 \
  --test_path data/ner/combined_test.jsonl
```

### 2) Entity Linking

Chạy pipeline end-to-end:
```bash
python3 -m clinical_nlp.entity_linking.pipeline path/to/input.txt -o output.json
```

Build embedding cache cho ICD-10 dictionary:
```bash
python3 -m clinical_nlp.entity_linking.build_embedding_cache
```

Sweep ngưỡng sanity cho matcher:
```bash
python3 -m clinical_nlp.entity_linking.sweep_sanity_threshold
```

## Ghi chú dữ liệu

- `data/ner/*` là tập dữ liệu NER đã convert/merge.
- `models/ner_v2/` là checkpoint NER hiện tại.
- Các file notebook/Colab không phải là đường chạy chuẩn của repo; ưu tiên các module trong `src/` và lệnh `python -m` ở trên.

## Trạng thái hiện tại

- NER có 5 loại thực thể: `BỆNH`, `TRIỆU_CHỨNG`, `THUỐC`, `THÔNG_TIN_BỆNH_NHÂN`, `KẾT_QUẢ_XÉT_NGHIỆM`.
- Entity Linking có cả matcher string-based và embedding fallback.
- Với các type sparse như `KẾT_QUẢ_XÉT_NGHIỆM`, ưu tiên chốt boundary rule và coverage của pipeline trước khi retrain thêm.

## Notes cho review code

- NER core: `src/clinical_nlp/ner/`
- Entity Linking core: `src/clinical_nlp/entity_linking/`
- Data conversion: `src/clinical_nlp/ner/convert/`
- Generated/model artifacts: `data/`, `models/`
