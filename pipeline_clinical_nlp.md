# Pipeline xử lý bài toán Clinical NLP tiếng Việt

## Ràng buộc đã xác nhận
- Không dùng LLM API đóng (Claude/GPT...) trong pipeline inference/training chính thức
- Được dùng LLM để hỗ trợ sinh/augment training data (offline, trước khi train)
- Phải nộp được: source code + data + **model weights** để BTC tái dựng trên private test
- Input: văn bản EHR tiếng Việt, 3 section (Tiền sử bệnh / Bệnh sử hiện tại / Đánh giá tại BV)
- Output: JSON list các entity với `text`, `type`, `candidates`, `assertions`, `position`

---

## PHASE 0 — Chuẩn bị & phân tích (không train gì cả)

**Việc làm:**
- Đọc thủ công 10-15 file trong 100 file test (chỉ đọc, không gán nhãn để train) để nắm văn phong, section pattern, loại entity xuất hiện
- Liệt kê schema chính thức: xác nhận với BTC (hoặc suy luận từ ví dụ đề) đầy đủ các `type` (THUỐC, TRIỆU_CHỨNG, BỆNH/CHẨN_ĐOÁN, KẾT_QUẢ_XÉT_NGHIỆM, THÔNG_TIN_BỆNH_NHÂN...) và đầy đủ các `assertions` khả dĩ (isHistorical, negated, planned...)
- Quyết định: vital sign packed-string (`VS98.3 12987 56 18 99RA`) có tách entity riêng hay gộp 1 entity — ảnh hưởng kiến trúc phase 3

**Output của phase:** 1 file schema.md ghi rõ toàn bộ type/assertion đã xác nhận, dùng làm "hợp đồng" cho các phase sau.

---

## PHASE 1 — Xây dựng training data (được phép dùng LLM ở đây)

### 1a. Thu thập data công khai
- ViMQ (SYMPTOM&DISEASE, MEDICAL_PROCEDURE, MEDICINE) — github Huy et al. 2021
- PhoNER_COVID19 (VinAI) — lọc lấy các câu có SYMPTOM&DISEASE, bỏ các type dịch tễ không liên quan (transportation, patient_id...)
- ViMedNER (nếu tìm được repo cụ thể) — bệnh, triệu chứng, nguyên nhân, chẩn đoán, điều trị

### 1b. Sinh thêm data giả lập bằng LLM (offline, không phải pipeline chính thức)
- Prompt LLM sinh đoạn văn bản EHR giả lập theo đúng cấu trúc 3-section, có chèn: viết tắt y khoa (po, bid, ecg...), lỗi chính tả dính chữ, vital sign packed-string, negation, historical, planned
- Review thủ công, sửa lỗi sinh nếu cần
- **QUAN TRỌNG:** không lấy `candidates` (mã ICD-10/RxNorm) từ LLM — LLM dễ bịa mã sai

### 1c. Gán nhãn (annotation)
- Gán span (start, end), type cho từng entity trong data đã sinh/thu thập
- Gán assertion theo cue rule đã thống nhất ở Phase 0
- Gán `candidates` bằng tra cứu dictionary thật (xem Phase 2)
- Công cụ gợi ý: Label Studio hoặc doccano (annotation tool mã nguồn mở, có thể tự host)

**Output của phase:** bộ training set (.jsonl) theo đúng schema, tách train/dev, KHÔNG chứa file nào trong 100 file test gốc.

---

## PHASE 2 — Xây dựng Dictionary tra cứu (Entity Linking resource)

### 2a. RxNorm (thuốc)
- Tải trực tiếp từ NLM (National Library of Medicine, Mỹ) — public, miễn phí, đầy đủ
- Build index: tên thuốc (+ liều lượng, dạng bào chế) → RxCUI
- Vì phần lớn tên thuốc trong văn bản Việt đã viết nguyên dạng tiếng Anh (aspirin, metoprolol...) → fuzzy string match là đủ cho đa số case, không cần cross-lingual

### 2b. ICD-10 (bệnh)
- Ưu tiên: bản dịch tiếng Việt chính thức của Bộ Y tế (Thông tư 06/2026/TT-BYT, icd.kcb.vn) — nếu không crawl được, dùng bản Excel/PDF phụ lục thông tư tải trực tiếp (thuvienphapluat.vn hoặc cổng BYT)
- Build dictionary: tên bệnh tiếng Việt (+ biến thể/từ đồng nghĩa dân gian) → mã ICD-10
- Nếu thiếu case cụ thể: bổ sung thủ công qua tra cứu UI khi gặp bệnh mới trong quá trình annotate

### 2c. Retrieval engine
- Multilingual embedding (LaBSE / bge-m3 / multilingual-e5) để encode cả span tiếng Việt và entry dictionary
- Kết hợp: (1) exact/fuzzy match trước, (2) embedding retrieval fallback khi không match được, (3) chọn mã tổng quát (mã cha 3 ký tự) khi input không đủ chi tiết để chọn mã con

**Output của phase:** 2 file index (RxNorm, ICD-10) sẵn sàng cho module linking ở Phase 4.

---

## PHASE 3 — Rule-layer cho structured pattern

Xử lý trước bằng regex/rule (không cần model, độ chính xác cao hơn NER thuần cho các pattern có cấu trúc cố định):

- **Vital signs packed-string:** `VS(\d+\.\d)(\d{3})(\d{2})\s(\d+)\s(\d+)\s(\d+)(RA|...)` → tách temp/BP/HR/RR/SpO2
- **Thuốc có cấu trúc chuẩn:** `<tên thuốc> <liều> <đường dùng> <tần suất>` (po, bid, daily, prn...) → tách sẵn boundary trước khi đưa vào NER, giúp model NER dễ định biên hơn
- **Section header detection:** dùng để tạo feature phụ trợ (không phải nhãn cứng) cho assertion ở Phase 5

**Output của phase:** module rule độc lập, chạy trước NER, output ra các span "chắc chắn" + feature phụ trợ.

---

## PHASE 4 — NER (Span detection + Type)

### Kiến trúc
- Fine-tune model mở có thể xuất weights: **PhoBERT** hoặc **ViHealthBERT** (đã pretrain trên domain sức khỏe tiếng Việt) làm base, thêm token classification head (BIO tagging)
- Input: text đã qua Phase 3 (rule-layer) làm feature bổ trợ (concat hoặc feed riêng)
- Train trên data từ Phase 1

### Xử lý offset chính xác
- **Bắt buộc:** `text` trong output luôn lấy bằng `input[start:end]` từ văn bản gốc (kể cả lỗi chính tả), không để model tự sinh lại text — tránh lệch WER do model "sửa" lỗi chính tả

**Output của phase:** model weights NER + script inference (span, type).

---

## PHASE 5 — Assertion Classification

### Kiến trúc 2 lớp (rule trước, model sau)
- **Rule-based (ưu tiên, xử lý trước):**
  - Cue negation: "không", "không có", "không ghi nhận" → negated
  - Cue historical: nằm trong section 1 + không có cue mâu thuẫn ("hôm nay", "hiện tại") → isHistorical
  - Cue planned: "lịch", "sẽ", "dự kiến", "tuần tới" → planned
- **Model fallback (cho case rule không cover):** classifier nhỏ (fine-tune trên embedding của span + cửa sổ ngữ cảnh xung quanh), train trên data Phase 1

**Lưu ý:** Section chỉ là feature phụ, cue trong câu luôn ưu tiên hơn (đã thấy case `atenolol (uống hôm nay)` mâu thuẫn với section).

**Output của phase:** module assertion, gắn thêm nhãn cho mỗi entity đã có từ Phase 4.

---

## PHASE 6 — Entity Linking (Normalization)

- Với mỗi entity type THUỐC/BỆNH, query vào dictionary + retrieval engine đã build ở Phase 2
- Trả về `candidates` (1 hoặc nhiều mã, theo đúng format đề bài — có thể trả top-k nếu không chắc chắn, vì metric dùng Jaccard nên trả dư mã sai sẽ bị phạt, cần cân bằng precision/recall)
- Đây là bước có trọng số điểm cao nhất (0.4) → nên đầu tư kỹ nhất

**Output của phase:** module linking hoàn chỉnh.

---

## PHASE 7 — Tích hợp Pipeline & Inference

```
Input .txt (100 file test)
      │
      ▼
[Rule-layer: Phase 3] ── tách vital signs, chuẩn hoá boundary thuốc
      │
      ▼
[NER model: Phase 4] ── (start, end, type)
      │
      ▼
text = input[start:end]  (bắt buộc lấy từ gốc)
      │
      ├──► [Assertion: Phase 5] ── assertions[]
      │
      └──► [Linking: Phase 6] ── candidates[]
      │
      ▼
Ghép JSON theo đúng format đề bài → output/{i}.json
```

**Output của phase:** script `inference.py` chạy end-to-end, input là thư mục .txt, output là thư mục JSON đúng format nộp bài.

---

## PHASE 8 — Đánh giá nội bộ (Self-evaluation)

- Tự viết script tính lại đúng công thức: `text_score` (1-WER), `assertions_score` (Jaccard), `candidates_score` (Jaccard weighted)
- Chạy trên **dev set tự tạo** (từ Phase 1, có gold label) — KHÔNG chạy trên 100 file test vì không có gold thật cho tập đó
- Dùng kết quả để soi lỗi: lỗi nhiều ở đâu (NER sai type, assertion sai cue, linking sai mã) → quay lại phase tương ứng để cải thiện

**Output của phase:** báo cáo lỗi, vòng lặp cải thiện.

---

## PHASE 9 — Đóng gói nộp bài

- `output.zip` — kết quả predict trên 100 file test (JSON theo đúng cấu trúc đề bài)
- Source code đầy đủ (data processing, training, inference)
- Data đã dùng (bao gồm cả phần LLM hỗ trợ sinh — ghi rõ trong README quy trình tạo, không giấu)
- Model weights (PhoBERT/ViHealthBERT đã fine-tune)
- README hướng dẫn cài đặt, tái lập từ đầu (để BTC verify trên private test)

---

## Ghi chú ưu tiên đầu tư thời gian

Theo trọng số điểm: `candidates_score` (0.4) > `text_score` (0.3) = `assertions_score` (0.3)
→ Nếu thời gian hạn chế, ưu tiên đầu tư Phase 2 (dictionary) + Phase 6 (linking) trước, vì đây vừa là phần khó nhất vừa là phần chiếm điểm cao nhất.
