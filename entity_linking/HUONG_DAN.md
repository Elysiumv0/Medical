# Hướng dẫn chạy Entity Linking Module ở máy local

## 1. Cài đặt môi trường

```bash
pip install pandas rapidfuzz openpyxl
# (tuỳ chọn, chỉ cần nếu dùng tầng embedding fallback)
pip install sentence-transformers faiss-cpu
```

## 2. Lấy dữ liệu ICD-10

Tải file Excel danh mục ICD-10 (ưu tiên bản chính thức Bộ Y tế nếu tìm được;
nếu không, dùng tạm bản BHXH):

```
https://benhviensuoikhoang.com/upload/1000551/20210823/ICD_23_8_2021_132741591570640911_a21a50083a.xlsx
```

Sau khi tải về, chạy lệnh sau để xem cấu trúc cột thật (QUAN TRỌNG — tên cột
mỗi file có thể khác nhau, không đoán mò):

```python
import pandas as pd
df = pd.read_excel("icd10.xlsx")
print(df.columns.tolist())
print(df.head(10))
```

Ghi lại tên cột chứa **mã bệnh** và **tên bệnh tiếng Việt** — sẽ cần điền vào
Bước 4.

## 3. Lấy dữ liệu RxNorm

1. Đăng ký tài khoản UMLS UTS (miễn phí): https://uts.nlm.nih.gov/uts/signup-login
2. Sau khi được duyệt (thường vài phút đến vài giờ), vào:
   https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
3. Tải **RxNorm Full Monthly Release** (file .zip)
4. Giải nén, tìm file `rrf/RXNCONSO.RRF` bên trong

File này khá lớn — nếu máy yếu, có thể lọc trước để giảm dung lượng:

```python
import pandas as pd
cols = ["RXCUI","LAT","TS","LUI","STT","SUI","ISPREF","RXAUI","SAUI",
        "SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"]
df = pd.read_csv("RXNCONSO.RRF", sep="|", names=cols, index_col=False, dtype=str)
df_filtered = df[(df["LAT"]=="ENG") & (df["SUPPRESS"]=="N")]
df_filtered = df_filtered[df_filtered["TTY"].isin(["SCD","SBD","IN","PIN"])]
df_filtered.to_csv("rxnorm_filtered.csv", index=False)
# File csv này nhỏ hơn nhiều, dễ upload lại cho Claude nếu cần hỗ trợ thêm
```

## 4. Sửa `run_pipeline_example.py`

Mở file, sửa đúng 2 chỗ theo cấu trúc cột thật đã xem ở Bước 2:

```python
icd_df = load_icd10(
    excel_path="icd10.xlsx",
    code_col="TÊN_CỘT_MÃ_THẬT",       # <-- sửa
    name_col="TÊN_CỘT_TÊN_BỆNH_THẬT",  # <-- sửa
)
```

## 5. Chạy leave-k-out để có con số recall trung thực đầu tiên

```bash
cd entity_linking
python3 run_pipeline_example.py
```

Kết quả in ra sẽ có dạng:

```
=== Leave-50-out, 10 rounds (500 cases total) ===
Recall (matcher tự tìm lại đúng mã case chưa từng thấy): XX.X%
Phân bố theo stage:
  exact: ...
  token_set_fuzzy: ...
  edit_distance_fuzzy: ...
  unmatched: ...
```

## 6. Gửi lại kết quả để mình giúp đọc & tune tiếp

Khi có kết quả, gửi lại:
- Số % recall tổng
- Phân bố theo stage (đặc biệt tỉ lệ `unmatched` — nếu cao, cần cải thiện threshold hoặc thêm alias nguồn ngoài)
- Nếu recall < 50-60%: đừng chạy tiếp trên 100 file test vội — gửi kết quả để mình giúp chẩn đoán nguyên nhân (threshold sai, dictionary thiếu case phổ biến, hay do đặc thù tiếng Việt cần thêm bước normalize khác)

## Các lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `KeyError: 'Ma'` khi load_icd10 | Tên cột không khớp | Kiểm tra lại `df.columns` như Bước 2 |
| RXNCONSO.RRF đọc bị lệch cột | File có ký tự `|` trong nội dung text | Thêm `quoting=3` (QUOTE_NONE) vào `pd.read_csv` |
| Encoding lỗi (chữ Việt hiển thị sai) | File Excel dùng encoding khác | Thử `pd.read_excel(path, engine="openpyxl")` |
| leave_k_out_eval chạy rất chậm với dict 13K dòng | Vòng lặp `for _, row in df.iterrows()` trong matcher.py là O(n) mỗi query | Nếu cần tăng tốc, có thể báo lại — mình sẽ tối ưu bằng vectorized fuzzy (rapidfuzz.process.cdist) thay vì loop |
