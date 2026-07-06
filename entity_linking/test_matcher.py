"""
Test bằng dictionary TỰ BỊA (không liên quan gout/graves/hen suyễn hay bất kỳ
case nào đã xuất hiện trong quá trình thảo luận trước) — để đảm bảo test này
không lại rơi vào bẫy "tự chấm trên case mình đã biết đáp án".
"""
import pandas as pd
from normalize import basic_normalize
from matcher import DiseaseMatcher, DrugMatcher
from evaluate import leave_k_out_eval, audit_matches

# Dictionary giả lập kiểu ICD-10 (tên bịa, cấu trúc thật)
fake_icd = pd.DataFrame([
    {"code": "X01", "name_vi": "Viêm phổi thùy trên phải"},
    {"code": "X02", "name_vi": "Viêm phế quản mạn tính"},
    {"code": "X03", "name_vi": "Suy thận mạn giai đoạn ba"},
    {"code": "X04", "name_vi": "Thoái hóa khớp gối hai bên"},
    {"code": "X05", "name_vi": "Loét dạ dày tá tràng"},
    {"code": "X06", "name_vi": "Viêm gan siêu vi B mạn tính"},
    {"code": "X07", "name_vi": "Thiếu máu thiếu sắt"},
    {"code": "X08", "name_vi": "Rối loạn lo âu lan tỏa"},
    {"code": "X09", "name_vi": "Đái tháo đường type hai"},
    {"code": "X10", "name_vi": "Tăng lipid máu hỗn hợp"},
])
fake_icd["norm_name_vi"] = fake_icd["name_vi"].apply(basic_normalize)

print("### TEST 1: Exact match ###")
matcher = DiseaseMatcher(fake_icd)
r = matcher.match("Viêm phổi thùy trên phải")
print(r)
assert r[0].code == "X01" and r[0].stage == "exact"

print("\n### TEST 2: Đảo trật tự từ (token-set fuzzy) ###")
r = matcher.match("thoái hóa hai bên khớp gối")  # đảo từ so với dict
print(r)
assert r and r[0].code == "X04"

print("\n### TEST 3: Lỗi gõ nhẹ (edit-distance fuzzy) ###")
r = matcher.match("viêm gan siêu vi B mạn tinh")  # thiếu dấu 'tính' -> 'tinh'
print(r)
assert r and r[0].code == "X06"

print("\n### TEST 4: Case hoàn toàn không liên quan -> phải trả rỗng ###")
r = matcher.match("gãy xương đùi trái")
print(r)
assert r == [], "Không được đoán bừa khi không liên quan!"

print("\n### TEST 5: Leave-k-out cross validation (đo generalize thật) ###")
recall, stages = leave_k_out_eval(fake_icd, DiseaseMatcher, k=3, n_rounds=3)

print("\n### TEST 6: Audit theo confidence (không gộp chung coverage) ###")
fake_entities = [
    {"text": "Viêm phổi thùy trên phải", "type": "BỆNH"},   # sẽ exact
    {"text": "khớp gối hai bên thoái hóa", "type": "BỆNH"},  # sẽ fuzzy
    {"text": "bệnh gì đó không có trong dict", "type": "BỆNH"},  # sẽ unmatched
]
audit_matches(fake_entities, matcher)

print("\n✅ Tất cả test pass — logic hoạt động đúng, sẵn sàng cắm dictionary thật vào.")
