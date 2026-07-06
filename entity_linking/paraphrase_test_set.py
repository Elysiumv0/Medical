"""
Bộ test set PARAPHRASE — đo đúng bài toán thật: input là cách diễn đạt lâm sàng/
dân gian tiếng Việt (KHÔNG trùng string với tên chuẩn trong dict), chạy matcher
với DICT ĐẦY ĐỦ NGUYÊN VẸN (không xoá gì cả, khác hẳn leave-k-out).

QUAN TRỌNG:
- Các case này tự soạn dựa trên kiến thức y khoa phổ thông, KHÔNG lấy từ 100 file
  test của cuộc thi — tránh circular evaluation.
- Mã ICD-10 liệt kê dưới đây là mã phổ biến, thường gặp trong y văn, nhưng
  BẮT BUỘC phải tự đối chiếu lại qua icd.kcb.vn (bản 06/2026/TT-BYT) trước khi
  dùng làm gold chính thức — vì có thể lệch phiên bản (3 vs 4 ký tự), và đây là
  domain rủi ro cao nếu tin nhầm.
"""

# Format: (paraphrase_vi, expected_icd10_code, tên_chuẩn_tham_khảo)
# Paraphrase cố tình viết khác hẳn cách gọi "sách vở" để test khả năng hiểu nghĩa
PARAPHRASE_TEST_SET = [
    # Tim mạch — chapter I, nơi leave-k-out báo 0% oan uổng nhất
    ("loạn nhịp tim nhanh không đều ở tâm nhĩ", "I48", "Rung nhĩ"),
    ("tim đập nhanh và không đều nhịp", "I48", "Rung nhĩ"),
    ("cao huyết áp", "I10", "Tăng huyết áp vô căn"),
    ("huyết áp lên cao", "I10", "Tăng huyết áp vô căn"),
    ("nhồi máu cơ tim cấp", "I21", "Nhồi máu cơ tim cấp"),
    ("bị đau tim, tắc mạch vành cấp", "I21", "Nhồi máu cơ tim cấp"),
    ("suy tim sung huyết", "I50", "Suy tim"),
    ("tim yếu, ứ nước phổi do tim", "I50", "Suy tim"),
    ("thiếu máu cơ tim mạn tính do xơ vữa", "I25", "Bệnh tim thiếu máu cục bộ mạn"),
    ("tai biến mạch máu não", "I63", "Nhồi máu não"),
    ("đột quỵ do thiếu máu não", "I63", "Nhồi máu não"),

    # Nội tiết - chuyển hoá
    ("tiểu đường tuýp 2", "E11", "Đái tháo đường type 2"),
    ("bệnh tiểu đường không phụ thuộc insulin", "E11", "Đái tháo đường type 2"),
    ("mỡ máu cao", "E78", "Rối loạn chuyển hoá lipoprotein"),
    ("cholesterol cao", "E78", "Rối loạn chuyển hoá lipoprotein"),

    # Hô hấp
    ("viêm phổi", "J18", "Viêm phổi không xác định tác nhân"),
    ("phổi bị nhiễm trùng, có đờm sốt", "J18", "Viêm phổi không xác định tác nhân"),
    ("hen suyễn", "J45", "Hen phế quản"),
    ("khó thở do co thắt phế quản mạn tính, tái phát", "J45", "Hen phế quản"),
    ("bệnh phổi tắc nghẽn mạn tính", "J44", "COPD"),
    ("phổi tắc nghẽn do hút thuốc lâu năm", "J44", "COPD"),
    ("viêm phế quản cấp", "J20", "Viêm phế quản cấp"),

    # Tiêu hoá
    ("viêm loét dạ dày", "K25", "Loét dạ dày"),
    ("đau bao tử, loét niêm mạc dạ dày", "K25", "Loét dạ dày"),
    ("xơ gan do rượu", "K70.3", "Xơ gan do rượu"),
    ("gan bị chai hoá do uống rượu nhiều năm", "K70.3", "Xơ gan do rượu"),
    ("viêm gan siêu vi B mạn", "B18.1", "Viêm gan virus B mạn tính"),
    ("nhiễm virus viêm gan B lâu năm", "B18.1", "Viêm gan virus B mạn tính"),

    # Thận - tiết niệu
    ("suy thận mạn", "N18", "Bệnh thận mạn"),
    ("thận suy giảm chức năng lâu ngày", "N18", "Bệnh thận mạn"),
    ("nhiễm trùng đường tiểu", "N39.0", "Nhiễm trùng đường tiết niệu"),
    ("viêm bàng quang, tiểu buốt tiểu rắt", "N39.0", "Nhiễm trùng đường tiết niệu"),

    # Cơ xương khớp
    ("thoái hoá khớp gối", "M17", "Thoái hoá khớp gối"),
    ("khớp gối mòn sụn, đau khi đi lại", "M17", "Thoái hoá khớp gối"),
    ("gút", "M10", "Gút"),
    ("bệnh gout, đau khớp ngón chân cái", "M10", "Gút"),

    # Nội tiết khác
    ("cường giáp", "E05", "Nhiễm độc giáp"),
    ("bệnh Basedow, tuyến giáp hoạt động quá mức", "E05", "Nhiễm độc giáp"),
    ("suy giáp", "E03", "Suy giáp"),

    # Huyết học
    ("thiếu máu thiếu sắt", "D50", "Thiếu máu do thiếu sắt"),
    ("máu thiếu sắt, da xanh xao mệt mỏi", "D50", "Thiếu máu do thiếu sắt"),

    # Tâm thần
    ("rối loạn lo âu lan toả", "F41.1", "Rối loạn lo âu lan toả"),
    ("hay lo lắng căng thẳng kéo dài không rõ nguyên nhân", "F41.1", "Rối loạn lo âu lan toả"),
    ("trầm cảm", "F32", "Giai đoạn trầm cảm"),
    ("mất ngủ mạn tính", "G47.0", "Rối loạn khởi phát và duy trì giấc ngủ"),

    # Nhiễm trùng khác
    ("sốt xuất huyết", "A90", "Sốt Dengue"),
    ("sốt xuất huyết Dengue", "A91", "Sốt xuất huyết Dengue"),
    ("nhiễm trùng huyết", "A41", "Nhiễm khuẩn huyết khác"),
    ("nhiễm trùng máu nặng, sốc nhiễm trùng", "A41", "Nhiễm khuẩn huyết khác"),

    # Da liễu / dị ứng
    ("viêm da dị ứng", "L20", "Viêm da cơ địa"),
    ("dị ứng da, ngứa nổi mẩn mạn tính", "L20", "Viêm da cơ địa"),

    # Mắt / tai mũi họng
    ("viêm xoang mạn tính", "J32", "Viêm xoang mạn"),
    ("viêm kết mạc mắt", "H10", "Viêm kết mạc"),
]

print(f"Tổng số case trong test set: {len(PARAPHRASE_TEST_SET)}")
