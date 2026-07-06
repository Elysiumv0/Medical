"""
Bộ paraphrase test SET MỞ RỘNG — 100 case mới, tách biệt với 53 case cũ.

NGUỒN: kiến thức y khoa phổ thông của AI (KHÔNG từ 100 file test của bạn,
KHÔNG soi đáp án paraphrase_test_set.py cũ để vá).

CẢNH BÁO: mã ICD-10 do AI liệt kê CẦN ĐƯỢC ĐỐI CHIẾU qua icd.kcb.vn
trước khi dùng làm gold. Đây là phiên bản khởi điểm, có thể sai lệch
(3 vs 4 ký tự, phiên bản ICD-10 2016 vs 2026 Bộ Y tế).

Định dạng: mỗi entry là (paraphrase_dân_gian, mã_ICD10_dự_kiến,
  canonical_name_trong_dict_để_kiểm_tra_sanity).
"""

PARAPHRASE_EXTENDED = [
    # === CHƯƠNG VI: BỆNH HỆ THẦN KINH (G00-G99) ===
    ("động kinh", "G40.9", "động kinh"),
    ("co giật kiểu động kinh", "G40.9", "động kinh"),
    ("bệnh alzheimer", "G30.9", "bệnh alzheimer"),
    ("mất trí nhớ tuổi già", "G30.9", "bệnh alzheimer"),
    ("đau đầu migraine", "G43.9", "migraine"),
    ("đau nửa đầu có aura", "G43.1", "migraine có aura"),
    ("đau đầu migraine kinh điển", "G43.1", "migraine có aura"),
    ("đau dây thần kinh tọa", "M54.3", "đau thần kinh tọa"),
    ("đau thần kinh liên sườn", "G58.0", "đau dây thần kinh liên sườn"),
    ("liệt mặt ngoại biên", "G51.0", "liệt bell"),
    ("liệt dây thần kinh số 7", "G51.0", "liệt bell"),
    ("hội chứng ống cổ tay", "G56.0", "hội chứng ống cổ tay"),
    ("tê tay kiểu ống cổ tay", "G56.0", "hội chứng ống cổ tay"),
    ("bệnh đa xơ cứng", "G35", "đa xơ cứng"),
    ("xơ cứng rải rác", "G35", "đa xơ cứng"),
    ("nhược cơ", "G70.9", "nhược cơ"),
    ("yếu cơ mắt sụp mí", "G70.0", "nhược cơ mắt"),

    # === CHƯƠNG VII: BỆNH MẮT (H00-H59) ===
    ("cườm mắt", "H25.9", "đục thủy tinh thể tuổi già"),
    ("đục thủy tinh thể", "H25.9", "đục thủy tinh thể tuổi già"),
    ("tăng nhãn áp", "H40.9", "glôcôm"),
    ("cườm nước", "H40.9", "glôcôm"),
    ("glôcôm góc đóng", "H40.2", "glôcôm góc đóng"),
    ("viêm kết mạc", "H10.9", "viêm kết mạc"),
    ("đau mắt đỏ", "H10.9", "viêm kết mạc"),
    ("viêm màng bồ đào", "H20.9", "viêm màng bồ đào"),
    ("viêm giác mạc", "H16.9", "viêm giác mạc"),
    ("bong võng mạc", "H33.0", "bong võng mạc có rách"),

    # === CHƯƠNG VIII: BỆNH TAI VÀ XƯƠNG CHŨM (H60-H95) ===
    ("viêm tai giữa", "H66.9", "viêm tai giữa"),
    ("viêm tai giữa cấp", "H66.0", "viêm tai giữa cấp"),
    ("viêm tai giữa mạn tính", "H66.3", "viêm tai giữa mạn tính"),
    ("viêm tai xương chũm", "H70.9", "viêm xương chũm"),
    ("ù tai", "H93.1", "ù tai"),
    ("chóng mặt do tai trong", "H81.0", "bệnh ménière"),
    ("bệnh ménière", "H81.0", "bệnh ménière"),

    # === CHƯƠNG IX: BỆNH TUẦN HOÀN (I00-I99) ===
    ("thiếu máu cơ tim cục bộ", "I25.9", "bệnh tim thiếu máu cục bộ mạn tính"),
    ("đau thắt ngực", "I20.9", "đau thắt ngực"),
    ("suy tim ứ huyết", "I50.0", "suy tim sung huyết"),
    ("suy tim trái", "I50.1", "suy tim trái"),
    ("suy tim phải", "I50.0", "suy tim sung huyết"),
    ("rối loạn nhịp tim", "I49.9", "rối loạn nhịp tim không đặc hiệu"),
    ("bệnh cơ tim giãn", "I42.0", "bệnh cơ tim giãn"),
    ("viêm nội tâm mạc nhiễm khuẩn", "I33.0", "viêm nội tâm mạc nhiễm khuẩn"),
    ("viêm màng ngoài tim", "I31.9", "viêm màng ngoài tim"),
    ("giãn tĩnh mạch chân", "I83.9", "giãn tĩnh mạch chi dưới"),
    ("suy van tĩnh mạch", "I87.2", "suy van tĩnh mạch"),
    ("viêm tĩnh mạch huyết khối", "I80.9", "viêm tĩnh mạch và huyết khối tĩnh mạch"),
    ("huyết khối tĩnh mạch sâu", "I80.2", "huyết khối tĩnh mạch sâu chi dưới"),

    # === CHƯƠNG X: BỆNH HÔ HẤP (J00-J99) ===
    ("viêm mũi dị ứng", "J30.4", "viêm mũi dị ứng"),
    ("viêm mũi xoang", "J32.9", "viêm xoang mạn tính"),
    ("viêm họng cấp", "J02.9", "viêm họng cấp"),
    ("viêm amidan", "J03.9", "viêm amidan cấp"),
    ("viêm thanh quản", "J04.0", "viêm thanh quản cấp"),
    ("viêm phế quản cấp", "J20.9", "viêm phế quản cấp"),
    ("bệnh phổi tắc nghẽn mạn tính", "J44.9", "bệnh phổi tắc nghẽn mạn tính"),
    ("copd giai đoạn cuối", "J44.9", "bệnh phổi tắc nghẽn mạn tính"),
    ("giãn phế quản", "J47", "giãn phế quản"),
    ("viêm phổi kẽ", "J84.9", "bệnh phổi kẽ"),
    ("xơ phổi", "J84.1", "bệnh phổi kẽ có xơ hóa"),

    # === CHƯƠNG XI: BỆNH TIÊU HÓA (K00-K93) ===
    ("khó nuốt", "R13", "khó nuốt"),
    ("nuốt nghẹn nuốt vướng", "R13", "khó nuốt"),
    ("trào ngược dạ dày thực quản", "K21.9", "bệnh trào ngược dạ dày thực quản"),
    ("ợ nóng ợ chua", "K21.0", "bệnh trào ngược dạ dày thực quản có viêm thực quản"),
    ("viêm dạ dày cấp", "K29.1", "viêm dạ dày cấp"),
    ("đau bao tử", "K29.9", "viêm dạ dày tá tràng"),
    ("sỏi đường mật", "K80.5", "sỏi đường mật"),
    ("sỏi ống mật chủ", "K80.5", "sỏi đường mật"),
    ("viêm túi thừa đại tràng", "K57.3", "viêm túi thừa đại tràng"),
    ("táo bón mạn tính", "K59.0", "táo bón mạn tính"),
    ("hội chứng ruột kích thích", "K58.9", "hội chứng ruột kích thích"),
    ("viêm ruột thừa cấp", "K35.8", "viêm ruột thừa cấp"),
    ("thoát vị bẹn", "K40.9", "thoát vị bẹn"),
    ("thoát vị rốn", "K42.9", "thoát vị rốn"),
    ("nứt hậu môn", "K60.2", "nứt hậu môn"),
    ("trĩ nội", "K64.9", "bệnh trĩ"),
    ("bệnh trĩ", "K64.9", "bệnh trĩ"),

    # === CHƯƠNG XII: BỆNH DA LIỄU (L00-L99) ===
    ("chàm thể tạng", "L20.9", "viêm da dị ứng"),
    ("viêm da dị ứng", "L20.9", "viêm da dị ứng"),
    ("viêm da mủ", "L08.0", "viêm da mủ"),
    ("phát ban do thuốc", "L27.0", "phát ban do thuốc"),
    ("mề đay dị ứng", "L50.0", "mề đay dị ứng"),
    ("mụn nhọt", "L02.9", "áp xe da nhọt nhọt đầu đinh"),
    ("áp xe da", "L02.9", "áp xe da nhọt nhọt đầu đinh"),

    # === CHƯƠNG XIII: CƠ XƯƠNG KHỚP (M00-M99) ===
    ("đau lưng cấp", "M54.5", "đau thắt lưng"),
    ("đau lưng mạn", "M54.5", "đau thắt lưng"),
    ("đau lưng dưới", "M54.5", "đau thắt lưng"),
    ("thoái hóa đốt sống cổ", "M47.9", "thoái hóa cột sống"),
    ("thoái hóa đốt sống lưng", "M47.9", "thoái hóa cột sống"),
    ("viêm cột sống dính khớp", "M45", "viêm cột sống dính khớp"),
    ("viêm khớp thiếu niên", "M08.0", "viêm khớp thiếu niên"),
    ("loãng xương", "M81.9", "loãng xương"),
    ("xốp xương", "M81.9", "loãng xương"),
    ("gãy xương do loãng xương", "M80.9", "loãng xương có gãy xương bệnh lý"),

    # === CHƯƠNG XIV: TIẾT NIỆU SINH DỤC (N00-N99) ===
    ("nhiễm trùng tiểu", "N39.0", "nhiễm khuẩn đường tiểu"),
    ("viêm đường tiết niệu", "N39.0", "nhiễm khuẩn đường tiểu"),
    ("sỏi niệu quản", "N20.1", "sỏi niệu quản"),
    ("sỏi bàng quang", "N21.0", "sỏi bàng quang"),
    ("viêm bể thận cấp", "N10", "viêm bể thận cấp"),
    ("viêm bàng quang", "N30.9", "viêm bàng quang"),
    ("tiểu không tự chủ", "N39.4", "tiểu không tự chủ"),
    ("són tiểu khi gắng sức", "N39.3", "tiểu không tự chủ do gắng sức"),
    ("phì đại tuyến tiền liệt", "N40", "phì đại tuyến tiền liệt"),
    ("u xơ tuyến tiền liệt", "N40", "phì đại tuyến tiền liệt"),

    # === CHƯƠNG XV: SẢN PHỤ KHOA (O00-O99) ===
    ("tiền sản giật", "O14.9", "tiền sản giật"),
    ("nhiễm độc thai nghén", "O14.9", "tiền sản giật"),
    ("sản giật", "O15.9", "sản giật"),
    ("đái tháo đường thai kỳ", "O24.4", "đái tháo đường thai kỳ"),
    ("nhau tiền đạo", "O44.0", "nhau tiền đạo"),
    ("nhau bong non", "O45.9", "nhau bong non"),
    ("sảy thai tự nhiên", "O03.9", "sảy thai tự nhiên"),
    ("thai ngoài tử cung", "O00.9", "thai ngoài tử cung"),
    ("viêm vùng chậu nữ", "N73.9", "viêm vùng chậu nữ"),

    # === CHƯƠNG XVI: BỆNH SƠ SINH (P00-P96) ===
    ("vàng da sơ sinh", "P59.9", "vàng da sơ sinh"),
    ("vàng da bệnh lý sơ sinh", "P55.9", "bệnh tan máu sơ sinh"),
    ("suy hô hấp sơ sinh", "P22.0", "hội chứng suy hô hấp sơ sinh"),
    ("nhiễm trùng sơ sinh", "P36.9", "nhiễm trùng sơ sinh"),

    # === CHƯƠNG XVII: DỊ TẬT BẨM SINH (Q00-Q99) ===
    ("tim bẩm sinh", "Q24.9", "dị tật bẩm sinh tim"),
    ("hở hàm ếch", "Q35.9", "khe hở vòm miệng"),
    ("sứt môi hở hàm ếch", "Q37.9", "khe hở vòm miệng kèm khe hở môi"),
    ("trật khớp háng bẩm sinh", "Q65.0", "trật khớp háng bẩm sinh một bên"),

    # === CHƯƠNG XVIII: TRIỆU CHỨNG, DẤU HIỆU (R00-R99) ===
    ("sốt không rõ nguyên nhân", "R50.9", "sốt không rõ nguyên nhân"),
    ("đau ngực không do tim", "R07.4", "đau ngực không đặc hiệu"),
    ("đau bụng cấp", "R10.0", "bụng cấp"),
    ("xuất huyết tiêu hóa trên", "K92.2", "xuất huyết tiêu hóa"),
    ("xuất huyết tiêu hóa dưới", "K92.2", "xuất huyết tiêu hóa"),

    # === UNG BƯỚU (C00-D48) ===
    ("ung thư gan nguyên phát", "C22.9", "u ác gan"),
    ("ung thư dạ dày", "C16.9", "u ác dạ dày"),
    ("ung thư đại trực tràng", "C19", "u ác đại tràng"),
    ("ung thư phổi không tế bào nhỏ", "C34.9", "u ác phế quản hoặc phổi"),
    ("ung thư vú", "C50.9", "u ác vú"),
    ("ung thư tuyến tiền liệt", "C61", "u ác tuyến tiền liệt"),
    ("ung thư cổ tử cung", "C53.9", "u ác cổ tử cung"),
    ("bạch cầu cấp", "C95.0", "bệnh bạch cầu cấp"),
    ("ung thư máu", "C95.9", "bệnh bạch cầu"),

    # === RỐI LOẠN TÂM THẦN (F00-F99) ===
    ("tâm thần phân liệt thể Paranoid", "F20.0", "tâm thần phân liệt thể paranoid"),
    ("rối loạn lo âu lan tỏa", "F41.1", "rối loạn lo âu lan tỏa"),
    ("ám ảnh sợ khoảng trống", "F40.0", "rối loạn ám ảnh sợ khoảng trống"),
    ("rối loạn ám ảnh cưỡng chế", "F42.9", "rối loạn ám ảnh cưỡng chế"),
    ("trầm cảm nặng", "F32.9", "giai đoạn trầm cảm"),
    ("rối loạn hoảng sợ", "F41.0", "rối loạn hoảng sợ"),
    ("nghiện rượu mạn", "F10.2", "rối loạn tâm thần và hành vi do rượu hội chứng lệ thuộc"),
    ("rối loạn giấc ngủ", "G47.9", "rối loạn giấc ngủ"),
]

# Kiểm tra không trùng lặp với bộ 53 case cũ
def check_no_overlap_with_original(original_set):
    """Nhập PARAPHRASE_TEST_SET từ paraphrase_test_set.py, trả về list conflict."""
    orig_codes = {entry[1] for entry in original_set}
    conflicts = []
    for txt, code, can in PARAPHRASE_EXTENDED:
        if code in orig_codes:
            conflicts.append((txt, code, f"code already in original set"))
    return conflicts


if __name__ == "__main__":
    print(f"Extended set: {len(PARAPHRASE_EXTENDED)} cases")
    # Print chapter distribution
    from collections import Counter
    chapters = Counter()
    for _, code, _ in PARAPHRASE_EXTENDED:
        # Extract first letter/letter-group
        if code[0].isalpha():
            ch = code[0]
            if ch in 'ABCD' and len(code) > 1 and code[1].isdigit():
                pass  # C,D are neoplasms
            chapters[ch] += 1
    for ch, count in sorted(chapters.items()):
        print(f"  Chapter {ch}: {count} cases")