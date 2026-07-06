"""
Batch 5 — tập trung RIÊNG cho KẾT_QUẢ_XÉT_NGHIỆM, vì đây là type có ít data
nhất (chỉ 8 câu/14 entity từ các batch trước). Mục tiêu: đủ số lượng để chia
3 tập (train/dev/test) đều có mẫu, không phải xoay vòng vài câu ít ỏi.

Phủ đa dạng loại xét nghiệm: huyết học, sinh hoá, vi sinh, chẩn đoán hình ảnh,
thăm dò chức năng (ECG, siêu âm) — để model học được PATTERN NGỮ CẢNH chung
(số liệu + đơn vị + tên chỉ số) chứ không chỉ học thuộc vài ví dụ cụ thể.
"""
import json
from generate_synthetic_ehr import parse_markup


RAW_NOTES_BATCH5 = [
    "Công thức máu cho thấy [bạch cầu tăng 15.000/mm3](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[CRP tăng cao 80 mg/L](KẾT_QUẢ_XÉT_NGHIỆM), phù hợp tình trạng nhiễm trùng.",

    "Sinh hoá máu: [glucose máu 6.5 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[ure 8.2 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), [creatinin 95 umol/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "trong giới hạn bình thường.",

    "Chức năng gan: [AST 120 U/L](KẾT_QUẢ_XÉT_NGHIỆM), [ALT 145 U/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "tăng nhẹ so với bình thường, theo dõi thêm.",

    "Điện giải đồ: [natri máu 138 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[kali máu 4.2 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), trong giới hạn cho phép.",

    "Khí máu động mạch: [pH 7.32](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[PaCO2 55 mmHg](KẾT_QUẢ_XÉT_NGHIỆM), phù hợp toan hô hấp.",

    "[Siêu âm bụng](KẾT_QUẢ_XÉT_NGHIỆM) ghi nhận gan nhiễm mỡ độ 2, không thấy "
    "sỏi mật, lách không to.",

    "[X-quang ngực thẳng](KẾT_QUẢ_XÉT_NGHIỆM) cho thấy tim to nhẹ, phổi không "
    "thấy tổn thương rõ.",

    "[CT scan sọ não](KẾT_QUẢ_XÉT_NGHIỆM) không phát hiện xuất huyết hay tổn "
    "thương nhu mô não cấp tính.",

    "[Điện tâm đồ](KẾT_QUẢ_XÉT_NGHIỆM) ghi nhận nhịp xoang đều, tần số 78 "
    "lần/phút, không thấy bất thường ST-T.",

    "[Siêu âm tim](KẾT_QUẢ_XÉT_NGHIỆM) cho thấy EF 55%, chức năng thất trái "
    "trong giới hạn bình thường, không hở van đáng kể.",

    "Cấy máu 2 mẫu: [dương tính với Staphylococcus aureus](KẾT_QUẢ_XÉT_NGHIỆM), "
    "nhạy với vancomycin.",

    "Tổng phân tích nước tiểu: [protein niệu dương tính 2+](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[bạch cầu niệu dương tính](KẾT_QUẢ_XÉT_NGHIỆM), nghi nhiễm trùng tiết niệu.",

    "Xét nghiệm đông máu: [PT kéo dài 18 giây](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[INR 1.6](KẾT_QUẢ_XÉT_NGHIỆM), cần điều chỉnh liều kháng đông.",

    "Chỉ số mỡ máu: [cholesterol toàn phần 6.8 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[LDL 4.2 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), tăng cao cần điều trị.",

    "Xét nghiệm tuyến giáp: [TSH 8.5 mIU/L](KẾT_QUẢ_XÉT_NGHIỆM), tăng cao, "
    "phù hợp suy giáp.",

    "[Nội soi dạ dày](KẾT_QUẢ_XÉT_NGHIỆM) phát hiện ổ loét hành tá tràng "
    "kích thước 0.8cm, test HP dương tính.",

    "Marker tim: [Troponin I 0.15 ng/mL](KẾT_QUẢ_XÉT_NGHIỆM), tăng nhẹ, "
    "theo dõi lại sau 6 giờ.",

    "[MRI cột sống thắt lưng](KẾT_QUẢ_XÉT_NGHIỆM) cho thấy thoát vị đĩa đệm "
    "L4-L5 chèn ép rễ thần kinh nhẹ.",

    "Xét nghiệm HbA1c cho kết quả [7.8%](KẾT_QUẢ_XÉT_NGHIỆM), phản ánh kiểm "
    "soát đường huyết chưa tốt trong 3 tháng qua.",

    "Định lượng D-dimer: [850 ng/mL](KẾT_QUẢ_XÉT_NGHIỆM), tăng cao, cần loại "
    "trừ thuyên tắc phổi.",

    "Xét nghiệm nước tiểu 24 giờ: [protein niệu 3.5g/24h](KẾT_QUẢ_XÉT_NGHIỆM), "
    "phù hợp hội chứng thận hư.",

    "[Siêu âm doppler tĩnh mạch chi dưới](KẾT_QUẢ_XÉT_NGHIỆM) phát hiện huyết "
    "khối tĩnh mạch sâu đoạn khoeo phải.",

    "Xét nghiệm khí máu: [SpO2 88%](KẾT_QUẢ_XÉT_NGHIỆM) khi thở khí trời, "
    "cải thiện lên 95% khi thở oxy 3 lít/phút.",

    "Kết quả giải phẫu bệnh sinh thiết: [u tuyến lành tính](KẾT_QUẢ_XÉT_NGHIỆM), "
    "không thấy tế bào ác tính.",

    "[Đo chức năng hô hấp](KẾT_QUẢ_XÉT_NGHIỆM) cho thấy FEV1 giảm còn 60% giá "
    "trị dự đoán, phù hợp rối loạn thông khí tắc nghẽn.",

    "Xét nghiệm CA-125: [tăng cao 180 U/mL](KẾT_QUẢ_XÉT_NGHIỆM), cần tầm soát "
    "thêm khối u buồng trứng.",

    "[Chụp mạch vành](KẾT_QUẢ_XÉT_NGHIỆM) phát hiện hẹp 70% động mạch vành "
    "phải đoạn gần, chỉ định can thiệp.",

    "Xét nghiệm PSA: [4.5 ng/mL](KẾT_QUẢ_XÉT_NGHIỆM), hơi cao so với tuổi, "
    "theo dõi thêm 3 tháng.",

    "Amylase máu tăng [450 U/L](KẾT_QUẢ_XÉT_NGHIỆM), lipase máu tăng "
    "[600 U/L](KẾT_QUẢ_XÉT_NGHIỆM), phù hợp viêm tụy cấp.",

    "[Đo mật độ xương DEXA](KẾT_QUẢ_XÉT_NGHIỆM) cho chỉ số T-score -2.8, "
    "phù hợp chẩn đoán loãng xương.",
]


def build_batch5(output_path: str):
    records = []
    for raw in RAW_NOTES_BATCH5:
        clean_text, entities = parse_markup(raw)
        records.append({"text": clean_text, "entities": entities})

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from collections import Counter
    type_counts = Counter()
    for rec in records:
        for e in rec["entities"]:
            type_counts[e["type"]] += 1
    print(f"Đã sinh {len(records)} câu batch 5 (tập trung KẾT_QUẢ_XÉT_NGHIỆM)")
    print(f"Phân bố type: {dict(type_counts)}")
    return records


if __name__ == "__main__":
    build_batch5("converted_data/synthetic_ehr_batch5_lab.jsonl")
