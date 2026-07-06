"""
Batch 3 — mở rộng mạnh, ưu tiên SỐ LƯỢNG + ĐA DẠNG để synthetic EHR có trọng
số đáng kể so với 7,406 câu ViMQ/PhoNER (văn phong khác hẳn đề bài thật).

Chiến lược đa dạng hoá lần này:
- Nhiều biến thể cue historical/negated/planned khác nhau (không lặp công thức)
- Nhiều cách viết tắt + kết hợp liều/route/frequency khác nhau cho THUỐC
- Case entity dài, phức hợp (nhiều từ) lẫn case ngắn (1 từ)
- Câu có nhiều lỗi chính tả/dính chữ hơn (đặc trưng đã quan sát rất rõ ở data thật)
- Nhiều loại vital sign packed-string khác nhau, luôn không gán nhãn
"""
import json
from generate_synthetic_ehr import parse_markup


RAW_NOTES_BATCH3 = [
    # Historical với cue khác nhau
    "Bệnh nhân có tiền sử [nhồi máu cơ tim](BỆNH) cách đây 5 năm, đã đặt "
    "[stent động mạch vành](THỦ_THUẬT), hiện duy trì [clopidogrel 75mg po daily](THUỐC) "
    "và [atorvastatin 40mg po daily](THUỐC).",

    "Trước đây từng nhập viện vì [viêm tụy cấp](BỆNH) do rượu, đã cai rượu được "
    "2 năm, không còn [đau bụng thượng vị](TRIỆU_CHỨNG) tái phát.",

    "Ghi nhận trong hồ sơ cũ: đã điều trị [lao phổi](BỆNH) đủ phác đồ 6 tháng, "
    "hoàn thành điều trị năm 2023, hiện không còn [ho ra máu](TRIỆU_CHỨNG).",

    # Negated đa dạng cue
    "Phủ nhận [đau đầu dữ dội](TRIỆU_CHỨNG), phủ nhận [nhìn đôi](TRIỆU_CHỨNG), "
    "chưa từng có [co giật](TRIỆU_CHỨNG) trước đây.",

    "Không có biểu hiện [vàng da](TRIỆU_CHỨNG), không [ngứa toàn thân](TRIỆU_CHỨNG), "
    "gan lách không sờ thấy.",

    "Loại trừ [viêm ruột thừa](BỆNH) do không có [đau hố chậu phải](TRIỆU_CHỨNG) "
    "điển hình, không [phản ứng thành bụng](TRIỆU_CHỨNG).",

    # Planned đa dạng cue
    "Sẽ hội chẩn ngoại khoa để cân nhắc [phẫu thuật thay khớp háng](THỦ_THUẬT), "
    "hiện tạm thời giảm đau bằng [tramadol 50mg po q8h:prn](THUỐC).",

    "Kế hoạch theo dõi thêm 48 giờ trước khi quyết định [nội soi mật tụy ngược "
    "dòng](THỦ_THUẬT), hiện tình trạng [đau bụng](TRIỆU_CHỨNG) đã giảm phần nào.",

    "Đề nghị chuyển tuyến trên để thực hiện [chụp cộng hưởng từ cột sống](THỦ_THUẬT), "
    "chưa có kết quả tại thời điểm ra viện.",

    # Lỗi chính tả / dính chữ nhiều hơn
    "Bệnh nhân nói bị [đau bụngâm ỉ](TRIỆU_CHỨNG) vùng thượng vị.",
    "Khai thác thêm: kèm theo [mệt mỏitoàn thân](TRIỆU_CHỨNG) và "
    "[chán ăn](TRIỆU_CHỨNG) rõ rệt trong 1 tuần.",

    # Câu ngắn, entity 1 từ
    "Bệnh nhân [sốt](TRIỆU_CHỨNG), [ho](TRIỆU_CHỨNG), [mệt](TRIỆU_CHỨNG). "
    "Chẩn đoán [cúm](BỆNH). Dùng [oseltamivir](THUỐC).",

    # Entity dài, phức hợp
    "Chẩn đoán xác định: [bệnh thận mạn giai đoạn cuối do đái tháo đường]"
    "(BỆNH), đang lọc máu chu kỳ 3 lần/tuần, dùng "
    "[erythropoietin tiêm dưới da mỗi tuần một lần](THUỐC).",

    # Vital sign kiểu khác, không gán nhãn
    "Sinh hiệu ghi nhận: VS37.2 11070 78 16 98RA, bệnh nhân than phiền "
    "[hồi hộp](TRIỆU_CHỨNG) nhẹ, không [ngất](TRIỆU_CHỨNG).",

    "Vào khoa: VS36.8 10565 102 24 94RA2L, thở oxy 2 lít/phút qua canula, "
    "vẫn còn [khó thở](TRIỆU_CHỨNG) khi gắng sức nhẹ.",

    # Đa entity cùng loại lặp lại (giống pattern thật)
    "Ngày 1: [sốt](TRIỆU_CHỨNG) 38.5 độ. Ngày 2: [sốt](TRIỆU_CHỨNG) 39.2 độ kèm "
    "[rét run](TRIỆU_CHỨNG). Ngày 3: [sốt](TRIỆU_CHỨNG) giảm còn 37.8 độ sau dùng "
    "[paracetamol](THUỐC), [rét run](TRIỆU_CHỨNG) không còn tái phát.",

    # Dị ứng thuốc phức tạp
    "Dị ứng ghi nhận: [sulfamethoxazole-trimethoprim](THUỐC) gây "
    "[phát ban](TRIỆU_CHỨNG) và [khó thở](TRIỆU_CHỨNG), tránh dùng nhóm sulfa, "
    "hiện đổi sang [ciprofloxacin 500mg po bid](THUỐC).",

    # Sốt xuất huyết — pattern phổ biến ở VN
    "Bệnh nhân [sốt cao liên tục](TRIỆU_CHỨNG) 4 ngày, [đau cơ khớp](TRIỆU_CHỨNG), "
    "[phát ban dạng chấm xuất huyết](TRIỆU_CHỨNG), xét nghiệm "
    "[tiểu cầu giảm còn 85000/mm3](KẾT_QUẢ_XÉT_NGHIỆM), chẩn đoán "
    "[sốt xuất huyết Dengue có dấu hiệu cảnh báo](BỆNH).",

    # Bệnh nền phức tạp + thuốc nhiều loại
    "Bệnh nhân có [tăng huyết áp](BỆNH), [rối loạn lipid máu](BỆNH), "
    "[gút mạn](BỆNH), đang dùng đồng thời [amlodipine 5mg](THUỐC), "
    "[rosuvastatin 20mg](THUỐC), [allopurinol 300mg](THUỐC) mỗi ngày.",

    # Xét nghiệm nhiều chỉ số
    "Kết quả xét nghiệm: [creatinine 2.1 mg/dL](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[eGFR 32 mL/phút](KẾT_QUẢ_XÉT_NGHIỆM), [kali máu 5.8 mmol/L](KẾT_QUẢ_XÉT_NGHIỆM), "
    "phù hợp [suy thận cấp trên nền mạn](BỆNH).",

    # Thần kinh — động kinh
    "Bệnh nhân [co giật toàn thân](TRIỆU_CHỨNG) kéo dài 3 phút, sau đó "
    "[lú lẫn](TRIỆU_CHỨNG) khoảng 10 phút, tiền sử [động kinh](BỆNH) đã 5 năm, "
    "hiện dùng [valproate 500mg po bid](THUỐC), gần đây tự ý ngưng thuốc.",

    # Hô hấp cấp cứu
    "Bệnh nhân [suy hô hấp cấp](BỆNH), [tím môi đầu chi](TRIỆU_CHỨNG), "
    "SpO2 giảm còn 82% khí trời, đặt nội khí quản cấp cứu, thở máy hỗ trợ, "
    "nghi [phù phổi cấp](BỆNH) trên nền [suy tim](BỆNH).",

    # Nhi - sốt co giật
    "Trẻ 18 tháng [sốt cao đột ngột](TRIỆU_CHỨNG) 40 độ kèm "
    "[co giật toàn thân 2 phút](TRIỆU_CHỨNG), sau co giật trẻ tỉnh, chẩn đoán "
    "[sốt cao co giật đơn giản](BỆNH), hạ sốt bằng "
    "[paracetamol đường hậu môn](THUỐC).",

    # Ngoại khoa cấp cứu
    "Đau bụng dữ dội vùng hố chậu phải, [phản ứng thành bụng](TRIỆU_CHỨNG) rõ, "
    "bạch cầu tăng cao, chẩn đoán [viêm ruột thừa cấp](BỆNH), chỉ định "
    "[phẫu thuật cắt ruột thừa nội soi](THỦ_THUẬT) cấp cứu.",

    # Tâm thần - loạn thần
    "Bệnh nhân có [hoang tưởng bị hại](TRIỆU_CHỨNG), [ảo thanh](TRIỆU_CHỨNG) ra lệnh, "
    "tiền sử [tâm thần phân liệt](BỆNH) đã 8 năm, hiện dùng "
    "[risperidone 2mg po daily](THUỐC), gia đình báo gần đây bỏ thuốc 2 tuần.",
]


def extend_synthetic_dataset(output_path: str):
    records = []
    for raw in RAW_NOTES_BATCH3:
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

    print(f"Đã sinh thêm {len(records)} đoạn văn EHR batch 3")
    print(f"Phân bố type: {dict(type_counts)}")
    print(f"Đã lưu: {output_path}")
    return records


if __name__ == "__main__":
    extend_synthetic_dataset("converted_data/synthetic_ehr_batch3.jsonl")
