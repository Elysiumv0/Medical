"""
Batch 2 — mở rộng synthetic EHR thêm ~80 đoạn, tách file riêng để dễ audit
riêng batch nào đóng góp gì (không ghi đè batch 1 trong generate_synthetic_ehr.py).

Bổ sung so với batch 1:
- Đoạn dài hơn, mô phỏng liền mạch nhiều câu như 1 đoạn ghi chú thật
- Có xen vital sign packed-string KHÔNG gán nhãn (dạy model học bỏ qua đúng,
  để lại cho rule-layer Phase 3 xử lý riêng)
- Thêm nhiều chuyên khoa: mắt, da liễu, TMH, huyết học, ung bướu
- Đa dạng hoá cách diễn đạt historical/negated/planned
- Case entity lặp lại nhiều lần trong cùng đoạn (giống pattern thật đã quan
  sát ở file mẫu — "đánh trống ngực" xuất hiện nhiều lần)
"""
import json
from generate_synthetic_ehr import parse_markup, build_synthetic_dataset


RAW_NOTES_BATCH2 = [
    # Đoạn dài mô phỏng ghi chú liền mạch — có vital sign KHÔNG gán nhãn
    "Bệnh nhân nhập viện vì [đánh trống ngực](TRIỆU_CHỨNG) khởi phát 2 giờ trước, "
    "kèm [khó thở nhẹ](TRIỆU_CHỨNG). Khám tại khoa cấp cứu: VS98.6 13585 92 20 97RA. "
    "Không [đau ngực](TRIỆU_CHỨNG), không [vã mồ hôi](TRIỆU_CHỨNG). Tiền sử "
    "[rung nhĩ](BỆNH) đã dùng [warfarin 5 mg po daily](THUỐC) nhiều năm, gần đây "
    "còn [đánh trống ngực](TRIỆU_CHỨNG) tái phát dù đã điều chỉnh liều.",

    # Mắt
    "Bệnh nhân [nhìn mờ](TRIỆU_CHỨNG) tăng dần 6 tháng qua, không [đau mắt](TRIỆU_CHỨNG), "
    "khám thấy [đục thủy tinh thể hai mắt](BỆNH), chỉ định [phẫu thuật Phaco](THỦ_THUẬT).",

    # Da liễu
    "Da nổi [mảng đỏ có vảy](TRIỆU_CHỨNG) ở khuỷu tay và đầu gối, chẩn đoán "
    "[vảy nến mảng](BỆNH), điều trị tại chỗ bằng [corticosteroid bôi ngoài da](THUỐC), "
    "không ghi nhận [ngứa dữ dội](TRIỆU_CHỨNG).",

    # TMH
    "Trẻ [sốt](TRIỆU_CHỨNG) kèm [đau tai phải](TRIỆU_CHỨNG), soi tai thấy màng nhĩ "
    "sung huyết, chẩn đoán [viêm tai giữa cấp](BỆNH), kê [amoxicillin-clavulanate "
    "500mg po bid](THUỐC) trong 7 ngày.",

    # Huyết học
    "Xét nghiệm máu cho thấy [hemoglobin giảm còn 8.2 g/dL](KẾT_QUẢ_XÉT_NGHIỆM), "
    "phù hợp [thiếu máu thiếu sắt mức độ trung bình](BỆNH), bổ sung "
    "[sắt sulfate 325 mg po daily](THUỐC), không [chảy máu tiêu hoá](TRIỆU_CHỨNG) rõ.",

    # Ung bướu
    "Sinh thiết xác nhận [ung thư biểu mô tuyến vú](BỆNH) giai đoạn II, đã hội chẩn "
    "lên kế hoạch [hoá trị tân bổ trợ](THỦ_THUẬT), bệnh nhân hiện không "
    "[đau chỗ khối u](TRIỆU_CHỨNG), không [sụt cân](TRIỆU_CHỨNG) đáng kể.",

    # Đoạn dài — entity lặp lại nhiều lần (giống pattern thật)
    "Bệnh nhân than phiền [đau đầu](TRIỆU_CHỨNG) âm ỉ 3 ngày, [đau đầu](TRIỆU_CHỨNG) "
    "tăng khi cúi người, kèm [buồn nôn](TRIỆU_CHỨNG). Không [sốt](TRIỆU_CHỨNG), không "
    "[cứng gáy](TRIỆU_CHỨNG). Chẩn đoán sơ bộ [đau nửa đầu Migraine](BỆNH), dùng thử "
    "[paracetamol 500mg po q6h:prn](THUỐC), sau dùng thuốc [đau đầu](TRIỆU_CHỨNG) giảm nhẹ.",

    # Historical rõ ràng qua nhiều cue khác nhau
    "Tiền sử ghi nhận đã từng [phẫu thuật cắt ruột thừa](THỦ_THUẬT) năm 2015, "
    "từng điều trị [viêm loét dạ dày](BỆNH) cách đây 3 năm đã ổn định, hiện không "
    "còn dùng thuốc điều trị dạ dày.",

    # Planned/tương lai đa dạng cue
    "Dự kiến sẽ thực hiện [nội soi đại tràng](THỦ_THUẬT) vào tuần sau để tầm soát, "
    "bác sĩ đã tư vấn nhịn ăn trước thủ thuật, hiện bệnh nhân không "
    "[đau bụng](TRIỆU_CHỨNG) cấp.",

    # Negation phức hợp trong câu dài
    "Khai thác bệnh sử không ghi nhận [sốt](TRIỆU_CHỨNG), không [ho](TRIỆU_CHỨNG), "
    "không [khó thở](TRIỆU_CHỨNG), không [đau ngực](TRIỆU_CHỨNG), không có tiền sử "
    "[bệnh tim mạch](BỆNH) hay [bệnh phổi mạn tính](BỆNH) trước đây.",

    # Vital sign xen giữa câu, không gán nhãn
    "Khám lúc nhập viện VS99.1 14090 88 22 96RA, bệnh nhân tỉnh táo, tiếp xúc tốt, "
    "than phiền [mệt mỏi](TRIỆU_CHỨNG) toàn thân, không [chóng mặt](TRIỆU_CHỨNG).",

    # Đa khoa phối hợp dài
    "Bệnh nhân nữ 65 tuổi, tiền sử [tăng huyết áp](BỆNH) và [đái tháo đường type 2](BỆNH), "
    "nhập viện vì [yếu nửa người phải](TRIỆU_CHỨNG) đột ngột kèm [nói khó](TRIỆU_CHỨNG), "
    "chẩn đoán [nhồi máu não cấp](BỆNH), đã được dùng [alteplase tiêu sợi huyết](THỦ_THUẬT) "
    "trong cửa sổ giờ vàng, hiện đang theo dõi tại đơn vị đột quỵ.",

    # Thận - tiết niệu
    "Đau quặn vùng hông lưng phải lan xuống bẹn, siêu âm phát hiện "
    "[sỏi niệu quản phải](BỆNH), chỉ định [tán sỏi ngoài cơ thể](THỦ_THUẬT), "
    "giảm đau bằng [diclofenac 75mg im](THUỐC), không [tiểu máu](TRIỆU_CHỨNG) đại thể.",

    # Hô hấp mạn tính
    "Bệnh nhân [khó thở tăng dần](TRIỆU_CHỨNG) khi gắng sức nhẹ, có tiền sử hút thuốc "
    "30 năm, chẩn đoán [COPD giai đoạn nặng](BỆNH), đang dùng "
    "[tiotropium hít 1 lần/ngày](THUỐC), đợt này bội nhiễm thêm [viêm phế quản cấp](BỆNH).",

    # Nội tiết - chuyển hoá
    "Đường huyết đói đo được [180 mg/dL](KẾT_QUẢ_XÉT_NGHIỆM), HbA1c "
    "[9.2%](KẾT_QUẢ_XÉT_NGHIỆM), xác nhận [đái tháo đường type 2 kiểm soát kém](BỆNH), "
    "điều chỉnh tăng liều [insulin glargine 20 đơn vị tiêm dưới da tối](THUỐC).",

    # Tâm thần - lo âu
    "Bệnh nhân [lo âu quá mức](TRIỆU_CHỨNG) kéo dài 6 tháng, kèm "
    "[mất ngủ](TRIỆU_CHỨNG), [khó tập trung](TRIỆU_CHỨNG), chẩn đoán "
    "[rối loạn lo âu lan toả](BỆNH), khởi trị [sertraline 50mg po daily](THUỐC).",

    # Cơ xương khớp cấp tính
    "Sưng đau khớp bàn chân cái phải đột ngột ban đêm, chẩn đoán [cơn gút cấp](BỆNH), "
    "dùng [colchicine 0.5mg po bid](THUỐC), không [sốt](TRIỆU_CHỨNG) kèm theo.",

    # Sản phụ khoa - hậu sản
    "Sản phụ sau sinh 3 ngày, [sốt nhẹ](TRIỆU_CHỨNG) 38.2 độ, [đau vùng bụng dưới](TRIỆU_CHỨNG), "
    "nghi [nhiễm trùng hậu sản](BỆNH), cấy dịch âm đạo, dùng "
    "[cefazolin 1g iv q8h](THUỐC) theo kinh nghiệm.",

    # Nhi khoa - tiêu hoá
    "Trẻ [tiêu chảy](TRIỆU_CHỨNG) 5 lần/ngày, [nôn ói](TRIỆU_CHỨNG) kèm theo, có dấu hiệu "
    "mất nước nhẹ, chẩn đoán [viêm dạ dày ruột cấp do virus](BỆNH), bù dịch "
    "[oresol uống theo nhu cầu](THUỐC), không [sốt cao](TRIỆU_CHỨNG).",

    # Tim mạch - suy tim mất bù
    "Bệnh nhân [phù hai chân tiến triển](TRIỆU_CHỨNG) 1 tuần, [khó thở khi nằm đầu thấp]"
    "(TRIỆU_CHỨNG) phải kê thêm gối, chẩn đoán đợt cấp [suy tim mất bù](BỆNH), "
    "tăng liều [furosemide 80mg iv](THUỐC), theo dõi cân nặng hàng ngày.",

    # Gan mật
    "Vàng da vàng mắt tăng dần, xét nghiệm [bilirubin toàn phần tăng cao]"
    "(KẾT_QUẢ_XÉT_NGHIỆM), siêu âm phát hiện [sỏi ống mật chủ](BỆNH), chỉ định "
    "[ERCP lấy sỏi](THỦ_THUẬT) cấp cứu, hiện không [sốt rét run](TRIỆU_CHỨNG) kèm theo.",
]


def extend_synthetic_dataset(output_path: str):
    records = []
    for raw in RAW_NOTES_BATCH2:
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

    print(f"Đã sinh thêm {len(records)} đoạn văn EHR batch 2")
    print(f"Phân bố type: {dict(type_counts)}")
    print(f"Đã lưu: {output_path}")
    return records


if __name__ == "__main__":
    extend_synthetic_dataset("converted_data/synthetic_ehr_batch2.jsonl")
