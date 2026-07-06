"""
Sinh data EHR giả lập theo đúng văn phong 3-section đã quan sát từ file mẫu
thật (có viết tắt y khoa, negation, historical, lỗi chính tả dính chữ...).

Cách làm: viết text với markup dạng [nội dung](TYPE), code tự động parse ra
text sạch (bỏ markup) + tính char offset chính xác — tránh đếm tay dễ sai.

QUAN TRỌNG:
- Đây là data TỰ VIẾT dựa trên hiểu biết y khoa phổ thông + quan sát CẤU TRÚC
  (không phải NỘI DUNG) từ 1 file mẫu đề bài đã cung cấp công khai trong lúc
  thảo luận — không phải từ 100 file test thật của cuộc thi.
- Vital sign packed-string (VD "VS98.3 12987 56 18 99RA") KHÔNG được gán nhãn
  entity ở đây — theo đúng quyết định kiến trúc đã thống nhất, phần này thuộc
  Phase 3 (rule-layer), không phải NER model.
"""
import re
import json

MARKUP_PATTERN = re.compile(r"\[([^\]]+)\]\((\w+)\)")


def parse_markup(marked_text: str):
    """Parse '[nội dung](TYPE)' -> (text sạch, list entities với char offset)."""
    entities = []
    clean_parts = []
    last_end = 0

    for m in MARKUP_PATTERN.finditer(marked_text):
        clean_parts.append(marked_text[last_end:m.start()])
        entity_text = m.group(1)
        entity_type = m.group(2)

        char_start = sum(len(p) for p in clean_parts)
        clean_parts.append(entity_text)
        char_end = char_start + len(entity_text)

        entities.append({"text": entity_text, "type": entity_type,
                          "position": [char_start, char_end]})
        last_end = m.end()

    clean_parts.append(marked_text[last_end:])
    clean_text = "".join(clean_parts)

    # Sanity check bắt buộc — nếu sai, lỗi sẽ lộ ngay khi chạy, không âm thầm
    for e in entities:
        s, en = e["position"]
        assert clean_text[s:en] == e["text"], (
            f"Offset lệch: '{clean_text[s:en]}' != '{e['text']}'"
        )

    return clean_text, entities


# ============================================================
# 20 đoạn văn EHR giả lập — phủ đa dạng pattern đã quan sát được
# từ cấu trúc file mẫu: negation, historical, viết tắt, lỗi chính
# tả dính chữ, câu dài phức hợp nhiều entity.
# ============================================================

RAW_NOTES = [
    # Tiền sử bệnh — thuốc + historical
    "Tiền sử dùng [amlodipine 5 mg po daily](THUỐC) và [aspirin 81 mg po daily](THUỐC) "
    "để điều trị [tăng huyết áp](BỆNH). Bệnh nhân có tiền sử [đái tháo đường type 2](BỆNH) "
    "đã 10 năm, hiện đang dùng [metformin 500 mg po bid](THUỐC).",

    # Negation nhiều pattern khác nhau
    "Bệnh nhân không [đau ngực](TRIỆU_CHỨNG), không [khó thở](TRIỆU_CHỨNG) khi nghỉ ngơi. "
    "Không ghi nhận [sốt](TRIỆU_CHỨNG) hay [ớn lạnh](TRIỆU_CHỨNG) trong 3 ngày qua.",

    # Lỗi chính tả dính chữ (mô phỏng đúng pattern đã thấy trong file mẫu)
    "Khi thăm khám, bệnh nhân than phiền [cảm giáckhó chịu](TRIỆU_CHỨNG) vùng thượng vị, "
    "kèm theo [buồn nônnhẹ](TRIỆU_CHỨNG) sau ăn.",

    # Câu dài, nhiều entity liên tiếp
    "Bệnh nhân nhập viện vì [đau ngực trái](TRIỆU_CHỨNG) lan ra sau lưng, kèm "
    "[vã mồ hôi](TRIỆU_CHỨNG), [buồn nôn](TRIỆU_CHỨNG), nghi ngờ [nhồi máu cơ tim cấp](BỆNH), "
    "đã được xử trí bằng [aspirin 325 mg](THUỐC) và [nitroglycerin ngậm dưới lưỡi](THUỐC).",

    # Viết tắt y khoa dày đặc
    "Kê đơn [clonazepam 0.5 mg po qam:prn](THUỐC) điều trị [lo âu](TRIỆU_CHỨNG), "
    "và [docusate sodium 100 mg po bid](THUỐC) phòng [táo bón](TRIỆU_CHỨNG).",

    # Planned/tương lai — không phải historical
    "Lên lịch tái khám với bác sĩ tim mạch, chỉ định [siêu âm tim qua thành ngực](THỦ_THUẬT) "
    "vào tuần tới để đánh giá [hở van hai lá](BỆNH) nghi ngờ trên lâm sàng.",

    # Bệnh mạn tính + biến chứng
    "Bệnh nhân có [xơ gan do rượu](BỆNH) giai đoạn mất bù, biến chứng [cổ trướng](TRIỆU_CHỨNG) "
    "và [hội chứng não gan](BỆNH) độ 2, đã được điều trị bằng [lactulose 30 ml po tid](THUỐC).",

    # Triệu chứng hô hấp phối hợp
    "Ho khan kéo dài 2 tuần, kèm [khó thở khi gắng sức](TRIỆU_CHỨNG), nghe phổi có "
    "[ran nổ hai đáy phổi](TRIỆU_CHỨNG), X-quang nghi [viêm phổi thùy dưới phải](BỆNH), "
    "chỉ định [ceftriaxone 1g iv q24h](THUỐC).",

    # Kết quả xét nghiệm
    "[Troponin T](KẾT_QUẢ_XÉT_NGHIỆM) tăng nhẹ ở mức [0.08 ng/mL](KẾT_QUẢ_XÉT_NGHIỆM), "
    "[ECG](KẾT_QUẢ_XÉT_NGHIỆM) ghi nhận [ST chênh xuống ở chuyển đạo V4-V6](KẾT_QUẢ_XÉT_NGHIỆM).",

    # Dị ứng + phản ứng thuốc
    "Bệnh nhân có tiền sử dị ứng [penicillin](THUỐC) gây [phát ban toàn thân](TRIỆU_CHỨNG), "
    "hiện được đổi sang [clindamycin 300 mg po q8h](THUỐC).",

    # Suy thận mạn + điều chỉnh liều
    "Do [suy thận mạn giai đoạn 4](BỆNH), điều chỉnh liều [enoxaparin](THUỐC) theo "
    "độ thanh thải creatinin, tránh dùng [NSAID](THUỐC) do nguy cơ [tổn thương thận cấp](TRIỆU_CHỨNG).",

    # Rối loạn nhịp tim
    "Holter ghi nhận [rung nhĩ đáp ứng thất nhanh](BỆNH) từng cơn, đã bắt đầu "
    "[metoprolol succinate 50 mg po daily](THUỐC), theo dõi thêm [đánh trống ngực](TRIỆU_CHỨNG).",

    # Sản khoa
    "Sản phụ 32 tuần, có [tiền sản giật](BỆNH) nhẹ, huyết áp dao động, không "
    "[phù chân](TRIỆU_CHỨNG) rõ, không [nhìn mờ](TRIỆU_CHỨNG) hay [đau đầu dữ dội](TRIỆU_CHỨNG).",

    # Nhi khoa
    "Trẻ 3 tuổi sốt cao 39.5 độ kèm [phát ban dạng sởi](TRIỆU_CHỨNG), [ho](TRIỆU_CHỨNG) và "
    "[chảy nước mũi](TRIỆU_CHỨNG), nghi [sởi](BỆNH), chưa tiêm phòng đầy đủ.",

    # Tâm thần
    "Bệnh nhân có tiền sử [trầm cảm nặng](BỆNH) tái phát, hiện dùng "
    "[sertraline 100 mg po daily](THUỐC), không ghi nhận [ý tưởng tự sát](TRIỆU_CHỨNG).",

    # Cơ xương khớp
    "Đau khớp gối hai bên tăng khi vận động, X-quang cho thấy [thoái hoá khớp gối](BỆNH) "
    "mức độ vừa, giảm đau bằng [paracetamol 500 mg po q6h:prn](THUỐC).",

    # Nội tiết
    "Xét nghiệm cho thấy [TSH tăng cao](KẾT_QUẢ_XÉT_NGHIỆM), phù hợp [suy giáp nguyên phát](BỆNH), "
    "bắt đầu bổ sung [levothyroxine 50 mcg po daily](THUỐC), tái khám sau 6 tuần.",

    # Nhiễm trùng huyết
    "Bệnh nhân vào viện trong tình trạng [sốc nhiễm trùng](BỆNH), [huyết áp tụt](TRIỆU_CHỨNG), "
    "[sốt cao rét run](TRIỆU_CHỨNG), cấy máu dương tính, dùng [meropenem 1g iv q8h](THUỐC).",

    # Tiêu hoá
    "Nội soi dạ dày phát hiện [loét hành tá tràng](BỆNH), điều trị bằng "
    "[omeprazole 20 mg po bid](THUỐC) và [amoxicillin 1g po bid](THUỐC) phác đồ diệt HP.",

    # Đa bệnh lý phối hợp (case phức tạp, nhiều entity dày đặc)
    "Bệnh nhân cao tuổi có [đái tháo đường type 2](BỆNH), [tăng huyết áp](BỆNH), "
    "[suy tim EF giảm](BỆNH) đang dùng [furosemide 40 mg po daily](THUỐC), "
    "[losartan 50 mg po daily](THUỐC), nhập viện vì [phù chân tiến triển](TRIỆU_CHỨNG) "
    "và [khó thở khi nằm](TRIỆU_CHỨNG) 3 ngày nay, không [đau ngực](TRIỆU_CHỨNG).",
]


def build_synthetic_dataset(output_path: str):
    records = []
    for raw in RAW_NOTES:
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

    print(f"Đã sinh {len(records)} đoạn văn EHR giả lập")
    print(f"Phân bố type: {dict(type_counts)}")
    print(f"Đã lưu: {output_path}")
    return records


if __name__ == "__main__":
    build_synthetic_dataset("converted_data/synthetic_ehr.jsonl")
