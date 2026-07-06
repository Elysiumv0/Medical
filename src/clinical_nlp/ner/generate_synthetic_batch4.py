"""Batch 4 — batch cuối, đẩy tổng synthetic lên gần 100 câu."""
import json
from generate_synthetic_ehr import parse_markup

RAW_NOTES_BATCH4 = [
    "Bệnh nhân [đau thắt lưng lan xuống chân trái](TRIỆU_CHỨNG) 2 tháng nay, "
    "MRI cho thấy [thoát vị đĩa đệm L4-L5](BỆNH), điều trị bảo tồn bằng "
    "[gabapentin 300mg po tid](THUỐC) và [vật lý trị liệu](THỦ_THUẬT).",

    "Tiền sử gia đình có [ung thư đại tràng](BỆNH), bệnh nhân được chỉ định "
    "[nội soi đại tràng tầm soát](THỦ_THUẬT) định kỳ, hiện không "
    "[đi cầu ra máu](TRIỆU_CHỨNG).",

    "Bệnh nhân [khó nuốt](TRIỆU_CHỨNG) tiến triển 3 tháng, sụt 5kg, nội soi "
    "phát hiện [ung thư thực quản giai đoạn tiến triển](BỆNH), hội chẩn đa "
    "chuyên khoa lên kế hoạch điều trị.",

    "Sau [phẫu thuật thay khớp gối](THỦ_THUẬT) 3 ngày, bệnh nhân "
    "[đau vết mổ](TRIỆU_CHỨNG) mức độ vừa, không [sốt](TRIỆU_CHỨNG), không "
    "[chảy dịch vết mổ](TRIỆU_CHỨNG) bất thường, giảm đau bằng "
    "[paracetamol kết hợp tramadol](THUỐC).",

    "Trẻ sinh non 32 tuần, hiện [thở nhanh](TRIỆU_CHỨNG) và [rút lõm ngực]"
    "(TRIỆU_CHỨNG), chẩn đoán [suy hô hấp sơ sinh](BỆNH), hỗ trợ thở CPAP, "
    "dùng [surfactant qua nội khí quản](THUỐC).",

    "Bệnh nhân nữ mang thai 28 tuần, [phù hai chân](TRIỆU_CHỨNG), huyết áp "
    "150/95, protein niệu dương tính, chẩn đoán [tiền sản giật](BỆNH), "
    "nhập viện theo dõi, dùng [methyldopa 250mg po tid](THUỐC).",

    "Khám sức khoẻ định kỳ phát hiện [rối loạn lipid máu](BỆNH) tình cờ, "
    "không có [triệu chứng lâm sàng](TRIỆU_CHỨNG) rõ ràng, tư vấn thay đổi "
    "lối sống trước khi cân nhắc dùng thuốc.",

    "Bệnh nhân [nổi hạch cổ](TRIỆU_CHỨNG) không đau 1 tháng, sinh thiết hạch "
    "xác định [lymphoma không Hodgkin](BỆNH), chuyển khoa ung bướu để "
    "[hoá trị](THỦ_THUẬT).",

    "Sau khi dùng [ibuprofen 400mg](THUỐC), bệnh nhân xuất hiện "
    "[đau thượng vị dữ dội](TRIỆU_CHỨNG) và [nôn ra máu](TRIỆU_CHỨNG), nghi "
    "[xuất huyết tiêu hoá do NSAID](BỆNH), ngưng thuốc ngay, truyền "
    "[omeprazole tĩnh mạch](THUỐC).",

    "Bệnh nhân cao tuổi [lú lẫn cấp](TRIỆU_CHỨNG) mới xuất hiện 2 ngày, "
    "không sốt, xét nghiệm nước tiểu [bạch cầu niệu dương tính]"
    "(KẾT_QUẢ_XÉT_NGHIỆM), nghi [nhiễm trùng tiểu gây mê sảng](BỆNH) ở "
    "người già.",

    "Tái khám sau mổ [cắt túi mật nội soi](THỦ_THUẬT) 2 tuần, vết mổ lành "
    "tốt, không [đau bụng](TRIỆU_CHỨNG), không [sốt](TRIỆU_CHỨNG), ăn uống "
    "bình thường.",

    "Bệnh nhân [ngứa da toàn thân](TRIỆU_CHỨNG) về đêm, không nổi mẩn rõ, "
    "xét nghiệm chức năng gan bất thường, nghi [ứ mật trong gan](BỆNH), "
    "chỉ định thêm [siêu âm gan mật](THỦ_THUẬT).",

    "VS38.9 12878 110 26 93RA — bệnh nhân [thở nhanh nông](TRIỆU_CHỨNG), "
    "[môi tím tái](TRIỆU_CHỨNG) nhẹ, nghe phổi ran ẩm hai bên, nghi "
    "[viêm phổi nặng hai bên](BỆNH).",

    "Bệnh nhân đái tháo đường lâu năm, [tê bì hai bàn chân](TRIỆU_CHỨNG) "
    "kiểu đi găng đi tất, chẩn đoán [biến chứng thần kinh ngoại biên do đái "
    "tháo đường](BỆNH), bổ sung [vitamin B1-B6-B12](THUỐC).",

    "Bệnh nhân [chảy máu cam tái phát](TRIỆU_CHỨNG) nhiều lần trong tháng, "
    "xét nghiệm đông máu bất thường, nghi [rối loạn đông máu](BỆNH), chuyển "
    "khoa huyết học.",

    "Sau tiêm vắc xin, trẻ [sốt nhẹ](TRIỆU_CHỨNG) 37.8 độ, [quấy khóc]"
    "(TRIỆU_CHỨNG), không [co giật](TRIỆU_CHỨNG), không cần dùng thuốc, "
    "theo dõi tại nhà.",

    "Bệnh nhân [đau vùng thắt lưng một bên](TRIỆU_CHỨNG) kèm [tiểu buốt]"
    "(TRIỆU_CHỨNG), [tiểu rắt](TRIỆU_CHỨNG), chẩn đoán [viêm đài bể thận cấp]"
    "(BỆNH), điều trị [ciprofloxacin 500mg po bid](THUỐC) 10 ngày.",

    "Khai thác tiền sử không có [dị ứng thuốc](TRIỆU_CHỨNG), không "
    "[hút thuốc lá](TRIỆU_CHỨNG), có uống rượu bia mức độ vừa phải, không "
    "[tiền sử phẫu thuật](THỦ_THUẬT) trước đây.",

    "Bệnh nhân [run tay khi nghỉ](TRIỆU_CHỨNG), [đi lại chậm chạp cứng đờ]"
    "(TRIỆU_CHỨNG), chẩn đoán [bệnh Parkinson giai đoạn sớm](BỆNH), khởi trị "
    "[levodopa-carbidopa 100/25mg po tid](THUỐC).",

    "Bệnh nhân [đau bụng quặn từng cơn](TRIỆU_CHỨNG) kèm [tiêu chảy xen kẽ "
    "táo bón](TRIỆU_CHỨNG) tái phát nhiều tháng, chẩn đoán "
    "[hội chứng ruột kích thích](BỆNH), tư vấn chế độ ăn và dùng "
    "[men vi sinh](THUỐC).",
]


def extend_synthetic_dataset(output_path: str):
    records = []
    for raw in RAW_NOTES_BATCH4:
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
    print(f"Đã sinh thêm {len(records)} đoạn batch 4")
    print(f"Phân bố type: {dict(type_counts)}")
    return records


if __name__ == "__main__":
    extend_synthetic_dataset("converted_data/synthetic_ehr_batch4.jsonl")
