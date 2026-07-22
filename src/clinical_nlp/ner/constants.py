LABEL_LIST = [
    "O",
    "B-THUỐC", "I-THUỐC",
    "B-BỆNH", "I-BỆNH",
    "B-TRIỆU_CHỨNG", "I-TRIỆU_CHỨNG",
    "B-THÔNG_TIN_BỆNH_NHÂN", "I-THÔNG_TIN_BỆNH_NHÂN",
    "B-KẾT_QUẢ_XÉT_NGHIỆM", "I-KẾT_QUẢ_XÉT_NGHIỆM",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
