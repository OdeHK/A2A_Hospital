"""
File này chứa các hàm để làm sạch văn bản trong các file .txt.
"""


import os
import re

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def tone_normalize(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text


def clean_text(text):
    # 1. Xoá control chars trước tiên (giữ \n, \t)
    text = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", text)

    # 2. Xoá các cụm (Xem Hình xxx) hoặc (Xem Bảng xxx)
    text = re.sub(r"\(Xem\s+(Hình|Bảng)[^)]+\)", "", text)

    # 2.1 Xoá cụm "Bảng xxx-yyy" hoặc "Hình xxx-yyy"
    text = re.sub(r"\b(?:Bảng|Hình)\s+\d+-\d+\b", "", text)

    # 2.2 Xoá cụm "CHƯƠNG <số>"
    text = re.sub(r"\bCHƯƠNG\s*\d*\b", "", text, flags=re.IGNORECASE)

    # 2.3 Xóa các cụm kiểu "PHẦN 14", "PHẦN 1", "PHẦN 123"…
    text = re.sub(r"\bPHẦN\s*\d+\b", "", text, flags=re.IGNORECASE)

    # # 3. Xoá số trang kiểu === Page 1005 ===
    # text = re.sub(r"=== Page \d+ ===", "", text)
    
    # 4. Xoá dấu "*" và "•"
    text = re.sub(r"[*•]", "", text)

    # 5. Xoá số trang lẻ tách dòng
    text = re.sub(r"\n\d+\n", "\n ", text)

    # 6. Nối các dòng nếu không kết thúc bằng dấu câu hoặc 
    text = re.sub(r"(?<![\.!?])\n+", " ", text)

    # 7. Chuẩn hóa dấu đầu dòng • hoặc - 
    text = re.sub(r"^[\s•\-]+\s*", "", text, flags=re.MULTILINE)

    # 8. Chuẩn hóa dấu phẩy và chấm phẩy
    text = re.sub(r"\s*([,;])\s*", r"\1 ", text)

    # 10. Giảm lặp ký tự >2 lần xuống 1 hoặc 2
    text = re.sub(r"(.)\1{3,}", r"\1", text)  # hoặc r"\1\1" nếu muốn giữ đôi

    # 11. Xóa số đầu nếu ngay sau nó là số khác
    text = re.sub(r"\b\d+\s+(?=\d)", "", text)

    # Xóa cả danh sách số liền nhau, dấu phẩy hoặc khoảng trắng phân tách
    text = re.sub(r"\b\d+(?:\s*,\s*|\s+)\d+(?:\s*,\s*\d+|\s+\d+)*\b", "", text)

    # Xóa dấu "," nếu lặp lại
    text = re.sub(r"(,\s*){2,}", ", ", text)  # giữ lại 1 dấu phẩy nếu muốn

    # 13. Gom khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)

    # 14. Xoá khoảng trắng 2 đầu
    return text.strip()

def process_txt_folder(input_folder, output_folder):
    all_texts = []

    # Duyệt tất cả file .txt trong thư mục
    for filename in os.listdir(input_folder):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        # Áp dụng hàm clean_text
        cleaned = clean_text(text)
        cleaned = tone_normalize(cleaned, dict_map)
        print(f"Đã xử lý file: {filename}")

        # Lưu vào file mới
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.join(output_folder, filename)
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(cleaned)

    print(f"Đã xử lý xong {len(os.listdir(input_folder))} file và lưu vào {output_folder}")