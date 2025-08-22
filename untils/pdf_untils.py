"""File này chứa các hàm để xử lý PDF, bao gồm:
- Lấy ra một trang từ PDF và lưu thành file PDF riêng.
- Trích xuất văn bản ngoài các bảng trong trang pdfplumber.
- Chuyển đổi PDF thành file txt theo từng đoạn trang.
- Định dạng bảng bằng cách sử dụng Gemini API.
"""


import os
import fitz  # PyMuPDF
import pdfplumber
import contextlib
import io
import google.generativeai as genai
# Cấu hình Gemini API
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(os.getenv("GEMINI_MODEL"))

def save_single_page(pdf_path, page_num, output_path):
    """
    Lấy ra 1 trang từ PDF và lưu thành file PDF riêng.
    page_num: chỉ số trang (0-based)
    """
    doc = fitz.open(pdf_path)
    new_doc = fitz.open()  # file empty pdf

    page = doc.load_page(page_num)
    new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"Trang {page_num} đã được lưu vào {output_path}")

def extract_text_outside_tables(pl_page):
    """
    Trả về text ngoài các bảng trong trang pdfplumber.
    """
    table_bboxes = [table.bbox for table in pl_page.find_tables()]
    words = pl_page.extract_words()
    non_table_words = []

    for w in words:
        x0, y0, x1, y1 = float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])
        in_table = any((x0 >= bx0 and x1 <= bx1 and y0 >= by0 and y1 <= by1)
                       for (bx0, by0, bx1, by1) in table_bboxes)
        if not in_table:
            non_table_words.append(w)

    # Gom text theo dòng
    lines = {}
    for w in non_table_words:
        line_key = round(w["top"], 1)
        lines.setdefault(line_key, []).append(w["text"])
    
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))

def format_table_with_llm(tables):
    """
    This function formats a single table only using the Gemini API.
    """
    if not tables:
        return ""

    cleaned_tables = [[cell if cell else "None" for cell in row] for table in tables for row in table if row]
    prompt = f"""
Chuyển bảng sau thành dạng dễ đọc theo format sau (nếu ô trống thì để None).Chỉ tạo ra theo format sau và không cần giải thích gì thêm:

Ví dụ format:
Bảng 1:
    dòng 1:
    - Tên cột 1: Nội dung cột 1 dòng 1
    - Tên cột 2: Nội dung cột 2 dòng 1
    dòng 2:
    - Tên cột 1: Nội dung cột 1 dòng 2
    - Tên cột 2: Nội dung cột 2 dòng 2
Bảng 2:
    dòng 1:
    - Tên cột 1: Nội dung cột 1 dòng 1
    - Tên cột 2: Nội dung cột 2 dòng 1
    dòng 2:
    - Tên cột 1: Nội dung cột 1 dòng 2
    - Tên cột 2: Nội dung cột 2 dòng 2

Bảng cần xử lý:
{cleaned_tables}
"""
    parts = [{"text": prompt}]
    response = model.generate_content(parts)
    return response.text

def convert_pdf2txt(pdf_path, output_dir, start_page=None, end_page=None, page_per_file=100, skip_pages=None):
    """
    Chuyển đổi PDF thành txt theo từng đoạn trang.
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF.
        output_dir (str): Thư mục để lưu file txt.
        start_page (int): Trang bắt đầu (1-based). Nếu None thì mặc định = 1.
        end_page (int): Trang kết thúc (1-based). Nếu None thì mặc định = tổng số trang.
        page_per_file (int): Số trang mỗi file txt.
        skip_pages (list): Danh sách số trang (1-based) cần bỏ qua.
    """
    os.makedirs(output_dir, exist_ok=True)
    warning_message = "Cannot set stroke color because 2 components are specified but only 1 (grayscale), 3 (rgb) and 4 (cmyk) are supported"
    skip_pages = skip_pages or []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        # Nếu không truyền start_page/end_page thì mặc định lấy toàn bộ
        start_page = start_page or 1
        end_page = end_page or total_pages
        if start_page < 1 or end_page > total_pages or start_page > end_page:
            raise ValueError(f"Phạm vi trang không hợp lệ: {start_page} - {end_page} (PDF có {total_pages} trang)")

        # Loop theo batch page_per_file
        for i in range(start_page - 1, end_page, page_per_file):
            batch_start = i + 1
            batch_end = min(i + page_per_file, end_page)
            output_file = os.path.join(output_dir, f"page_{batch_start}_{batch_end}.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                for page_num in range(batch_start - 1, batch_end):
                    if skip_pages and page_num + 1 in skip_pages:
                        print(f"⏭️ Skipping page {page_num + 1}")
                        continue
                    try:
                        buf = io.StringIO()
                        with contextlib.redirect_stderr(buf):
                            page = pdf.pages[page_num]
                            text = extract_text_outside_tables(page)
                            tables = page.extract_tables()
                            if tables:
                                tables = format_table_with_llm(tables)
                            else:
                                tables = ""
                        
                        stderr_output = buf.getvalue()
                        if warning_message in stderr_output:
                            raise RuntimeError(stderr_output.strip())
                        print(f"✅ Extracted successfully with pdfplumber in page {page_num + 1}") 
                    except Exception as e:
                        try:
                            print(f"⚠️ pdfplumber error on page {page_num + 1}. Using PyMuPDF fallback.")
                            with fitz.open(pdf_path) as doc:
                                page = doc.load_page(page_num)
                                text = page.get_text()
                                tables = ""  
                            print(f"✅ Extracted successfully with fitz in page {page_num + 1}")
                        except Exception as e:
                            print(f"FAILED {e}")
                            text, tables = "", ""

                    header = f"=== Page {page_num + 1} ==="
                    page_content = header + '\n' + text.strip() + '\n' + tables.strip()
                    f.write(page_content + "\n\n")

            print(f"📄 Saved pages {batch_start} to {batch_end} in {output_file}")