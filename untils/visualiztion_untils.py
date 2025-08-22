"""File này chứa các hàm để hiển thị nội dung các trang PDF đã được xử lý thành file .txt.
Các hàm này sử dụng Streamlit để tạo giao diện người dùng.
Lưu ý: Các file .txt phải được tạo ra từ các trang PDF đã xử lý trước đó.
"""

from pathlib import Path
import streamlit as st

OUTPUT_DIR = "processed_data_2"   # Thư mục cố định
CLEAN_DATA_DIR = "cleaned_data"   # Thư mục chứa dữ liệu đã làm sạch

def get_page_text(txt_dir, page_number):
    """
    Lấy nội dung text từ 1 trang trong thư mục chứa các file .txt đã sinh ra
    page_number: bắt đầu từ 1 (số trang gốc trong PDF)
    """
    txt_dir = Path(txt_dir)

    for file in txt_dir.glob("page_*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()

        # Parse theo marker số trang thật
        pages = data.split("=== Page ")[1:]
        for p in pages:
            header, content = p.split("===", 1)
            num = int(header.strip())
            if num == page_number:
                return content.strip()

    return ""

def get_total_pages(txt_dir):
    """Tìm số trang lớn nhất trong thư mục"""
    txt_dir = Path(txt_dir)
    max_page = 0
    for file in txt_dir.glob("page_*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            data = f.read()
        pages = data.split("=== Page ")[1:]
        for p in pages:
            header, _ = p.split("===", 1)
            num = int(header.strip())
            if num > max_page:
                max_page = num
    return max_page

def compare_pdf_pages_ui(output_dir: str = OUTPUT_DIR, clean_data_dir: str = CLEAN_DATA_DIR):
    """Streamlit UI: So sánh nội dung file .txt giữa 2 thư mục (gốc & đã làm sạch)."""
    st.set_page_config(page_title="PDF Page Comparison", layout="wide")
    st.title("📖 PDF Page Comparison (from .txt batches)")

    txt_dir = Path(output_dir)
    if not txt_dir.exists():
        st.error(f"❌ Thư mục `{output_dir}` không tồn tại.")
        return

    total_pages = get_total_pages(txt_dir)
    if total_pages == 0:
        st.warning("⚠️ Không tìm thấy trang nào trong thư mục.")
        return

    st.success(f"✅ Tìm thấy {total_pages} trang trong thư mục `{output_dir}`.")

    # Chọn số trang để so sánh
    page_number = st.number_input("Chọn trang:", min_value=1, max_value=total_pages, step=1)

    # Nội dung gốc
    content = get_page_text(txt_dir, page_number)
    if content:
        st.text_area(f"Nội dung trang {page_number} (Gốc)", content, height=300)
    else:
        st.error("❌ Không tìm thấy nội dung cho trang này.")

    # Nội dung đã làm sạch
    cleaned_txt_dir = Path(clean_data_dir)
    if not cleaned_txt_dir.exists():
        st.error(f"❌ Thư mục `{clean_data_dir}` không tồn tại.")
        return

    cleaned_content = get_page_text(cleaned_txt_dir, page_number)
    if cleaned_content:
        st.text_area(f"Nội dung trang {page_number} (Đã làm sạch)", cleaned_content, height=300)
    else:
        st.error("❌ Không tìm thấy nội dung đã làm sạch cho trang này.")

def pdf_page_viewer_ui(output_dir: str = OUTPUT_DIR):
    """Streamlit UI: Xem nội dung các trang txt từ thư mục đã xử lý."""
    st.set_page_config(page_title="PDF Page Viewer", layout="wide")
    st.title("📖 PDF Page Viewer (from .txt batches)")

    txt_dir = Path(output_dir)

    if not txt_dir.exists():
        st.error(f"❌ Thư mục `{output_dir}` không tồn tại.")
        return

    total_pages = get_total_pages(txt_dir)
    if total_pages == 0:
        st.warning("⚠️ Không tìm thấy trang nào trong thư mục.")
        return

    st.success(f"✅ Tìm thấy {total_pages} trang trong thư mục `{output_dir}`.")

    # Chọn số trang để xem
    page_number = st.number_input("Chọn trang:", min_value=1, max_value=total_pages, step=1)

    # Hiển thị luôn khi thay đổi số trang
    content = get_page_text(txt_dir, page_number)
    if content:
        st.text_area(f"Nội dung trang {page_number}", content, height=600)
    else:
        st.error("❌ Không tìm thấy nội dung cho trang này.")
