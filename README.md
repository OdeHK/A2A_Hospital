# A2A_Hospital

# RAG Chatbot Streamlit Project
Kho lưu trữ này chứa triển khai của một chatbot dựa trên Retrieval-Augmented Generation (RAG) sử dụng Streamlit. Dự án được thiết kế để xử lý các tài liệu PDF, trích xuất và tiền xử lý văn bản, nhúng nó thành các vector, lưu trữ trong cơ sở dữ liệu vector, truy xuất thông tin liên quan, và tạo câu trả lời bằng mô hình ngôn ngữ. Dưới đây là tổng quan về các file và mục đích của chúng trong dự án.
# Cấu trúc dự án

- .gitignore: Xác định các file và thư mục cần bỏ qua khi commit vào Git (ví dụ: file tạm, thông tin xác thực).
- README.md: File này, cung cấp tổng quan về dự án và các thành phần của nó.
- config.py: File cấu hình để lưu trữ các đường dẫn, tên mô hình, và khóa API. Nó tập trung hóa các thiết lập như đường dẫn file PDF, tên mô hình (ví dụ: BAAI/bge-m3, LLaMA), và thông tin xác thực của cơ sở dữ liệu Qdrant.
- data_loader.py: Chịu trách nhiệm tải văn bản từ file PDF và thực hiện chuẩn hóa ban đầu. Nó giải quyết các vấn đề như chồng chữ hoặc lỗi định dạng trong PDF bằng cách sử dụng tọa độ của các từ (giãn cách hoặc nén) để điều chỉnh khoảng cách phù hợp.
- preprocessor.py: Xử lý văn bản bằng regex để làm sạch và cấu trúc các câu từ dữ liệu đã tải. File này cũng bao gồm các lớp để tính toán tokenization, chuẩn bị văn bản cho các bước tiếp theo.
- text_preprocessor.py: Giúp khởi tạo mô hình vncorenlp một lần duy nhất để tránh lỗi JVM (Java Virtual Machine), đảm bảo hiệu suất và tính ổn định khi xử lý văn bản tiếng Việt.
- embedder.py: Chứa lớp embedding generator để khởi tạo và nhúng các đoạn văn bản đã được token hóa thành vector, giúp lưu trữ hiệu quả trong cơ sở dữ liệu.
- vector_db.py: Được sử dụng để lưu trữ các đoạn văn bản đã nhúng (embeddings) vào cơ sở dữ liệu vector Qdrant, hỗ trợ truy xuất nhanh chóng.
- retriever.py: Tạo lớp giúp truy xuất (retrieve) các điểm dữ liệu và thực hiện reranking dựa trên truy vấn (query), tối ưu hóa kết quả trả về.
- generator.py: Khởi tạo một mô hình ngôn ngữ (LLM) và tạo hàm để sinh câu trả lời dựa trên truy vấn, hoàn thiện phần tạo nội dung của chatbot.
- med_chatbot.py: Xây dựng giao diện Streamlit để thực thi pipeline RAG chatbot, tích hợp toàn bộ quy trình từ tải dữ liệu đến tạo câu trả lời.

# Hướng dẫn sử dụng

- Cài đặt các thư viện phụ thuộc cần thiết (ví dụ: pip install streamlit qdrant-client vncorenlp ....).
- Cấu hình các thông tin trong config.py (đường dẫn file, khóa API, v.v.).
- Chạy ứng dụng Streamlit bằng lệnh: streamlit run < PATH_to_med_chatbot.py>


## Lưu ý

- Đảm bảo file PDF được chỉ định trong config.py tồn tại và có thể truy cập.
- Kiểm tra khóa API (Qdrant, Groq) để đảm bảo kết nối thành công.