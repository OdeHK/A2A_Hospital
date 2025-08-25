import os
import warnings
import py_vncorenlp

# Biến toàn cục để lưu instance duy nhất
_vncorenlp_instance = None

class VnTextProcessor:
    """
    Wrapper cho VnCoreNLP, dùng mô hình Singleton.
    Mục đích: tránh khởi tạo JVM nhiều lần gây lỗi "VM is already running".
    """
    def __init__(self, 
                 save_dir: str = "./models/vncorenlp", 
                 annotators: list = None):
        global _vncorenlp_instance

        annotators = annotators or ["wseg"]
        os.makedirs(save_dir, exist_ok=True)

        # Nếu đã có instance, dùng lại
        if _vncorenlp_instance is not None:
            self.processor = _vncorenlp_instance
            return

        try:
            # Tải model nếu chưa có
            py_vncorenlp.download_model(save_dir=save_dir)

            # Khởi tạo VnCoreNLP
            self.processor = py_vncorenlp.VnCoreNLP(
                annotators=annotators,
                save_dir=save_dir
            )

            # Lưu instance để tái sử dụng
            _vncorenlp_instance = self.processor

        except ValueError as e:
            # Bắt lỗi JVM đã chạy
            if "VM is already running" in str(e):
                warnings.warn("JVM đã khởi tạo, dùng DummyProcessor thay thế.")
                self.processor = DummyProcessor()
            else:
                raise e

    def preprocess(self, text: str) -> dict:
        """
        Tiền xử lý văn bản: tách từ
        """
        tokens = self.processor.word_segment(text)
        return {"word_segmented": tokens}


class DummyProcessor:
    """
    Processor giả phòng trường hợp JVM đã chạy trước đó.
    """
    def word_segment(self, text: str) -> list:
        return text.split()
