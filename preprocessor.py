import pandas as pd
from text_preprocessor import VnTextProcessor  
from config import VNCORENLP_SAVE_DIR
import re
import unicodedata
from typing import List

def preprocess_text(text):
    """
    Hàm preprocess text: làm sạch và chuẩn hóa text từ trang PDF.
    """
    # Chuẩn hóa tiếng Việt
    text = unicodedata.normalize('NFC', text)
    
    # Thay thế tab (\t) thành space
    text = text.replace('\t', ' ')
    
    # Giữ nguyên ký tự tiếng Việt, chỉ xóa ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\sÀ-ỹ.,!?]', ' ', text)  
    
    # Xóa xuống dòng, khoảng trắng thừa
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Xử lý hyphen (nối từ bị ngắt dòng)
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # Xóa khoảng trắng trước dấu câu
    text = re.sub(r'\s([?.!,;:])', r'\1', text)

    # Xóa số trang (dài hơn 2 chữ số) hoặc số đứng 1 mình
    text = re.sub(r'\b\d{3,}\b', ' ', text)  
    
    # Xóa các chuỗi toàn số hoặc số đứng lẻ
    text = re.sub(r'\b\d+\b', '', text)

    # Xóa các ký tự/từ đơn lẻ (1 ký tự alphabets hoặc ký tự unicode tiếng Việt)
    text = re.sub(r'\b[^\W\d_]\b', '', text)  # chỉ giữ lại chữ dài >1

    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # Xóa số index đầu dòng (1., 2., 3., 4, ...)
    text = re.sub(r'^\d+\.\s*|\b\d+\s', ' ', text)  
    
    # Xóa các từ như "CHƯƠNG", "Phần", "bảng" (bao gồm biến thể viết hoa/thường)
    text = re.sub(r'\b(CHƯƠNG|Phần|bảng|CHƯƠNG|PHẦN|BẢNG)\b', ' ', text, flags=re.IGNORECASE)
    
    # Strip lại lần cuối
    text = text.strip()

    return text

class TextPreprocessor:
    """
    Xử lý văn bản: tách từ, POS tagging, NER.
    """
    def __init__(self):
        # Dùng singleton để chỉ khởi tạo JVM 1 lần
        self.model = VnTextProcessor(
            save_dir=VNCORENLP_SAVE_DIR,
            annotators=["wseg", "pos", "ner", "parse"]
        ).processor

    def preprocess(self, text: str) -> dict:
        """
        Tiền xử lý văn bản và trả về:
            - word_segmented_join: chuỗi các từ tách cách nhau bởi khoảng trắng
            - word_segmented: list token
            - pos_tags: list (token, POS)
            - ner_tags: list (token, NER != 'O')
        """
        output = self.model.annotate_text(text)
        word_segmented = []
        # pos_tags = []
        # ner_tags = []

        for _, words in output.items():
            if isinstance(words, list):
                word_segmented.extend([word["wordForm"] for word in words])
                # pos_tags.extend([(word["wordForm"], word["posTag"]) for word in words])
                # ner_tags.extend([
                #     (word["wordForm"], word["nerLabel"])
                #     for word in words if word["nerLabel"] != "O"
                # ])

        return {
            "word_segmented_join": " ".join(word_segmented),
            "word_segmented": word_segmented,
            #"pos_tags": pos_tags,
            #"ner_tags": ner_tags,
        }


class DataPreprocessor:
    """
    Xử lý DataFrame chứa cột 'context' để tách từ và tính token count.
    """
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()

    def _tokenize(self, text: str):
        return self.text_preprocessor.preprocess(text)["word_segmented"]

    def tokenize_and_count(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm cột token_count vào DataFrame."""
        df["token_count"] = [
            len(self._tokenize(str(row["context"]))) for _, row in df.iterrows()
        ]
        return df

    def tokenize_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trả về DataFrame mới với các cột tách từ và tag."""
        df_copy = df.copy()
        df_copy[["word_segmented_join", "tokens"]] = df_copy["context"].apply(
            lambda x: pd.Series(self.text_preprocessor.preprocess(str(x)))
        )
        return df_copy
