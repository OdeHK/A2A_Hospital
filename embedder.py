import torch
import re
from FlagEmbedding import BGEM3FlagModel
from text_preprocessor import VnTextProcessor  # Dùng singleton thay vì import trực tiếp VnCoreNLP
from config import EMBEDDING_MODEL_NAME, VNCORENLP_SAVE_DIR


class EmbeddingGenerator:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, device=None, max_length=512):
        """
        Khởi tạo EmbeddingGenerator với mô hình BGE-M3.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.max_length = max_length

        # Dùng singleton để chỉ khởi tạo JVM 1 lần
        self.vncorenlp = VnTextProcessor(
            save_dir=VNCORENLP_SAVE_DIR,
            annotators=["wseg"]
        ).processor

        # Khởi tạo mô hình embedding
        self.model = BGEM3FlagModel(model_name, device=device)

    def embed_query(self, processed_text):
        """
        Tính embedding cho một câu truy vấn đã được xử lý.
        """
        try:
            if not processed_text or not isinstance(processed_text, str):
                raise ValueError("Processed text phải là một chuỗi không rỗng.")

            embedding = self.model.encode(processed_text)

            if isinstance(embedding, dict) and 'dense_vecs' in embedding:
                return embedding['dense_vecs'].tolist()
            else:
                raise ValueError("Embedding không chứa 'dense_vecs'.")
        except Exception as e:
            print(f"Lỗi khi tính embedding cho query: {e}")
            return [0.0] * 1024  # Trả về vector rỗng nếu lỗi

    def embed_documents(self, texts):
        """
        Tính embedding cho danh sách văn bản.
        """
        embeddings = self.model.encode(texts)
        if isinstance(embeddings, dict) and 'dense_vecs' in embeddings:
            return [v.tolist() for v in embeddings['dense_vecs']]
        return [v.tolist() for v in embeddings]

    def preprocess_and_tokenize(self, text):
        """
        Tiền xử lý và tokenize văn bản sử dụng VnCoreNLP (qua singleton).
        """
        try:
            if self.vncorenlp is None:
                return text

            # Chuẩn hóa
            text = text.strip().lower()

            # Tách từ
            annotated = self.vncorenlp.annotate_text(text)

            tokens = []
            for _, words in annotated.items():
                if isinstance(words, list):
                    tokens.extend([word['wordForm'] for word in words if word['wordForm']])

            # Làm sạch token
            tokens = [re.sub(r'\s+', ' ', word).strip() for word in tokens]

            return " ".join(tokens)
        except Exception as e:
            print(f"Lỗi khi tiền xử lý và tokenize: {e}")
            return text

    def get_dense_size(self):
        """
        Lấy kích thước của dense vector.
        """
        sample_text = "This is a test sentence."
        sample_embedding = self.model.encode(sample_text)
        if isinstance(sample_embedding, dict) and 'dense_vecs' in sample_embedding:
            return len(sample_embedding['dense_vecs'][0])
        return len(sample_embedding[0])
