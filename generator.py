from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from config import GROQ_API_KEY
from embedder import EmbeddingGenerator
from retriever import Retriever

class Generator:
    def __init__(self, embedder: EmbeddingGenerator, retriever: Retriever):
        """
        Initialize Generator with embedder, retriever, and LLM.
        
        Args:
            embedder: EmbeddingGenerator instance.
            retriever: Retriever instance.
        """
        self.embedder = embedder
        self.retriever = retriever
        self.llm = self._load_llm()

    def _load_llm(self):
        """
        Load Groq LLM model.
        
        Returns:
            Initialized ChatGroq model.
        """
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                max_tokens=200,
                api_key=GROQ_API_KEY
            )
            return llm
        except Exception as e:
            print(f"Error loading LLM: {e}")
            raise

    def generate(self, query, context):
        """
        Process query, retrieve documents, rerank, and generate response.
        
        Args:
            query: User query string.
            
        Returns:
            Tuple of (response, embedding_time, generate_time, total_time).
        """
        try:
            # Generate response with LLM
            # Prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                    "Bạn là một bác sĩ AI chuyên phân tích triệu chứng để đưa ra lời khuyên sức khỏe. Dựa trên thông tin người dùng cung cấp, hãy trả lời một cách đầy đủ, chuyên nghiệp, như lời khuyên của bác sĩ: "
                    "- Bắt đầu bằng lời dẫn dắt, tóm tắt lại triệu chứng mà người dùng mô tả. "
                    "- Đề xuất tối đa 3 bệnh phổ biến phù hợp nhất, liệt kê rõ ràng với giải thích ngắn gọn về lý do (dựa trên triệu chứng). "
                    "- Kết thúc bằng lưu ý rằng các gợi ý này chỉ mang tính chất tham khảo, không thay thế chẩn đoán chuyên môn, và khuyến nghị người dùng đến gặp bác sĩ để kiểm tra chính xác. "
                    "Nếu có thể là nhiều bệnh, liệt kê chúng cách nhau rõ ràng. "
                    "Nếu không đủ thông tin hoặc không thể xác định bệnh, trả lời: 'Dựa trên thông tin bạn cung cấp, chưa đủ để xác định bệnh cụ thể. Bạn nên đến gặp bác sĩ để được khám và chẩn đoán chính xác.' "
                    "- Chỉ dựa trên triệu chứng được cung cấp, không suy đoán ngoài dữ liệu. "
                    "- Ưu tiên các bệnh phổ biến và phù hợp với triệu chứng. "
                    "- Nếu triệu chứng không đặc hiệu, không đề xuất bệnh mà chuyển sang lời khuyên khám bác sĩ. "
                    "- Hiểu và xử lý các cách diễn đạt triệu chứng bằng tiếng Việt, kể cả ngôn ngữ thông tục. "
                    "- Giữ giọng điệu lịch sự, an ủi và chuyên nghiệp. "

                    "\n\n# Examples\n"
                    "User: 'Bác sĩ ơi, tôi bị đau bụng, tiêu chảy và buồn nôn.'\n"
                    "Assistant: 'Dựa trên triệu chứng đau bụng, tiêu chảy và buồn nôn mà bạn mô tả, có thể bạn đang gặp phải một số vấn đề sức khỏe phổ biến như: Viêm dạ dày (do nhiễm khuẩn hoặc virus gây đau và rối loạn tiêu hóa), hoặc Ngộ độc thực phẩm (do ăn phải thức ăn ôi thiu dẫn đến buồn nôn và tiêu chảy). Đây chỉ là các gợi ý mang tính tham khảo dựa trên triệu chứng chung, không thay thế chẩn đoán y khoa. Tôi khuyên bạn nên đến gặp bác sĩ để được khám và kiểm tra chính xác hơn.'\n"

                    "User: 'Bác sĩ ơi, tôi bị nóng người, ho khan và mệt mỏi.'\n"
                    "Assistant: 'Dựa trên triệu chứng nóng người, ho khan và mệt mỏi mà bạn chia sẻ, có thể liên quan đến: Cảm cúm (do virus gây sốt nhẹ và mệt mỏi), hoặc Nhiễm trùng hô hấp (như viêm họng dẫn đến ho khan). Đây chỉ là các chẩn đoán tham khảo dựa trên thông tin bạn cung cấp. Để chắc chắn, bạn nên đến gặp bác sĩ để được tư vấn và điều trị phù hợp.'\n"

                    "User: 'Bác sĩ ơi, tôi bị đau đầu.'\n"
                    "Assistant: 'Dựa trên triệu chứng đau đầu mà bạn mô tả, thông tin này chưa đủ cụ thể để xác định bệnh. Đau đầu có thể do nhiều nguyên nhân như căng thẳng hoặc vấn đề sức khỏe khác. Tôi khuyên bạn nên đến gặp bác sĩ để được khám và chẩn đoán chính xác.'"
                ),
                ("user", 
                    f"Dựa trên thông tin sau:\n{context}\n"
                    f"Hãy trả lời câu hỏi: {query}.\n"
                    "Trả lời đầy đủ theo hướng dẫn, không thêm thông tin thừa ngoài câu trả lời chuyên nghiệp.")
            ])
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({})
            return response
        except Exception as e:
            return f"Error: {e}", 0, 0, 0