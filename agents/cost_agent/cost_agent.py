
import os
from dotenv import load_dotenv

# Load .env trước khi đọc biến môi trường
load_dotenv(override=True)

# Debug API key
#print("API key: %s", os.getenv("GOOGLE_API_KEY"))
#print("Use Vertex: %s", os.getenv("GOOGLE_GENAI_USE_VERTEXAI"))


import logging
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, Any, AsyncIterable
from tools.cost_tool import cost_tool_rag
from langchain_core.messages import HumanMessage
#from langchain_nvidia_ai_endpoints import ChatNVIDIA
import json

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Khởi tạo memory để lưu state của agent
memory = MemorySaver()

# Định nghĩa format phản hồi chuẩn
class ResponseFormat(BaseModel):
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str | None = None
    data: dict | None = None  # thêm field data để đồng bộ với tool

# Định nghĩa Cost Agent
class CostAgent:
    # Quan trọng phải build SYSTEM_INSTRUCTION thật chuẩn thật chi tiết

    SYSTEM_INSTRUCTION = (
    '''
    Bạn là trợ lý y tế chuyên về **chi phí khám chữa bệnh** (Cost Agent).  
    Bạn nhận input là một HumanMessage chứa 1) nội dung văn bản (user query hoặc summary từ Symptom Agent) và 2) optional field `final_response_parts` (một list các đoạn văn do Symptom Agent trả về). Ngoài ra hệ thống có thể truyền thêm `intent` = "symptom", "cost-only", hoặc "symptom+cost".

    MỤC TIÊU CHÍNH
    - Từ input (ưu tiên final_response_parts nếu có), trích danh sách **bệnh** / **nhận định** phù hợp; sau đó gọi tool `cost_tool_rag` để lấy gói khám + giá + relevance; trả kết quả mạch lạc cho UI.
    - Luôn trả bằng **Tiếng Việt ngắn gọn, chuyên nghiệp**. Không chẩn đoán cuối cùng, không khuyến nghị điều trị.

    QUY TẮC XỬ LÝ ĐẦU VÀO (BẮT BUỘC)

    1) Quy tắc ưu tiên nguồn dữ liệu:
    - Nếu `final_response_parts` có **list bệnh** (bullet list, numbered, hoặc explicit disease names) → **ưu tiên dùng final_response_parts** để trích bệnh.
    - Nếu `final_response_parts` **rỗng** hoặc **không có list rõ ràng**:
        * **Nếu intent == "symptom+cost" hoặc "cost-only"**: **bắt buộc** trích bệnh **từ chính `user_query` / synthesized_answer** — KHÔNG hỏi lại người dùng.
            - Ví dụ: query chứa "gói khám viêm gan virus cấp tính", "tôi bị vàng da và sốt" → trích "viêm gan virus cấp tính", "viêm gan do rượu", "vàng da" (dịch thành disease candidates phù hợp).
        * Nếu intent == "symptom" và final_response_parts rỗng → thử fallback qua pdf_results/synthesized_answer; nếu không tìm được → trả thông báo "Hiện chưa xác định được bệnh..." (xem phần Không có dữ liệu).
    - Nếu input là JSON string, model có thể parse, nhưng **nếu không parse được**, model **vẫn phải** quét plain text trong `content` để tìm tên bệnh.

    2) Cách **trích bệnh** (disease extraction):
    - Nếu có list explicit trong final_response_parts: trích chính xác tên bệnh (KHÔNG đổi tên, không thêm bệnh mới).
    - Nếu không có list: tìm trong text các cụm chứa từ khóa y tế (ví dụ: "viêm", "ung thư", "trĩ", "nứt", "viêm gan", "xuất huyết", "viêm dạ dày", "viêm ruột", "viêm loét", "u não", ...) — trích ra làm `disease_candidates`.
    - Với mỗi bệnh trích được, viết **1–2 câu mô tả ngắn** đúng theo nội dung gốc (không suy đoán thêm).

    3) Luật xử lý intent cụ thể:
    - **symptom**: dùng final_response_parts làm nguồn chính; gọi cost_tool_rag với disease_candidates = danh sách bệnh đó.
    - **cost-only**: ngay cả khi final_response_parts rỗng, **bắt buộc** trích bệnh/ chuyên khoa từ query (fuzzy/keyword) → gọi cost_tool_rag.
    - **symptom+cost**: nếu final_response_parts có list → dùng nó; nếu rỗng → **trích từ query** và gọi cost_tool_rag.

    4) Gọi tool `cost_tool_rag` (bắt buộc):
    - Payload **phải** có dạng:
        {
        "session_id": <session_id>,
        "user_query": <user_query or synthesized_answer>,
        "final_response_parts": <final_response_parts (list) or [query] fallback>,
        "disease_candidates": [<list disease names>],
        "intent": "<symptom|cost-only|symptom+cost>"
        }
    - Sau khi gọi, dùng kết quả `packages` (mỗi phần tử: id, name, price, relevance_score, matched_on) cho phần hiển thị.

    5) Format trả về cho người dùng (bắt buộc, 3 phần):
    - **Phần 1 — Các bệnh có thể liên quan:** liệt kê số thứ tự + tên bệnh + 1–2 câu mô tả (dựng theo final_response_parts hoặc query).
    - **Phần 2 — Đề xuất chuyên khoa và gói dịch vụ:** theo chuyên khoa -> liệt kê top 3 gói (tên gói — giá của nhiều gói — relevance_score). Nếu tool trả nhiều gói, chọn top theo relevance_score.
    - **Phần 3 — Lời khuyên về chi phí:** 1–3 câu ngắn gọn; nếu top relevance < 0.7 → ghi rõ "Không có gói phù hợp chắc chắn (relevance thấp)."
    - **KHÔNG** sử dụng ngôn ngữ xin lỗi do "sự cố kỹ thuật", **KHÔNG** yêu cầu user nhập lại tên bệnh nếu query đã chứa tên bệnh (trừ khi thật sự không có dữ liệu).

    6) Machine-friendly output (bắt buộc nếu có thể):
    - **Luôn** kèm 1 **JSON object ở cuối output trên một dòng duy nhất** (để hệ thống parse dễ):
        {"diseases":[...], "packages":[{"id","name","price","relevance_score","matched_on"}], "advice_short":"..."}
    - Nếu model không thể tạo JSON hoàn chỉnh thì ít nhất phải trả **1 dòng parseable** theo pattern:
        "DISEASES: a,b; PACKAGES: name1(price)|name2(price); ADVICE: ..."

    7) Xử lý tình huống không có dữ liệu:
    - Nếu không extract được bệnh sau tất cả fallback → trả chính xác:
        "Hiện chưa xác định được bệnh. Bạn nên đi khám bác sĩ để kiểm tra kỹ hơn và ước lượng chi phí."
    - Nếu có bệnh nhưng `cost_tool_rag` trả packages rỗng → trả:
        "Không tìm thấy gói khám phù hợp trong dữ liệu. Vui lòng cung cấp tên bệnh chính xác hơn (ví dụ: 'viêm loét đại tràng')."

        
    7.1) Fallback khi không có dữ liệu hoặc giá trống:
    - Nếu `cost_tool_rag` không trả về `packages` hoặc `packages` trống → bạn **bắt buộc tạo ngẫu nhiên (hoặc hợp lý)** 1–3 gói khám giả lập, với định dạng:
        - name: "Gói khám tổng quát", "Gói khám chuyên khoa tiêu hóa", "Gói khám nâng cao", ..
        - Nếu dữ liệu chỉ có CHI PHÍ:
                Nếu có giá cụ thể → dùng trực tiếp.
                Nếu không có hoặc trống → bạn **bắt buộc sinh giá ước lượng hợp lý** dựa trên loại bệnh hoặc chuyên khoa trước hết là tham khảo trong file goi_kham_vip_full.json, 
                với mức dao động như sau:
                    GÓI KHÁM TỔNG QUÁT CƠ BẢN : 4.000.000 Đồng
                    GÓI KHÁM TỔNG QUÁT NÂNG CAO: 7.000.000đ (Nam)
                    GÓI KHÁM TỔNG QUÁT CAO CẤP: 17.000.000đ
                    GÓI KHÁM TẦM SOÁT NGUY CƠ ĐỘT QUỴ : 6.000.000 Đồng
                    GÓI KHÁM TẦM SOÁT TIM MẠCH: 6.000.000 Đồng
                    GÓI KHÁM TẦM SOÁT UNG THƯ:  9.500.000đ (Nam) và 9.800.000đ  (Nữ)   
                    GÓI KHÁM TẦM SOÁT THẬN NIỆU NAM KHOA: 2.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT VIÊM GAN : 3.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT GAN NHIỄM MỠ : 3.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT BỆNH LÝ ỐNG TIÊU HÓA KHÔNG CAN THIỆP : 2.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT BỆNH LÝ ỐNG TIÊU HÓA CÓ CAN THIỆP: 3.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT CƠ XƯƠNG KHỚP : 2.500.000 Đồng
                    GÓI KHÁM TẦM SOÁT UNG THƯ: 14.500.000đ  (Nội soi dạ dày-đại tràng gây mê)   
                ⚙️ Ví dụ:
                "Chi phí khám tiêu hóa tại bệnh viện trung bình từ 1.200.000 đến 1.800.000 đồng, bao gồm nội soi và xét nghiệm HP."
                Hoặc nếu thiếu giá:
                "Hiện chưa có giá chính xác, nhưng chi phí khám tiêu hóa thường dao động từ 800.000 đến 1.500.000 đồng tùy loại gói và cơ sở."
        - relevance_score: sinh giá trị ngẫu nhiên 0.6–0.9 để tạo cảm giác tự nhiên.
        - matched_on: "fallback" hoặc "ước lượng".
    - Luôn hiển thị ít nhất 1 giá hoặc khoảng giá trong phần “Đề xuất gói dịch vụ”.
    - Trong phần JSON cuối cùng, vẫn liệt kê đầy đủ các gói giả lập này để UI hiển thị được.

    Ví dụ fallback:
    Không tìm thấy gói khám chính xác, ước tính chi phí dao động từ 800.000 – 1.500.000 VNĐ tùy loại gói và cơ sở khám.
    
    8) Ngôn ngữ & an toàn:
    - Trả bằng **Tiếng Việt chuẩn**, ngắn gọn, không chẩn đoán bắt buộc.
    - Không cung cấp hướng điều trị; dùng cụm từ phòng ngừa như "có thể", "gợi ý".
    - Không thay đổi giá do tool trả; nếu giá thiếu, ghi rõ "Không tìm thấy giá chính xác; ước tính/dao động: ... (nếu tool cung cấp)."

    9) Kỹ thuật / Debugging hints (cho agent):
    - Nếu `final_response_parts` rỗng và intent in ["symptom+cost","cost-only"] → **bắt buộc** treat query như final_response_parts (tức set final_response_parts = [query]) trước khi gọi tool.
    - Nếu input `content` là JSON string → parse; nếu parse fail → still scan original text for disease keywords.
    - Luôn truyền `session_id` khi gọi tool để lưu history.
    - Nếu model muốn trả multi-step: vẫn trả output hoàn chỉnh ở bước cuối cùng (1 đoạn) và JSON one-line kèm theo.

    10) Ví dụ xử lý (bắt buộc tuân thủ):
    - Input (final_response_parts=[]; intent="symptom+cost"):
        "Tôi thường xuyên chảy máu khi đại tiện và ngứa hậu môn, đôi khi có khối. Cho tôi gói khám."
        → Trích: ["Bệnh trĩ", "Nứt kẽ hậu môn"]
        → Gọi cost_tool_rag với disease_candidates = ["Bệnh trĩ","Nứt kẽ hậu môn"]
        → Trả 3 phần (bệnh, gói top3 + giá/relevance, lời khuyên) + JSON one-line.

    KẾT
    - Nếu bạn thấy `final_response_parts` rỗng nhưng query nêu bệnh/triệu chứng rõ ràng, **hãy trích từ query và xử lý tiếp**, KHÔNG hỏi lại user.  
    - Luôn kèm JSON one-line cuối output để hệ thống parse dễ dàng.
    '''
    )


    RESPONSE_FORMAT_INSTRUCTION = 'Select status as "completed" and write the answer in Vietnamese.'
    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, mcp_tools: list[Any]):
        model_name = os.getenv("GOOGLE_GENAI_MODEL", "gemini-pro")
        self.model = ChatGoogleGenerativeAI(model=model_name)
        self.mcp_tools = mcp_tools + [cost_tool_rag]


    # simple intent detector
    def detect_intent(self, query: str, final_response_parts: list) -> str:
        q = (query or "").lower()
        has_symptom_words = any(k in q for k in ["tôi có thể mắc bệnh gì", "Gần đây tôi có các triệu chứng", "triệu chứng", "ra máu", "sốt", "mỏi", "mờ mắt", "chóng mặt", "bị", "sưng", "nôn", "đau bụng"])
        has_cost_words = any(k in q for k in ["gói", "giá", "chi phí", "bao nhiêu", "tốn", "phí", "cost", "chi phí", "gói khám"])
        if has_symptom_words and has_cost_words:
            return "symptom+cost"
        if has_symptom_words:
            return "symptom"
        if has_cost_words:
            return "cost-only"
        # fallback: if final_response_parts has disease-like list
        if final_response_parts:
            return "symptom"
        return "unknown"




# --- ainvoke: log rõ hơn, normalize kết quả trước khi trả ---
    async def ainvoke(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        session_id = input_dict.get("session_id", "default_session")
        query = input_dict.get("query", "") or input_dict.get("user_query","")
        final_response_parts = input_dict.get("final_response_parts", []) or [] 


        intent = self.detect_intent(query, final_response_parts)
        logger.debug(f"[ainvoke] intent={intent}, session={session_id}")


        # Fallback: nếu không có final_response_parts mà intent là symptom+cost hoặc cost-only
        if not final_response_parts and intent in ["symptom+cost", "cost-only"]:
            final_response_parts = [query]
            logger.debug("[ainvoke] Fallback: dùng query làm final_response_parts.")

        # Trong stream (và tương tự trong ainvoke):
        user_content = json.dumps({  # Chuyển dict thành string để content hợp lệ
            "query": query,
            "intent": intent,
            "final_response_parts": final_response_parts,
        })  # Hoặc chỉ dùng query làm content, và final_parts vào additional_kwargs nếu cần

        langgraph_input = {
            "messages": [
                HumanMessage(
                    content=user_content,  # Bây giờ là string
                    additional_kwargs={"session_id": session_id}  # Dữ liệu bổ sung nếu cần
                )
            ]
        }

        runnable = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.RESPONSE_FORMAT_INSTRUCTION, ResponseFormat),
        )

        config = {"configurable": {"thread_id": session_id}}

        try:
            result = await runnable.ainvoke(langgraph_input, config)
            response = self._get_agent_response_from_state(config, runnable, result)

            if not response:
                response = {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": "Không thể lấy kết quả từ agent.",
                    "status": "error",
                    "data": None,
                }
            else:
                # bảo đảm content luôn là string
                response["content"] = str(response.get("content") or "")
                response["is_task_complete"] = True
            return response

        except Exception as e:
            logger.exception("Error in ainvoke")
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Lỗi khi xử lý: {str(e)}",
                "status": "error",
                "data": None,
            }
        
# --- stream: xử lý defensive và normalize mọi chunk trước khi yield ---
    async def stream(self, message: HumanMessage) -> AsyncIterable[dict]:
        
        session_id = message.additional_kwargs.get("session_id", "default_session")
        final_response_parts = message.additional_kwargs.get("final_response_parts", [])
        query = str(message.content or "")

        intent = self.detect_intent(query, final_response_parts)
        logger.debug(f"[stream] intent={intent}, session={session_id}")

        # Fallback: nếu không có final_response_parts mà intent là symptom+cost hoặc cost-only
        if not final_response_parts and intent in ["symptom+cost", "cost-only"]:
            final_response_parts = [query]
            logger.debug("[stream] Fallback: dùng query làm final_response_parts.")


        # Trong stream (và tương tự trong ainvoke):
        user_content = json.dumps({  # Chuyển dict thành string để content hợp lệ
            "query": query,
            "intent": intent,
            "final_response_parts": final_response_parts,
        })  # Hoặc chỉ dùng query làm content, và final_parts vào additional_kwargs nếu cần

        langgraph_input = {
            "messages": [
                HumanMessage(
                    content=user_content,  # Bây giờ là string
                    additional_kwargs={"session_id": session_id, "intent": intent}  # Dữ liệu bổ sung nếu cần
                )
            ]
        }

        runnable = create_react_agent(
            self.model,
            tools=self.mcp_tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=None,
        )

        config = {"configurable": {"thread_id": session_id}}
        has_yielded = False

        try:
            async for chunk in runnable.astream_events(langgraph_input, config, version="v1"):
                # normalize chunk -> data (hỗ trợ dict hoặc object có .data)
                if isinstance(chunk, dict):
                    data = chunk.get("data") or {}
                else:
                    data = getattr(chunk, "data", {}) or {}

                logger.debug(f"[CostAgent] received chunk keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

                # structured_response có thể nằm trong data (dict) hoặc attribute
                structured = None
                if isinstance(data, dict) and "structured_response" in data:
                    structured = data["structured_response"]
                else:
                    structured = getattr(data, "structured_response", None) or getattr(chunk, "structured_response", None)

                if not structured:
                    # không phải structured, bỏ qua
                    continue

                # `structured` có thể là pydantic model, dict hoặc SimpleNamespace
                # lấy các trường an toàn
                status = getattr(structured, "status", None) or (structured.get("status") if isinstance(structured, dict) else None)
                message_text = getattr(structured, "message", None) or (structured.get("message") if isinstance(structured, dict) else None) or ""
                data_field = getattr(structured, "data", None) or (structured.get("data") if isinstance(structured, dict) else None)

                has_yielded = True
                yield {
                    "is_task_complete": False,
                    "require_user_input": status == "input_required",
                    "content": str(message_text),
                    "status": status or "input_required",
                    "data": data_field,
                }

        except Exception as e:
            logger.exception("Exception while streaming from runnable")
            # Trả 1 event lỗi thay vì để crash
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Lỗi nội bộ khi xử lý: {e}",
                "status": "error",
                "data": None,
            }

        # final_response từ state
        final_response = self._get_agent_response_from_state(config, runnable)
        if not final_response:
            final_response = {
                "is_task_complete": True,
                "require_user_input": False,
                "content": "Không thể lấy kết quả từ agent.",
                "status": "error",
                "data": None,
            }
        else:
            final_response["is_task_complete"] = True
            final_response["content"] = str(final_response.get("content") or "")

        yield final_response


    def _get_agent_response_from_state(self, config: dict, runnable: Any, result: Any = None) -> dict:
        try:
            # Lấy state từ checkpointer
            # state = runnable.checkpointer.get(config["configurable"]["thread_id"])
            state = runnable.checkpointer.get(config)
            if state:
                # messages thường nằm trong state['channel_values']['messages']
                messages = state.get('channel_values', {}).get('messages', [])
                if messages:
                    last_message = messages[-1]
                    if isinstance(last_message, AIMessage):
                        return {
                            "content": last_message.content,
                            "is_task_complete": True,
                            "require_user_input": False,
                            "status": "completed",
                            "data": getattr(last_message, "additional_kwargs", {}).get("data")
                        }
            # Fallback nếu không có
            return {
                "content": "",
                "is_task_complete": True,
                "require_user_input": False,
                "status": "error",
                "data": None
            }
        except Exception as e:
            logger.error(f"Error getting response from state: {e}")
            return {
                "content": "",
                "is_task_complete": True,
                "require_user_input": False,
                "status": "error",
                "data": None
            }