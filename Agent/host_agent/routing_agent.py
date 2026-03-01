import asyncio
import json
import logging
import os
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv

from common.client import A2ACardResolver
from common.types import (
    AgentCard,
    TaskSendParams,
    Part,
    SendTaskResponse,
    Task,
    Message,
    TextPart,
)

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.tool_context import ToolContext

from remote_agent_connection import RemoteAgentConnections, TaskUpdateCallback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

AGENT_KEYWORD = {
    "diagnose" : {
        "keywords" : [
            "đau", "bị", "triệu chứng", "bệnh gì"
        ],
    },
}

class RoutingAgent:
    """
    Host Agent - Điều phối các agent y tế chuyên biệt.
    """

    def __init__(self, task_callback: TaskUpdateCallback | None = None):
        self.task_callback = task_callback
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ''

    async def _async_init_components(self, remote_agent_addresses: list[str]) -> None:
        """Khởi tạo connections đến remote agents"""
        for address in remote_agent_addresses:
            try:
                # Try to connect directly without using A2ACardResolver
                logger.info(f"Attempting to connect to: {address}")
                
                # Create a mock agent card if we can't fetch it
                try:
                    async with httpx.AsyncClient(timeout=10) as client:
                        response = await client.get(f"{address}/.well-known/agent.json")
                        if response.status_code == 200:
                            card_data = response.json()
                            card = AgentCard(**card_data)
                            remote_connection = RemoteAgentConnections(
                                agent_card=card, agent_url=address
                            )
                            self.remote_agent_connections[card.name] = remote_connection
                            self.cards[card.name] = card
                            logger.info(f"✅ Connected to: {card.name} at {address}")
                            continue
                except Exception as fetch_error:
                    logger.warning(f"Could not fetch agent card from {address}: {fetch_error}")
                
                # Fallback: Create a mock card
                agent_name = address.split("://")[-1].replace(".", "_").replace(":", "_")
                mock_card = AgentCard(
                    name=f"Agent_{agent_name}",
                    description=f"Remote agent at {address}",
                    url=address,
                    version="1.0.0",
                )
                remote_connection = RemoteAgentConnections(
                    agent_card=mock_card, agent_url=address
                )
                self.remote_agent_connections[mock_card.name] = remote_connection
                self.cards[mock_card.name] = mock_card
                logger.info(f"⚠️ Created mock card for: {address}")
                
            except Exception as e:
                logger.error(f"Failed to connect {address}: {e}", exc_info=True)

        # Build agents info string
        if self.cards:
            agent_info = []
            for card in self.cards.values():
                agent_info.append({
                    "name": card.name,
                    "description": card.description,
                    "skills": [s.description for s in card.skills] if card.skills else []
                })
            
            self.agents = json.dumps(agent_info, ensure_ascii=False, indent=2)
            logger.info(f"🔋 Available agents:\n{self.agents}")
        else:
            logger.warning("No remote agents connected!")
            self.agents = "[]"

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
        task_callback: TaskUpdateCallback | None = None,
    ) -> 'RoutingAgent':
        """Factory method với async init"""
        instance = cls(task_callback)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self) -> Agent:
        """Tạo ADK Agent instance"""
        gemini_model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        return Agent(
            model=gemini_model,
            name='Host_Agent',
            instruction=self.root_instruction,
            before_model_callback=self.before_model_callback,
            description='Host Agent for medical multi-agent orchestration',
            tools=[self.send_task_to_agent],
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """
        Fixed prompt - ngăn LLM tự tạo biến không tồn tại
        """
        state = context.state
        active_agent = self._check_active_agent(state)
        
        return f"""
    **VAI TRÒ CỦA BẠN:**
    Bạn là Host Agent - trợ lý y tế thông minh điều phối các agent chuyên biệt.

    **CÁC AGENT KHẢ DỤNG:**
    {self.agents}

    **AGENT ĐANG ACTIVE:** {active_agent}

    ---

    ## NGUYÊN TẮC HOẠT ĐỘNG

    ### 1. PHÂN TÍCH REQUEST
    Xác định loại request:
    - **Single task**: 1 agent → "Tôi bị sốt"
    - **Multi-task**: nhiều agents → "Tôi bị sốt và muốn biết chi phí"

    ### 2. SỬ DỤNG TOOL

    **Tool duy nhất:**
    ```
    send_task_to_agent(agent_name, task_description)
    ```

    **⚠️ QUAN TRỌNG VỀ TOOL RESPONSE:**
    - Tool này trả về **TEXT THUẦN** (string)
    - **KHÔNG** phải object, **KHÔNG** có properties
    - **KHÔNG** được viết `{{response.diagnosis}}` hay `{{response1 summary}}`
    - Response là text hoàn chỉnh, đọc và present lại tự nhiên

    ### 3. CÁCH XỬ LÝ RESPONSE ĐÚNG

    **❌ SAI - Không làm thế này:**
    ```python
    response1 = send_task_to_agent("Agent chẩn đoán", "...")
    # SAI: response1.diagnosis ❌
    # SAI: {{response1 summary}} ❌
    # SAI: extract from response1 ❌
    ```

    **✅ ĐÚNG - Làm như thế này:**
    ```python
    # Bước 1: Gọi tool
    response1 = send_task_to_agent("Agent chẩn đoán", "Phân tích triệu chứng: đau đầu")

    # Bước 2: Đọc response1 (là text string)
    # VD response1 = "Có thể là đau nửa đầu. Nên khám nếu kéo dài >3 ngày."

    # Bước 3: Present lại cho user
    "Dựa trên triệu chứng, bạn có thể bị đau nửa đầu. 
    Tôi khuyên bạn nên đi khám nếu đau kéo dài hơn 3 ngày.
    Bạn có muốn đặt lịch không?"
    ```

    ---

    ## CASE STUDIES CHI TIẾT

    ### Case 1: Single Task - Chẩn đoán
    ```
    User: "Tôi bị sốt 38 độ"

    Bước 1 - Gọi tool:
    diagnosis_response = send_task_to_agent(
        "Agent chẩn đoán",
        "Phân tích triệu chứng: Bệnh nhân bị sốt 38 độ"
    )

    Bước 2 - diagnosis_response là string, VD:
    "Sốt 38 độ có thể do nhiễm virus hoặc nhiễm trùng. 
    Theo dõi 24h, nếu sốt tăng hoặc có thêm triệu chứng thì nên khám."

    Bước 3 - Trình bày cho user:
    "Dựa trên triệu chứng sốt 38 độ, có thể bạn bị nhiễm virus hoặc nhiễm trùng nhẹ.
    Bạn nên:
    - Theo dõi nhiệt độ trong 24h
    - Uống nhiều nước, nghỉ ngơi
    - Nếu sốt tăng cao hoặc có thêm triệu chứng (ho, đau đầu...), hãy đi khám

    Bạn có muốn tôi đặt lịch khám ngay không?"
    ```

    ### Case 2: Multi-Task - Chẩn đoán + Chi phí
    ```
    User: "Tôi bị đau bụng và muốn biết chi phí gói khám nam"

    Bước 1 - Gọi agent chẩn đoán:
    diag_text = send_task_to_agent(
        "Agent chẩn đoán",
        "Phân tích triệu chứng: đau bụng"
    )
    # diag_text = "Đau bụng có thể do viêm dạ dày hoặc rối loạn tiêu hóa..."

    Bước 2 - Gọi agent chi phí:
    cost_text = send_task_to_agent(
        "Agent chi phí",
        "Chi phí gói khám tổng quát nam giới"
    )
    # cost_text = "Gói khám nam cơ bản: 1.5 triệu đồng, bao gồm..."

    Bước 3 - ĐỌC CẢ 2 text responses và tổng hợp:
    "Về triệu chứng của bạn:
    Đau bụng có thể do viêm dạ dày hoặc rối loạn tiêu hóa. 
    Bạn nên khám chuyên khoa Tiêu hóa.

    Về chi phí:
    Gói khám tổng quát nam cơ bản có giá 1.5 triệu đồng, 
    bao gồm khám lâm sàng và các xét nghiệm cơ bản.

    Gói này phù hợp với tình trạng của bạn vì có kiểm tra tiêu hóa.
    Bạn có muốn đặt lịch khám không?"
    ```

    ### Case 3: Booking - Thu thập thông tin
    ```
    User: "Đặt lịch khám"

    # Thiếu info → hỏi trước
    "Tôi sẽ giúp bạn đặt lịch. Cho tôi biết:
    - Họ tên?
    - Ngày muốn khám?
    - Giờ nào tiện?
    - Email để nhận xác nhận?"

    # Sau khi user cung cấp đủ
    User: "Tên: Nguyễn Văn A, Ngày: 25/10, Giờ: 10h, Email: vana@email.com"

    booking_text = send_task_to_agent(
        "Agent đặt lịch",
        "Đặt lịch khám: Họ tên: Nguyễn Văn A, Ngày: 25/10/2025, Giờ: 10:00, Email: vana@email.com"
    )

    # booking_text = "Đặt lịch thành công! Mã: BK001, Bác sĩ: BS. Trần B..."

    # Present:
    "✅ Đặt lịch thành công!
    Họ tên: Nguyễn Văn A
    Ngày giờ: 25/10/2025 - 10:00
    Mã đặt lịch: BK001
    Bác sĩ: BS. Trần B

    Email xác nhận đã gửi đến vana@email.com
    Vui lòng đến trước 15 phút."
    ```

    ---

    ## QUY TẮC BẮT BUỘC

    ### ✅ LUÔN LUÔN:
    1. **Tool trả về TEXT**, không phải object
    2. **ĐỌC response text** và hiểu nội dung
    3. **PRESENT lại** bằng lời của bạn, tự nhiên
    4. **KHÔNG tự tạo** biến hay properties (`response.field` ❌)
    5. **KHÔNG viết** template string với biến (`{{response}}` ❌)
    6. Task description phải có **ĐẦY ĐỦ context**

    ### ❌ KHÔNG BAO GIỜ:
    1. Viết `response1.diagnosis` hoặc `.field` bất kỳ
    2. Viết `{{response summary}}` hoặc template tương tự
    3. Hỏi "Tôi nên gọi agent nào?"
    4. Copy nguyên văn response không suy nghĩ
    5. Delegate thiếu context

    ---

    ## XỬ LÝ NHIỀU RESPONSES

    **Khi có 2+ responses:**

    ```python
    # Gọi tuần tự
    text1 = send_task_to_agent("Agent 1", "task 1 với đầy đủ context")
    text2 = send_task_to_agent("Agent 2", "task 2 với đầy đủ context")

    # Bây giờ bạn có:
    # - text1: string chứa response từ Agent 1
    # - text2: string chứa response từ Agent 2

    # Đọc và hiểu cả 2, rồi tổng hợp:
    "Dựa trên phân tích:

    [Tóm tắt text1 bằng lời của bạn]

    [Tóm tắt text2 bằng lời của bạn]

    [Kết nối logic giữa 2 responses]

    [Follow-up question]"
    [TỔNG HỢP KẾT QUẢ\]
    ```

    **VÍ DỤ CỤ THỂ:**
    ```
    # Đã có:
    text1 = "Triệu chứng đau đầu có thể là migraine..."
    text2 = "Gói khám thần kinh: 800k"

    # Tổng hợp:
    "Về triệu chứng đau đầu của bạn, có thể đây là cơn migraine. 
    Tôi khuyên bạn nên khám chuyên khoa Thần kinh.

    Về chi phí, gói khám thần kinh có giá 800.000đ, bao gồm 
    khám lâm sàng và tư vấn chuyên khoa.

    Bạn có muốn đặt lịch khám ngay không?"
    ```

    ---

    ## DEBUG TIPS

    Nếu bạn thấy mình muốn viết:
    - `response.something` → ❌ STOP! Response là text, không có properties
    - `{{variable}}` → ❌ STOP! Không dùng template, viết text bình thường
    - `extract from response` → ❌ STOP! Đọc và hiểu, rồi viết lại

    Luôn nhớ:
    ```
    Tool output = Plain text string
    Your job = Read it, understand it, present it naturally
    ```

    ---

    ## MỤC TIÊU CUỐI CÙNG

    User trò chuyện với bạn như với bác sĩ tư vấn thật:
    - Thân thiện, chuyên nghiệp
    - Hiểu và giải thích rõ ràng
    - Chủ động giải quyết vấn đề
    - Tổng hợp thông tin mạch lạc
    - Quan tâm đến trải nghiệm

    **Agent làm việc ẩm thầm phía sau, user chỉ thấy BẠN.**
    """

    def _check_active_agent(self, state: dict) -> str:
        """Check active agent from state"""
        if (
            'session_id' in state
            and 'session_active' in state
            and state['session_active']
            and 'active_agent' in state
        ):
            return state['active_agent']
        return 'None'

    def before_model_callback(
        self,
        callback_context: CallbackContext,
        llm_request,
    ):
        """Initialize session"""
        state = callback_context.state
        if 'session_active' not in state or not state['session_active']:
            if 'session_id' not in state:
                state['session_id'] = str(uuid.uuid4())
            state['session_active'] = True
            logger.info(f"🆕 New session: {state['session_id']}")

    def list_remote_agents(self):
        """List available agents"""
        if not self.cards:
            return []
        
        return [
            {'name': card.name, 'description': card.description}
            for card in self.cards.values()
        ]

    async def send_task_to_agent(
        self,
        agent_name: str,
        task_description: str,
        tool_context: ToolContext,
    ) -> str:
        """
        Gửi task đến remote agent.
        
        Args:
            agent_name: Tên agent cần delegate (VD: "Agent chẩn đoán")
            task_description: Mô tả CHI TIẾT task + context đầy đủ
            tool_context: Tool context
        
        Returns:
            Response text từ agent (hoặc error message)
        """
        logger.info(f"📤 Delegating to: {agent_name}")
        logger.debug(f"Task: {task_description[:100]}...")
        
        # Validate agent exists
        if agent_name not in self.remote_agent_connections:
            available = list(self.remote_agent_connections.keys())
            error_msg = f"❌ Agent '{agent_name}' không tồn tại. Available: {available}"
            logger.error(error_msg)
            return error_msg
        
        # Update state
        state = tool_context.state
        state['active_agent'] = agent_name
        
        # Get task_id and context_id
        task_id = state.get('task_id')
        context_id = state.get('context_id', str(uuid.uuid4()))
        message_id = str(uuid.uuid4())
        
        # Build task params
        task_params = TaskSendParams(
            id=task_id or str(uuid.uuid4()),
            sessionId=context_id,
            message=Message(
                role='user',
                parts=[TextPart(text=task_description)],
                metadata={'message_id': message_id}
            ),
        )
        
        # Send to agent
        try:
            client = self.remote_agent_connections[agent_name]
            send_response: SendTaskResponse = await client.send_task(task_params)
            
            logger.debug(f"Response: {send_response.model_dump_json(exclude_none=True, indent=2)}")
            
            # Check for errors
            if send_response.error:
                error_msg = f"❌ Agent error: {send_response.error}"
                logger.error(error_msg)
                return error_msg
            
            # Extract result
            if not send_response.result:
                return "❌ Không nhận được response từ agent"
            
            task_result: Task = send_response.result
            response_text = self._extract_response_text(task_result)
            
            logger.info(f"✅ Got response from {agent_name}: {len(response_text)} chars")
            return response_text
            
        except Exception as e:
            error_msg = f"❌ Error calling {agent_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def _extract_response_text(self, task: Task) -> str:
        """Extract text from Task result"""
        if not task.artifacts:
            return "Không có phản hồi từ agent"
        
        texts = []
        for artifact in task.artifacts:
            for part in artifact.parts:
                if hasattr(part, 'text') and part.text:
                    texts.append(part.text)
                elif isinstance(part, dict) and 'text' in part:
                    texts.append(part['text'])
        
        return "\n".join(texts) if texts else "Không có phản hồi từ agent"


def _get_initialized_routing_agent_sync() -> Agent:
    """Synchronously creates and initializes the RoutingAgent."""

    async def _async_main() -> Agent:
        routing_agent_instance = await RoutingAgent.create(
            remote_agent_addresses=[
                os.getenv("DIAGNOSE_AGENT_URL", "http://localhost:10001"),
                os.getenv("BOOKING_AGENT_URL", "http://localhost:10003"),
                os.getenv("COST_AGENT_URL", "http://localhost:10002"),
            ]
        )
        return routing_agent_instance.create_agent()

    try:
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if 'asyncio.run() cannot be called from a running event loop' in str(e):
            logger.debug(
                'Warning: Could not initialize RoutingAgent with asyncio.run(): %s',
                e,
            )
        raise


root_agent = _get_initialized_routing_agent_sync()