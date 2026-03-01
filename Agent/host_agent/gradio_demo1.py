"""
Gradio Demo Interface for A2A Medical Agent System
Kết nối với Host Agent và hiển thị multi-agent orchestration
"""

import asyncio
import json
import uuid
from typing import List, Dict, Any, Optional
import gradio as gr
import httpx
from datetime import datetime

# Configuration
HOST_AGENT_URL = "http://127.0.0.1:8083"
TIMEOUT = httpx.Timeout(120.0, connect=10.0)

MAX_RETRIES = 3
RETRY_DELAY = 2.0

class A2AAgentChat:
    def __init__(self, host_url: str = HOST_AGENT_URL):
        self.host_url = host_url
        self.session_id = None
        self.conversation_history = []
        self._client: Optional[httpx.AsyncClient] = None
        self.current_active_agent = "None"
        self.available_agents = {}  # Track which agents are available
        
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=TIMEOUT)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def _reset_session(self):
        """Reset session - create new session ID"""
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.current_active_agent = "None"
    
    async def _check_agent_health(self) -> Dict[str, Any]:
        """Check agent health with retry"""
        for attempt in range(MAX_RETRIES):
            try:
                client = await self.get_client()
                response = await client.get(
                    f"{self.host_url}/.well-known/agent.json",
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    card = response.json()
                    return {
                        "status": "healthy",
                        "name": card.get("name", "Unknown"),
                        "description": card.get("description", ""),
                        "version": card.get("version", "")
                    }
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                    
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }
                
            except httpx.ConnectError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return {
                    "status": "error",
                    "error": f"Cannot connect to {self.host_url}"
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                return {"status": "error", "error": str(e)}
        
        return {"status": "error", "error": "Max retries exceeded"}
    
    async def _send_message_streaming(self, message: str) -> List[Dict[str, Any]]:
        """Send message with error handling"""
        if not self.session_id:
            self._reset_session()
        
        task_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tasks/sendSubscribe",
            "params": {
                "id": task_id,
                "sessionId": self.session_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}]
                }
            }
        }
        
        events = []
        
        try:
            client = await self.get_client()
            
            async with client.stream(
                'POST',
                self.host_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    return [{
                        "error": True,
                        "message": f"HTTP {response.status_code}: {response.text[:200]}"
                    }]
                
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        try:
                            event_data = json.loads(line[6:])
                            events.append(event_data)
                        except json.JSONDecodeError:
                            continue
                            
        except httpx.ConnectError:
            return [{
                "error": True,
                "message": f"Connection failed. Is Host Agent running at {self.host_url}?"
            }]
        except httpx.TimeoutException:
            return [{
                "error": True,
                "message": "Request timeout. The agent might be overloaded."
            }]
        except Exception as e:
            return [{
                "error": True,
                "message": f"Error: {str(e)}"
            }]
        
        return events if events else [{
            "error": True,
            "message": "No response received"
        }]
    
    def _extract_response_text(self, events: List[Dict[str, Any]]) -> str:
        """Extract text from events with better error handling"""
        
        # Check for errors first
        if events and events[0].get("error"):
            self.current_active_agent = "None"
            return f"❌ {events[0].get('message', 'Unknown error')}"
        
        responses = []
        
        # Try to detect active agent from events
        for event in events:
            result = event.get("result")
            if not result:
                continue
            
            # Status messages
            if "status" in result:
                status = result["status"]
                if status.get("message"):
                    parts = status["message"].get("parts", [])
                    for part in parts:
                        if part.get("type") == "text" and part.get("text"):
                            text = part["text"].strip()
                            if text:
                                responses.append(text)
                                # Detect agent name from response
                                self._detect_active_agent(text)
            
            # Artifacts
            if "artifact" in result:
                artifact = result["artifact"]
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("type") == "text" and part.get("text"):
                        responses.append(f"📎 {part['text']}")
        
        if responses:
            # Remove duplicates
            seen = set()
            unique = []
            for r in responses:
                if r not in seen:
                    seen.add(r)
                    unique.append(r)
            return "\n\n".join(unique)
        
        self.current_active_agent = "None"
        return "⚠️ Không nhận được phản hồi. Vui lòng thử lại."
    
    def _detect_active_agent(self, text: str):
        """Detect which agent is active from response text"""
        text_lower = text.lower()
        
        # Check for delegation patterns
        if "delegating to" in text_lower or "gửi task đến" in text_lower:
            if "diagnose" in text_lower or "chẩn đoán" in text_lower:
                self.current_active_agent = "🩺 Agent Chẩn đoán"
            elif "booking" in text_lower or "đặt lịch" in text_lower:
                self.current_active_agent = "📅 Agent Đặt lịch"
            elif "cost" in text_lower or "chi phí" in text_lower:
                self.current_active_agent = "💰 Agent Chi phí"
        # Check for agent mentions in text
        elif "agent chẩn đoán" in text_lower or "diagnose" in text_lower:
            self.current_active_agent = "🩺 Agent Chẩn đoán"
        elif "agent đặt lịch" in text_lower or "booking" in text_lower:
            self.current_active_agent = "📅 Agent Đặt lịch"
        elif "agent chi phí" in text_lower or "cost" in text_lower:
            self.current_active_agent = "💰 Agent Chi phí"
        # Check for specific keywords
        elif any(word in text_lower for word in ["triệu chứng", "bệnh", "đau", "sốt"]):
            self.current_active_agent = "🩺 Agent Chẩn đoán"
        elif any(word in text_lower for word in ["đặt lịch", "appointment", "booking"]):
            self.current_active_agent = "📅 Agent Đặt lịch"
        elif any(word in text_lower for word in ["giá", "chi phí", "cost", "price"]):
            self.current_active_agent = "💰 Agent Chi phí"
        else:
            self.current_active_agent = "🏥 Host Agent"
    
    async def chat(self, message: str, history: List[List[str]]) -> tuple:
        """Xử lý chat message và trả về history + active agent status"""
        if not message.strip():
            return history, "", self._get_agent_status()
        
        # Thêm user message vào history
        history.append([message, None])
        
        # Set processing state
        self.current_active_agent = "⏳ Processing..."
        
        # Gửi request và nhận streaming events
        events = await self._send_message_streaming(message)
        
        # Extract response
        response_text = self._extract_response_text(events)
        
        # Cập nhật response trong history
        history[-1][1] = response_text
        
        # Lưu vào conversation history
        self.conversation_history.append({
            "user": message,
            "agent": response_text,
            "timestamp": datetime.now().isoformat(),
            "events_count": len(events)
        })
        
        return history, "", self._get_agent_status()
    
    def _get_agent_status(self) -> str:
        """Get current agent status for display"""
        # Highlight active agent
        agents_list = """### 📋 Agents có sẵn:
"""
        
        agents = [
            ("diagnose", "🩺 Agent Chẩn đoán", "Agent chẩn đoán"),
            ("booking", "📅 Agent Đặt lịch", "Agent đặt lịch"),
            ("cost", "💰 Agent Chi phí", "Agent chi phí")
        ]
        
        for key, icon_name, check_name in agents:
            is_available = self.available_agents.get(key, True)
            is_active = check_name.lower() in self.current_active_agent.lower()
            
            if not is_available:
                agents_list += f"- {icon_name} ❌ *Không khả dụng*\n"
            elif is_active:
                agents_list += f"- **{icon_name}** ⚡ *Đang xử lý*\n"
            else:
                agents_list += f"- {icon_name} ✅\n"
        
        return f"""### 🤖 Agent đang hoạt động

**{self.current_active_agent}**

---

{agents_list}

---

**Session ID:** `{self.session_id or 'Chưa bắt đầu'}`  
**Tin nhắn:** {len(self.conversation_history)}
"""
    
    async def check_agents_availability(self):
        """Check which agents are actually available by querying Host Agent"""
        try:
            client = await self.get_client()
            # Try to get agent info from Host Agent
            response = await client.get(
                f"{self.host_url}/.well-known/agent.json",
                timeout=10.0
            )
            
            if response.status_code == 200:
                # Assume all agents are available if Host Agent is up
                # In real implementation, Host Agent should expose this info
                self.available_agents = {
                    "diagnose": True,
                    "booking": True, 
                    "cost": True
                }
            else:
                self.available_agents = {
                    "diagnose": False,
                    "booking": False,
                    "cost": False
                }
        except Exception as e:
            print(f"Could not check agents: {e}")
            self.available_agents = {
                "diagnose": False,
                "booking": False,
                "cost": False
            }
    
    async def get_initial_status(self) -> str:
        """Lấy status ban đầu của agent"""
        health = await self._check_agent_health()
        await self.check_agents_availability()
        
        if health["status"] == "healthy":
            # Build agents list with availability status
            agents_status = ""
            agents_info = [
                ("diagnose", "🩺 Agent Chẩn đoán", "Phân tích triệu chứng"),
                ("booking", "📅 Agent Đặt lịch", "Đặt lịch khám bệnh"),
                ("cost", "💰 Agent Chi phí", "Tra cứu giá dịch vụ")
            ]
            
            for key, name, desc in agents_info:
                if self.available_agents.get(key, False):
                    agents_status += f"- {name} ✅\n"
                else:
                    agents_status += f"- {name} ❌ *Không khả dụng*\n"
            
            return f"""### 🤖 Agent đang hoạt động

**Không có agent nào**

---

### 📋 Agents có sẵn:
{agents_status}

---

**Hệ thống:** ✅ Sẵn sàng  
**Host Agent:** {health['name']}  
**Session:** Chưa bắt đầu  
**Tin nhắn:** 0
"""
        else:
            return f"""### ❌ Lỗi kết nối

**Error:** {health.get('error', 'Unknown error')}

Hãy đảm bảo Host Agent đang chạy tại {self.host_url}
"""


# Initialize chat instance
chat_instance = A2AAgentChat()


# Gradio Interface
def create_demo():
    """Tạo Gradio demo interface"""
    
    with gr.Blocks(
        title="A2A Medical Agent Chat",
        theme=gr.themes.Soft(),
        css="""
        .chatbot-container {
            height: 600px !important;
        }
        .agent-status {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 16px;
        }
        .examples-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🏥 Medical Agent System
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 Agent Chat",
                    height=500,
                    show_copy_button=True,
                    elem_classes=["chatbot-container"]
                )
                
                # Input (Enter to send)
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Nhập câu hỏi và nhấn Enter... (VD: Tôi bị đau đầu và muốn đặt lịch khám)",
                    lines=1,
                    max_lines=5
                )
                
                # Example queries
                with gr.Accordion("📋 Câu hỏi mẫu", open=False):
                    gr.Markdown("""
                    ```
                    • Tôi muốn đặt lịch khám.
                    • Tôi tên là Nguyễn Văn A, muốn khám lúc 9h sáng. Email của tôi là email@gmail.com
                    • Tôi bị đau bụng, nôn óa và sốt 3 ngày nay thì bị bệnh gì?
                    • Cho tôi chi phí gói khám tổng quát cơ bản của nam?
                    • Kiểm tra sức khỏe tổng quát có những gói nào?
                    ```
                    """, elem_classes=["examples-box"])
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                    reset_btn = gr.Button("🔄 New Session", size="sm")
            
            with gr.Column(scale=1):
                # Agent status panel - CHỈ HIỆN AGENT ĐANG HOẠT ĐỘNG
                agent_status = gr.Markdown(
                    "🔄 Đang tải...",
                    elem_classes=["agent-status"]
                )
        
        # Event handlers
        async def send_message(message, history):
            new_history, empty_text, status = await chat_instance.chat(message, history)
            return new_history, empty_text, status
        
        async def get_initial_status():
            return await chat_instance.get_initial_status()
        
        def clear_chat():
            return [], ""
        
        def reset_session():
            chat_instance._reset_session()
            return [], "", chat_instance._get_agent_status()
        
        # Connect events - ENTER ĐỂ GỬI
        msg_input.submit(
            fn=send_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, agent_status]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input]
        )
        
        reset_btn.click(
            fn=reset_session,
            outputs=[chatbot, msg_input, agent_status]
        )
        
        # Load initial status
        demo.load(
            fn=get_initial_status,
            outputs=agent_status
        )
    
    return demo


if __name__ == "__main__":
    # Tạo và chạy demo
    demo = create_demo()
    
    print("------------------------------------")
    print("Starting A2A Medical Agent Demo")
    print("------------------------------------")
    print(f"Host Agent URL: {HOST_AGENT_URL}")
    print(f"Gradio Interface: http://localhost:7860")
    print("------------------------------------")
    print("Nhấn ENTER để gửi tin nhắn")
    print("------------------------------------")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # ⭐ PUBLIC URL
        show_error=True
    )