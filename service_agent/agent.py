from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field
from langchain_groq import ChatGroq

from typing import Any, Dict, AsyncIterable, Literal

from .utils import get_package_by_id, convert_packages_to_str, PACKAGES
from .prompts import *

import os
import re
import json
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ-API-KEY")
# GROQ_API_KEY = 'GROQ-API-KEY'

llm_qwen3_32b_wr = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=GROQ_API_KEY
)
llm_qwen3_32b_nr = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    reasoning_effort='none',
    reasoning_format="hidden",
    api_key=GROQ_API_KEY
)

llm_llama_8b = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

class ListPackages(BaseModel):
    """Danh sách dịch vụ khám sức khỏe"""
    service_ids: set[int] = Field(
        description="Danh sách id dịch vụ, là các giá trị duy nhất, không trùng lặp. Nếu không chọn dịch vụ nào, trả về [0]",
    )

class HistoryStatus(BaseModel):
    """Trạng thái kiểm tra xem câu hỏi hiện tại có cần thêm nội dung bên ngoài hay chỉ dựa vào thông tin trong lịch sử chat là đủ"""
    decision: int = Field(description="0 nếu đủ thông tin trong lịch sử, 1 nếu cần retrieve thêm")

class ResponseFormat(BaseModel):
    """    Mô hình chuẩn hóa định dạng phản hồi của hệ thống.

    Thuộc tính:
    ----------
    status : Literal["input_required", "completed", "error"]
        Trạng thái phản hồi của hệ thống. Có 3 giá trị hợp lệ:
        - "input_required": hệ thống yêu cầu thêm dữ liệu đầu vào.
        - "completed": quá trình xử lý đã hoàn tất thành công.
        - "error": xảy ra lỗi trong quá trình xử lý.

    message : str
        Thông báo chi tiết đi kèm trạng thái, có thể mô tả yêu cầu đầu vào,
        kết quả xử lý, hoặc nội dung lỗi.
"""
    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str

def summarize_history(state: MessagesState):
    messages = state["messages"]
    
    sys_msg = SUMMARY_HISTORY
    response = llm_qwen3_32b_wr.invoke([SystemMessage(content=sys_msg)] + messages)
    cleaned = re.sub(r"<think>.*?</think>", "", response.content, flags=re.DOTALL)

    try:
        summary_json = json.loads(cleaned)
        summary_text = summary_json.get("summary", "Không có gì để tóm tắt.")
    except Exception:
        summary_text = cleaned[:250]

    summary_message = SystemMessage(content=f"[Tóm tắt hội thoại trước đó]: {summary_text}")
    state["messages"] = [summary_message]
    # Chỉ giữ summary, không append gì khác
    return {"messages": [summary_message]}

def filter_messages(state: MessagesState):
    messages = state["messages"]
    delete_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            source = msg.additional_kwargs.get("source")
            if source in ["check_history", "select_services"]:
                delete_messages.append(RemoveMessage(id=msg.id))

    return {"messages": delete_messages}

def check_history(state: MessagesState):
    """
    Cải thiện hàm check_history với xử lý tốt hơn
    """
    messages = state['messages']
    
    # Kiểm tra nếu không có lịch sử hoặc chỉ có câu hỏi đầu tiên
    if len(messages) <= 1:
        response = AIMessage(
            content="Check history - No sufficient history",
            additional_kwargs={"decision": 1, "source": "check_history", "reason": "insufficient_history"}
        )
        new_memory = state["messages"] + [response]  # state["messages"] lúc này chỉ là summary
        return {"messages": new_memory}
    
    # Tạo structured output với schema rõ ràng hơn
    try:
        sys_msg = CHECKING_HISTORY_INSTRUCTION
        structed_llm = llm_qwen3_32b_wr.with_structured_output(HistoryStatus)
        
        # Thêm context về task hiện tại
        enhanced_messages = [SystemMessage(content=sys_msg)] + messages
        
        response_structured = structed_llm.invoke(enhanced_messages)
        
        # Validate response
        decision = response_structured.decision if hasattr(response_structured, 'decision') else response_structured
        if decision not in [0, 1]:
            # Fallback nếu model trả về không đúng format
            decision = 1
            
        response = AIMessage(
            content=f"Check history - Decision: {decision}",
            additional_kwargs={
                "decision": decision,
                "source": "check_history",
                "model_response": response_structured.model_dump() if hasattr(response_structured, 'model_dump') else str(response_structured)
            }
        )
        
    except Exception as e:
        # Error handling - mặc định chọn retrieve để an toàn
        response = AIMessage(
            content="Check history - Error occurred, defaulting to retrieve",
            additional_kwargs={"decision": 1, "source": "check_history", "error": str(e)}
        )
    
    new_memory = state["messages"] + [response]  # state["messages"] lúc này chỉ là summary
    return {"messages": new_memory}


def route_message(state: MessagesState):
    last_message = state["messages"][-1]

    decision = last_message.additional_kwargs.get("decision", [])
    print('Decision:', decision)
    if decision == 0:
        return "answer_without_retrival"
    elif decision == 1:
        return "select_services"
    else:
        print("[LOG]❌Error: Output of 'check_history' state is invalid, the value must be 0 or 1")


def select_services(state:MessagesState):
    sys_msg = SELECTING_INSTRUCTION.format(packages=convert_packages_to_str(PACKAGES))
    messages = state["messages"]
    structed_llm = llm_qwen3_32b_wr.with_structured_output(ListPackages)
    response = structed_llm.invoke([SystemMessage(content=sys_msg)] + messages)
    print('Output of selecting services:', response)
    response = AIMessage(
        content="Chọn gói dịch vụ",
        additional_kwargs={**response.model_dump(), "source": "select_services"}
    )

    new_memory = state["messages"] + [response]  # state["messages"] lúc này chỉ là summary
    return {"messages": new_memory}


def answer_with_retrival(state:MessagesState):
    messages = state["messages"]
    selected_service_ids = messages[-1].additional_kwargs.get("service_ids", [])
    print('Selected ids:', selected_service_ids)
    selected_packages = get_package_by_id(ids=selected_service_ids)
    # print('***'*30)
    # print("Gói đã chọn:\n", selected_packages)
    # print('***'*30)

    sys_msg = ANSWER_WITH_SERVICE_INSTRUCTION.format(selected_packages=selected_packages)

    llm_struct = llm_qwen3_32b_wr.with_structured_output(ResponseFormat,method="json_schema")
    response = llm_struct.invoke([SystemMessage(content=sys_msg)] + state["messages"])

    # new_memory = state["messages"] + [response]  # state["messages"] lúc này chỉ là summary
    return {
        "messages": state["messages"] + [AIMessage(content=response.message)],
        "structured_response": response
    }

def answer_without_retrival(state:MessagesState):
    sys_msg = ANSWER_WITHOUT_SERVICE_INSTRUCTION
    llm_struct = llm_qwen3_32b_wr.with_structured_output(ResponseFormat,method="json_schema")
    response = llm_struct.invoke([SystemMessage(content=sys_msg)] + state["messages"])

    # new_memory = state["messages"] + [response]  # state["messages"] lúc này chỉ là summary
    return {
        "messages": state["messages"] + [AIMessage(content=response.message)],
        "structured_response": response
    }
    

def build_graph() -> StateGraph:
    within_thread_memory = MemorySaver()
    builder = StateGraph(MessagesState)


    builder.add_node('filter_messages', filter_messages)
    builder.add_node('summarize_history', summarize_history)
    builder.add_node('check_history', check_history)
    builder.add_node('select_services', select_services)
    builder.add_node('answer_with_retrival', answer_with_retrival)
    builder.add_node('answer_without_retrival', answer_without_retrival)

    # Kết nối node
    builder.add_edge(START, 'filter_messages')
    builder.add_edge('filter_messages', 'summarize_history')
    builder.add_edge('summarize_history', 'check_history')
    builder.add_conditional_edges('check_history', route_message)
    builder.add_edge('select_services', 'answer_with_retrival')
    
    builder.add_edge('answer_with_retrival', END)
    builder.add_edge('answer_without_retrival', END)

    
    graph = builder.compile(checkpointer=within_thread_memory)
    return graph


class ServiceAgent:
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]
    
    def __init__(self):
        self.graph = build_graph()
    
    def invoke(self, query, sessionId) -> str:
        config = {"configurable": {"thread_id": sessionId}}
        inputs = {"messages": [HumanMessage(content=query)]}
        self.graph.invoke(inputs, config)        
        return self.get_agent_response(config)
    
    async def stream(self, query, sessionId) -> AsyncIterable[Dict[str, Any]]:
        inputs = {"messages": [HumanMessage(content=query)]}
        config = {"configurable": {"thread_id": sessionId}}

        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False, # true
                    "require_user_input": False,
                    "content": "Looking up the exchange rates...",
                }

            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing the exchange rates..",
                }            
        
        yield self.get_agent_response(config)

        
    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)        
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(structured_response, ResponseFormat): 
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message
                }
            elif structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }
        
    

from pprint import pprint

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    print("Khởi tạo đồ thị...")
    graph = build_graph()
    print("Đồ thị khởi tạo xong.")
    while True:
        input_text = input("Nhập câu hỏi của bệnh nhân: ")
        if input_text.lower() in ['exit', 'quit', 'q']:
            break

        input_mes = HumanMessage(content=input_text)

        for chunk in graph.stream({"messages": input_mes}, config, stream_mode="values"):

            chunk["messages"][-1].pretty_print()