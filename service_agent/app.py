import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from langchain_core.messages import HumanMessage
from service_agent.agents import build_graph
from time import time

# --- Cấu hình trang ---
st.set_page_config(page_title="Healthcare Service Agent", layout="wide")
st.title("🩺 Healthcare Service Agent")

# --- Khởi tạo graph và memory trong session_state ---
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "messages" not in st.session_state:
    st.session_state.messages = []

graph = st.session_state.graph

# --- Sidebar chọn thread_id ---
thread_id = st.sidebar.text_input("Thread ID", value="1")
config = {"configurable": {"thread_id": thread_id}}

# --- Vùng chat ---
st.subheader("💬 Chat với Agent")

# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Ô nhập chat
if prompt := st.chat_input("Nhập câu hỏi của bệnh nhân..."):
    # Hiển thị ngay tin nhắn người dùng
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gọi agent
    input_mes = HumanMessage(content=prompt)
    response_text = ""
    start_time = time()

    for chunk in graph.stream({"messages": [input_mes]}, config, stream_mode="updates"):
        if "answer_with_retrival" in chunk or "answer_without_retrival" in chunk:
            node_key = "answer_with_retrival" if "answer_with_retrival" in chunk else "answer_without_retrival"
            messages = chunk[node_key].get("messages", [])
            if messages:
                last_msg = messages[-1]
                if last_msg.type == "ai":
                    end_time = time()
                    elapsed = end_time - start_time
                    response_text = last_msg.content
                    st.chat_message("assistant").write(
                        f"{response_text}\n\n---\n⏱️ Thời gian phản hồi: {elapsed:.2f} giây"
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response_text}
                    )

    # Debug: in summary hiện tại
    # state = graph.get_state(config)
    # st.sidebar.write("📌 Current summary:", state.get("summary"))
