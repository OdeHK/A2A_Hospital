import streamlit as st
from langchain_core.messages import HumanMessage
from service_agent.agents import build_graph, load_packages, data

# Khởi tạo đồ thị
graph = build_graph(load_packages(data))

st.set_page_config(page_title="Healthcare Service Agent", layout="wide")

st.title("🩺 Healthcare Service Agent")

# Sidebar chọn thread_id
thread_id = st.sidebar.text_input("Thread ID", value="1")
config = {"configurable": {"thread_id": thread_id}}

# Vùng chat
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("💬 Chat với Agent")

# Hiển thị lịch sử hội thoại
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Ô nhập
if prompt := st.chat_input("Nhập câu hỏi của bệnh nhân..."):
    # Hiển thị tin nhắn người dùng
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Gọi agent
    input_mes = HumanMessage(content=prompt)

    response_text = ""
    for chunk in graph.stream({"messages": [input_mes]}, config, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        if last_msg.type == "ai":
            response_text = last_msg.content

    # Hiển thị tin nhắn assistant
    st.chat_message("assistant").write(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})
