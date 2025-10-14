import gradio as gr
from .agent_g import ServiceAgent  
import uuid
import time


# 🎨 CSS để tăng kích thước avatar


# Khởi tạo agent
agent = ServiceAgent()

# Mỗi người dùng sẽ có 1 session riêng biệt (thread_id riêng)
def get_session_id():
    return str(uuid.uuid4())

def chat_stream(message, history, session_id):
    """
    message: tin nhắn mới từ người dùng
    history: danh sách [(user, bot)]
    session_id: ID lưu hội thoại agent
    """
    # --- Bước 1: Hiển thị tin nhắn người dùng ngay ---
    history.append((message, ""))
    yield "", history  # hiện ngay dòng người dùng vừa nhập

    # --- Bước 2: Gọi agent để lấy phản hồi ---
    try:
        response = agent.stream(message, session_id)
        full_reply = response["messages"][-1].content
    except Exception as e:
        full_reply = f"Lỗi: {str(e)}"

    # --- Bước 3: Stream dần phản hồi ---
    partial = ""
    for ch in full_reply:
        partial += ch
        history[-1] = (message, partial)
        time.sleep(0.010)
        yield "", history

    # --- Bước 4: Trả kết quả cuối cùng ---
    yield "", history


# 🚀 Giao diện Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## Agent Tư vấn Dịch vụ")

    session_id = gr.State(get_session_id())

    chatbot_ui = gr.Chatbot(
        height=360,
        show_label=False,
        bubble_full_width=False,
        show_copy_button=False,
        avatar_images=("system_image/cr7.jpg", "system_image/m10.jpg")  # ảnh đại diện user/bot
    )

    message_box = gr.Textbox(
        placeholder="Nhập tin nhắn và nhấn Enter để gửi...",
        label="Tin nhắn",
        show_label=True,
    )

    clear_btn = gr.Button("🧹 Xóa hội thoại")

    # Khi nhấn Enter -> gửi tin nhắn (stream)
    message_box.submit(
        chat_stream,                   # hàm xử lý
        [message_box, chatbot_ui, session_id],  # input
        [message_box, chatbot_ui],              # output
        queue=True
    )

    # Xóa UI (và tạo session mới nếu cần)
    def reset_agent():
        new_session_id = get_session_id()
        return new_session_id, []

    clear_btn.click(reset_agent, None, [session_id, chatbot_ui])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=1010, share=True)