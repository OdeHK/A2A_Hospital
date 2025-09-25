import streamlit as st
import os, datetime, re, json, logging
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
from helpers import is_valid_booking_date as validate_date
from helpers import generate_time_slots
from db import find_available_doctors

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# ====== Setup ======
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY_1"))
model = GenerativeModel("gemini-2.5-flash-lite")

st.set_page_config(page_title="Chatbot Đặt lịch khám", layout="centered")
st.title("🤖 Chatbot Đặt lịch khám")

# ====== Init session_state ======
for key, default in {
    "messages": [],
    "booking_info": {},
    "show_booking_form": False,
    "ask_email": False,
    "show_email_form": False,
    "pending_booking": None,   # lưu danh sách bác sĩ rảnh để user xác nhận
    "disable_chat_input": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ====== Helper ======
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.write(content)

def toggle_chat_input(flag: bool):
    st.session_state.disable_chat_input = flag


# ====== Hàm phân loại intent ======
def classify_intent(user_text):
    prompt = f"""
    Bạn là hệ thống phân loại intent.
    Văn bản: "{user_text}"
    Nếu người dùng đang muốn đặt lịch khám hoặc đề cập tới việc hẹn khám => trả về: booking_request
    Nếu không => trả về: chitchat
    Chỉ trả về đúng 1 từ.
    """
    resp = model.generate_content(prompt)
    return resp.text.strip().lower()


# ====== Hàm trích xuất ngày giờ từ câu ======
def extract_datetime(user_text):
    today = datetime.date.today()
    prompt = f"""
    Hãy phân tích câu sau và tìm ngày + giờ mà người dùng muốn khám (nếu có).
    Trả về JSON dạng:
    {{"date": "YYYY-MM-DD" hoặc null, "time": "HH:MM" hoặc null}}.
    Hôm nay là {today.strftime('%Y-%m-%d')}.
    Câu: "{user_text}"
    """
    resp = model.generate_content(prompt)
    try:
        data = json.loads(re.search(r"\{.*\}", resp.text, re.S).group())
        return data.get("date"), data.get("time")
    except:
        return None, None


# ====== Hiển thị lịch sử chat ======
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ====== Input chat (ẩn nếu disable_chat_input=True) ======
if not st.session_state.disable_chat_input:
    user_input = st.chat_input("Nhập tin nhắn...")
else:
    user_input = None


# ====== Xử lý input ======
if user_input:
    add_message("user", user_input)

    # Nếu đang chờ user chọn bác sĩ
    if st.session_state.pending_booking:
        chosen = None
        user_lower = user_input.lower()
        matches = []

        # --- 1. Check theo tên (có thể khớp một phần) ---
        for doc in st.session_state.pending_booking["available"]:
            if doc["name"].lower() in user_lower or user_lower in doc["name"].lower():
                matches.append(doc)

        # --- 2. Nếu user nhập kèm số thứ tự ---
        if "đầu tiên" in user_lower or "1" in user_lower:
            chosen = st.session_state.pending_booking["available"][0]
        elif "2" in user_lower or "thứ hai" in user_lower:
            if len(st.session_state.pending_booking["available"]) >= 2:
                chosen = st.session_state.pending_booking["available"][1]
        elif "3" in user_lower or "thứ ba" in user_lower:
            if len(st.session_state.pending_booking["available"]) >= 3:
                chosen = st.session_state.pending_booking["available"][2]

        # --- 3. Nếu tìm thấy nhiều bác sĩ ---
        if not chosen and len(matches) == 1:
            chosen = matches[0]
        elif not chosen and len(matches) > 1:
            reply = "❌ Có nhiều bác sĩ trùng tên. Vui lòng chọn rõ hơn:\n"
            for i, doc in enumerate(matches, 1):
                reply += f"{i}. {doc['name']} (Phòng {doc['room']})\n"
            add_message("assistant", reply)
            st.stop()

        if chosen:
            booking_info = {
                "doctor_id": chosen["doctor_id"],
                "doctor_name": chosen["name"],
                "date": st.session_state.pending_booking["date"],
                "time": st.session_state.pending_booking["time"]
            }
            st.session_state.booking_info = booking_info
            # Chuyển date string thành object
            date_obj = datetime.datetime.strptime(booking_info["date"], "%Y-%m-%d").date()

            # Lấy tên thứ bằng tiếng Việt
            weekday_map = {
                0: "Thứ Hai",
                1: "Thứ Ba",
                2: "Thứ Tư",
                3: "Thứ Năm",
                4: "Thứ Sáu",
                5: "Thứ Bảy",
                6: "Chủ Nhật"
            }
            weekday = weekday_map[date_obj.weekday()]

            # Format lại ngày giờ đẹp
            date_str = date_obj.strftime("%d/%m/%Y")
            time_str = booking_info["time"][:5]  # cắt còn HH:MM

            reply = (
                f"✅ Bạn đã đặt lịch với bác sĩ {chosen['name']} "
                f"vào {time_str} {weekday}, {date_str}."
            )
            add_message("assistant", reply)

            # Sau đó hỏi email
            st.session_state.pending_booking = None
            st.session_state.ask_email = True
            toggle_chat_input(True)  # ẩn chat khi hỏi email
            st.rerun()
        else:
            add_message("assistant", "❌ Mình chưa hiểu bạn muốn chọn bác sĩ nào. Vui lòng nhập lại tên hoặc số thứ tự.")

    else:
        # Phân loại intent
        intent = classify_intent(user_input)
        logging.debug(f"Intent: {intent}")

        if intent == "chitchat":
            reply = model.generate_content(user_input).text
            add_message("assistant", reply)

        elif intent == "booking_request":
            date_extracted, time_extracted = extract_datetime(user_input)
            logging.debug(f"Extracted date: {date_extracted}, time: {time_extracted}")

            if date_extracted and time_extracted:
                if not validate_date(date_extracted):
                    reply = "❌ Ngày không hợp lệ!"
                    add_message("assistant", reply)
                else:
                    time_full = f"{time_extracted}:00" if len(time_extracted) == 5 else time_extracted
                    available = find_available_doctors(date_extracted, time_full)
                    if not available:
                        add_message("assistant", f"❌ Rất tiếc, không có bác sĩ nào rảnh vào {time_extracted} ngày {date_extracted}.")
                        add_message("assistant", "👉 Bạn vui lòng chọn lại thời gian khác:")

                        # Bật form nhập lại
                        st.session_state.show_booking_form = True
                        toggle_chat_input(True)
                        st.session_state.booking_info = {"date": date_extracted}
                        st.rerun()
                    else:
                        reply = "✅ Các bác sĩ có thể khám:\n"
                        for doc in available:
                            reply += f"- {doc['name']} (Phòng {doc['room']})\n"
                        reply += "\n👉 Bạn muốn đặt với bác sĩ nào?"

                        st.session_state.pending_booking = {
                            "date": date_extracted,
                            "time": time_full,
                            "available": available
                        }
                        add_message("assistant", reply)
            else:
                # Thiếu ngày/giờ -> bật form 
                st.session_state.show_booking_form = True 
                toggle_chat_input(True) # ẩn chat khi hiển thị form 
                # Lưu sẵn những gì đã bắt được 
                if date_extracted: 
                    st.session_state.booking_info["date"] = date_extracted 
                if time_extracted: 
                    if len(time_extracted) == 5: 
                        time_extracted = f"{time_extracted}:00" 
                    st.session_state.booking_info["time"] = time_extracted 
                st.rerun()


# ====== Form đặt lịch ======
if st.session_state.show_booking_form:
    with st.chat_message("assistant"):
        st.write("📅 Vui lòng điền thông tin đặt lịch:")

        prev_info = st.session_state.booking_info

        with st.form("booking_form"):
            name = st.text_input("Tên bệnh nhân (không bắt buộc)", value=prev_info.get("name", ""))

            # Giới hạn ngày hợp lệ
            min_date = datetime.date.today() + datetime.timedelta(days=1)
            max_date = datetime.date.today() + datetime.timedelta(days=14)

            # Ngày mặc định
            if "date" in prev_info and prev_info["date"]:
                try:
                    default_date = datetime.datetime.strptime(prev_info["date"], "%Y-%m-%d").date()
                    if not (min_date <= default_date <= max_date):
                        add_message(
                            "assistant",
                            f"❌ Ngày bạn nhập ({default_date.strftime('%d/%m/%Y')}) không hợp lệ.\n"
                            f"👉 Lịch hợp lệ là từ {min_date.strftime('%d/%m/%Y')} đến {max_date.strftime('%d/%m/%Y')}."
                        )
                        default_date = min_date
                except Exception:
                    default_date = min_date
            else:
                default_date = min_date

            date = st.date_input(
                "Ngày khám",
                value=default_date,
                min_value=min_date,
                max_value=max_date
            )

            # Giờ mặc định
            if "time" in prev_info and prev_info["time"]:
                try:
                    default_time = datetime.datetime.strptime(prev_info["time"], "%H:%M:%S").time()
                except Exception:
                    try:
                        default_time = datetime.datetime.strptime(prev_info["time"], "%H:%M").time()
                    except Exception:
                        default_time = datetime.time(9, 0)
            else:
                default_time = datetime.time(9, 0)

            # Danh sách slot 30 phút từ 06:00 -> 18:00
            time_slots = generate_time_slots("06:00", "18:00", 30)

            time = st.selectbox(
                "Giờ khám",
                options=time_slots,
                index=time_slots.index(default_time) if default_time in time_slots else 6,
                format_func=lambda t: t.strftime("%H:%M")
            )

            submit = st.form_submit_button("Xem bác sĩ rảnh")

        if submit:
            date_str = date.strftime("%Y-%m-%d")
            time_str = time.strftime("%H:%M:%S")
            available = find_available_doctors(date_str, time_str)

            if not available:
                add_message("assistant", f"❌ Rất tiếc, không có bác sĩ nào rảnh vào {time.strftime('%H:%M')} ngày {date_str}.")
                add_message("assistant", "👉 Bạn vui lòng chọn lại thời gian khác:")
                # Giữ form mở để user chọn lại
                st.session_state.show_booking_form = True
                toggle_chat_input(True)
                st.session_state.booking_info = {"date": date_str}
                st.rerun()
            else:
                reply = "✅ Các bác sĩ có thể khám:\n"
                for doc in available:
                    reply += f"- {doc['name']} (Phòng {doc['room']})\n"
                reply += "\n👉 Bạn muốn đặt với bác sĩ nào?"

                st.session_state.pending_booking = {
                    "date": date_str,
                    "time": time_str,
                    "available": available
                }
                st.session_state.booking_info["name"] = name
                add_message("assistant", reply)

            # Tắt form, bật lại chat
            st.session_state.show_booking_form = False
            toggle_chat_input(False)
            st.rerun()


# ====== Hỏi email bằng Yes/No ======
if st.session_state.ask_email:
    with st.chat_message("assistant"):
        st.write("👉 Bạn có muốn để lại email để nhận thông báo không?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Có"):
                st.session_state.show_email_form = True
                st.session_state.ask_email = False
                st.rerun()
        with col2:
            if st.button("❌ Không"):
                add_message("assistant", "👍 Cảm ơn bạn! Lịch khám đã được ghi nhận.")
                st.session_state.ask_email = False
                toggle_chat_input(False)  # bật chat lại
                st.rerun()


# ====== Form nhập email ======
if st.session_state.show_email_form:
    with st.chat_message("assistant"):
        with st.form("email_form"):
            email = st.text_input("Nhập email của bạn")
            submit_email = st.form_submit_button("Gửi")

        if submit_email:
            if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                add_message("assistant", f"📧 Email {email} đã được ghi nhận. Cảm ơn bạn!")
                st.session_state.show_email_form = False
                toggle_chat_input(False)  # bật chat lại
            else:
                add_message("assistant", "❌ Email không hợp lệ, vui lòng nhập lại.")
            st.rerun()
