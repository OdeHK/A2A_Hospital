import streamlit as st
import os, datetime, re, json, logging, random, string
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
from helpers import (
    is_valid_booking_date as validate_date,
    generate_time_slots,
    send_email,
    write_confirm_email,
    write_cancel_email
)
from db import find_available_doctors
from email_settings import from_email_default, password_default, sdt

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
    "pending_booking": None,
    "disable_chat_input": False,
    "user_email": None,
    "booking_done": False
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


# ====== Phân loại intent ======
def classify_intent(user_text):
    prompt = f"""
    Bạn là hệ thống phân loại intent.
    Văn bản: "{user_text}"
    Nếu người dùng đang muốn đặt lịch khám hoặc đề cập tới việc hẹn khám => trả về: booking_request
    Nếu không => trả về: chitchat
    """
    resp = model.generate_content(prompt)
    return resp.text.strip().lower()


# ====== Trích xuất ngày giờ ======
def extract_datetime(user_text):
    today = datetime.date.today()
    prompt = f"""
    Phân tích câu sau và tìm ngày + giờ (nếu có).
    Trả về JSON: {{"date": "YYYY-MM-DD" hoặc null, "time": "HH:MM" hoặc null}}.
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


# ====== Input chat ======
if not st.session_state.disable_chat_input and not st.session_state.booking_done:
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

        for doc in st.session_state.pending_booking["available"]:
            if doc["name"].lower() in user_lower or user_lower in doc["name"].lower():
                matches.append(doc)

        if "1" in user_lower or "đầu tiên" in user_lower:
            chosen = st.session_state.pending_booking["available"][0]
        elif "2" in user_lower:
            if len(st.session_state.pending_booking["available"]) >= 2:
                chosen = st.session_state.pending_booking["available"][1]

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

            date_obj = datetime.datetime.strptime(booking_info["date"], "%Y-%m-%d").date()
            weekday_map = {0:"Thứ Hai",1:"Thứ Ba",2:"Thứ Tư",3:"Thứ Năm",4:"Thứ Sáu",5:"Thứ Bảy",6:"Chủ Nhật"}
            weekday = weekday_map[date_obj.weekday()]
            date_str = date_obj.strftime("%d/%m/%Y")
            time_str = booking_info["time"][:5]

            reply = f"✅ Bạn đã đặt lịch với bác sĩ {chosen['name']} vào {time_str} {weekday}, {date_str}."
            add_message("assistant", reply)

            st.session_state.pending_booking = None
            st.session_state.ask_email = True
            toggle_chat_input(True)
            st.rerun()
        else:
            add_message("assistant", "❌ Mình chưa hiểu bạn muốn chọn bác sĩ nào.")
    else:
        intent = classify_intent(user_input)
        if intent == "chitchat":
            reply = model.generate_content(user_input).text
            add_message("assistant", reply)
        else:
            date_extracted, time_extracted = extract_datetime(user_input)
            if date_extracted and time_extracted and validate_date(date_extracted):
                time_full = f"{time_extracted}:00" if len(time_extracted)==5 else time_extracted
                available = find_available_doctors(date_extracted, time_full)
                if available:
                    reply = "✅ Các bác sĩ có thể khám:\n"
                    for doc in available:
                        reply += f"- {doc['name']} (Phòng {doc['room']})\n"
                    reply += "\n👉 Bạn muốn đặt với bác sĩ nào?"
                    st.session_state.pending_booking = {"date": date_extracted,"time": time_full,"available": available}
                    add_message("assistant", reply)
                else:
                    add_message("assistant", "❌ Không có bác sĩ nào rảnh thời gian đó.")
            else:
                st.session_state.show_booking_form = True
                toggle_chat_input(True)
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



# ====== Hỏi email ======
if st.session_state.ask_email:
    with st.chat_message("assistant"):
        st.write("👉 Bạn có muốn để lại email để nhận thông báo không?")
        c1,c2 = st.columns(2)
        if c1.button("✅ Có"):
            st.session_state.show_email_form = True
            st.session_state.ask_email = False
            st.rerun()
        if c2.button("❌ Không"):
            add_message("assistant","👍 Lịch khám đã được ghi nhận.")
            st.session_state.booking_done = True
            st.session_state.ask_email = False
            toggle_chat_input(False)
            st.rerun()


# ====== Nhập email + gửi mail xác nhận ======
if st.session_state.show_email_form:
    with st.chat_message("assistant"):
        with st.form("email_form"):
            email = st.text_input("Nhập email của bạn")
            submit_email = st.form_submit_button("Gửi")
        if submit_email:
            if re.match(r"[^@]+@[^@]+\.[^@]+", email):
                st.session_state.user_email = email
                booking_code = ''.join(random.choices(string.ascii_uppercase+string.digits,k=8))
                data = {
                    "TenBenhVien": "Hehe Hospital",
                    "MaDatLich": booking_code,
                    "HoTen": st.session_state.booking_info.get("name","Quý khách"),
                    "Ngay": st.session_state.booking_info["date"],
                    "Gio": st.session_state.booking_info["time"][:5],
                    "ChiNhanh": "Cơ sở 1",
                    "DiaChi": "123 Đường ABC, Quận 1, TP.HCM",
                    "Hotline": sdt,
                    "EmailCSKH": "cskh@hehehospital.vn",
                    "ZaloChatLink": "https://zalo.me/hehehospital",
                    "LinkDoiHuy": "https://hehehospital.vn/lichkham",
                    "GioLamViec": "Thứ 2 - Thứ 7: 7h00 - 17h00",
                    "Website": "https://hehehospital.vn"
                }
                subject, body = write_confirm_email(data)
                send_email(email, from_email_default, password_default, subject, body)
                add_message("assistant","✅ Email xác nhận đã được gửi!")
                st.session_state.booking_done = True
                st.session_state.show_email_form = False
                toggle_chat_input(False)
                st.rerun()
            else:
                add_message("assistant","❌ Email không hợp lệ.")
                st.rerun()


# ====== Sau khi đã đặt xong: hiện nút Đổi / Huỷ ======
if st.session_state.booking_done:
    with st.chat_message("assistant"):
        st.write("📌 Bạn muốn làm gì tiếp theo?")
        c1, c2 = st.columns(2)

        # ---------- Nút Đổi lịch ----------
        if c1.button("🔁 Đổi lịch"):
            st.session_state.show_booking_form = True
            st.session_state.booking_done = False
            toggle_chat_input(True)
            st.rerun()

        # ---------- Nút Huỷ lịch ----------
        if c2.button("❌ Huỷ lịch"):
            if st.session_state.user_email and st.session_state.booking_info:
                from datetime import datetime
                # Chuẩn bị dữ liệu gửi mail hủy
                data = {
                    "TenBenhVien": "Hehe Hospital",
                    "MaDatLich": st.session_state.booking_info.get("MaDatLich"),
                    "HoTen": st.session_state.booking_info.get("HoTen"),
                    "Ngay": st.session_state.booking_info.get("Ngay"),
                    "Gio": st.session_state.booking_info.get("Gio"),
                    "ChiNhanh": st.session_state.booking_info.get("ChiNhanh"),
                    "DiaChi": st.session_state.booking_info.get("DiaChi"),
                    "LyDoHuy": "Người dùng yêu cầu huỷ lịch",
                    "NgayHuy": datetime.now().strftime("%Y-%m-%d"),
                    "Hotline": "0123456789",
                    "EmailCSKH": "cskh@hehehospital.vn",
                    "ZaloChatLink": "https://zalo.me/hehehospital",
                    "LinkDoiHuy": "https://hehehospital.vn/lichkham",
                    "GioLamViec": "7h30 - 17h00",
                    "Website": "https://hehehospital.vn"
                }

                # Tạo nội dung mail huỷ lịch
                subject, body = write_cancel_email(data)

                # Gửi email huỷ lịch
                send_email(
                    st.session_state.user_email,
                    from_email_default,
                    password_default,
                    subject,
                    body
                )

            # Thông báo và reset trạng thái
            add_message("assistant", "🗑️ Lịch khám đã được huỷ.")
            for k in ["booking_info", "user_email"]:
                st.session_state[k] = None
            st.session_state.booking_done = False
            toggle_chat_input(False)
            st.rerun()
