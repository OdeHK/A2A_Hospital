# app.py (updated)

import streamlit as st
import os, re, json, logging, random, string, sqlite3
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure
from helpers import (
    is_valid_booking_date as validate_date,
    generate_time_slots,
    send_email,
    write_confirm_email,
    write_cancel_email,
    choose_doctor
)
from db import (
    get_all_doctors,
    find_available_doctors,
    get_available_slots,
    format_slots_human_readable,
)
from email_settings import from_email_default, password_default, sdt

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# ====== Setup ======
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY_2"))
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
    "booking_done": False,
    "last_doctor_list": None   # lưu danh sách bác sĩ gần nhất bot vừa show
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ====== Helper display / UI ======
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role):
        st.write(content)

def toggle_chat_input(flag: bool):
    st.session_state.disable_chat_input = flag

# ====== Hybrid intent detection: rules first, fallback to model only if necessary ======
def classify_intent(user_text: str) -> str:
    text = user_text.lower().strip()

    # ====== Nhóm booking (rõ ràng về việc đặt lịch) ======
    booking_kw = [
        "đặt lịch", "muốn đặt", "book", "đăng ký",
        "muốn khám", "hẹn khám", "đặt khám"
    ]

    # ====== Nhóm doctor_info (xem lịch, danh sách, ngày khám) ======
    doctor_kw = [
        "danh sách bác sĩ", "những bác sĩ", "có bác sĩ nào", "ai là bác sĩ",
        "ngày khám", "lịch khám", "xem lịch",
        "còn trống", "rảnh", "lịch của", "lịch trống",
        "bác sĩ số", "phòng khám"
    ]

    # ====== Nhóm chitchat (giao tiếp xã giao) ======
    chitchat_kw = ["xin chào", "hi", "hello", "cảm ơn", "thanks", "ok", "được"]
    # --- Chitchat ---
    if any(k in text for k in chitchat_kw):
        return "chitchat"
    
    # --- Ưu tiên booking ---
    if any(k in text for k in booking_kw):
        return "booking_request"
    
    # --- Ưu tiên doctor_info ---
    if any(k in text for k in doctor_kw):
        return "doctor_info"



    # --- Fallback dùng model cho câu mơ hồ ---
    try:
        prompt = f"""
        Bạn là hệ thống phân loại intent.
        Văn bản: "{user_text}"
        Nếu người dùng đang muốn đặt lịch khám hoặc hẹn khám => trả về: booking_request
        Nếu người dùng muốn hỏi danh sách bác sĩ hoặc lịch khám/lịch trống của bác sĩ => trả về: doctor_info
        Nếu không => trả về: chitchat
        """
        resp = model.generate_content(prompt)
        return resp.text.strip().lower()
    except Exception:
        return "chitchat"


# ====== Trích xuất ngày giờ đơn giản (regex + relative Vietnamese) ======
def extract_datetime(user_text: str):
    """
    Trả về (date_str, time_str)
    - date_str: "YYYY-MM-DD" hoặc None
    - time_str: "HH:MM" hoặc None
    """

    text = user_text.lower().strip()

    # ===== Date extraction =====
    today = date.today()
    date_target = None

    if re.search(r"\bhôm\s+nay\b|\btoday\b", text):
        date_target = today
    elif re.search(r"\bngày\s+mai\b|\bmai\b", text):
        date_target = today + timedelta(days=1)
    elif re.search(r"\bngày\s+kia\b|\bkia\b", text):
        date_target = today + timedelta(days=2)
    elif re.search(r"\bmốt\b|\bngày\s+mốt\b", text):
        date_target = today + timedelta(days=2)
    else:
        # dạng số: 20/9[/2025] hoặc 20-09-25
        m = re.search(r"(\d{1,2})[\/\-](\d{1,2})(?:[\/\-](\d{2,4}))?", text)
        if m:
            d, mo, yraw = int(m.group(1)), int(m.group(2)), m.group(3)
            year = today.year
            if yraw:
                year = int(yraw) + 2000 if len(yraw) == 2 else int(yraw)
            try:
                date_target = date(year, mo, d)
            except ValueError:
                date_target = None

    # ===== Time extraction =====
    time_target = None
    # dạng 09:30 hoặc 9:30
    mtime = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if not mtime:
        # dạng 9h30, 9h, 9 giờ 30
        mtime = re.search(r"\b(\d{1,2})\s*(?:h|giờ)\s*(\d{1,2})?\b", text)

    if mtime:
        h = int(mtime.group(1))
        mmin = int(mtime.group(2)) if mtime.group(2) else 0
        if 0 <= h < 24 and 0 <= mmin < 60:
            time_target = f"{h:02d}:{mmin:02d}"

    # ===== Output =====
    date_str = date_target.strftime("%Y-%m-%d") if date_target else None
    time_str = time_target

    return date_str, time_str

# ====== Hiển thị lịch sử chat (giữ nguyên) ======
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ====== Input chat (giữ nguyên) ======
if not st.session_state.disable_chat_input and not st.session_state.booking_done:
    user_input = st.chat_input("Nhập tin nhắn...")
else:
    user_input = None

# ====== Xử lý input ======
if user_input:
    add_message("user", user_input)

    # Nếu đang chờ user chọn bác sĩ (pending booking)
    if st.session_state.pending_booking:
        # sử dụng choose_doctor dựa trên danh sách pending_booking["available"]
        available = st.session_state.pending_booking["available"]
        chosen = choose_doctor(user_input, available)

        if isinstance(chosen, list) and len(chosen) > 1:
            reply = "❌ Có nhiều bác sĩ trùng tên. Vui lòng chọn rõ hơn:\n"
            for i, doc in enumerate(chosen, 1):
                reply += f"{i}. {doc['name']} (Phòng {doc['room']})\n"
            add_message("assistant", reply)
            st.stop()

        if chosen:
            booking_info = {
                "MaDatLich": f"DL{int(datetime.now().timestamp())}",  # sinh mã đặt lịch tạm
                "HoTen": st.session_state.get("user_name", "Khách hàng"),  # hoặc tên lấy từ user
                "doctor_id": chosen.get("doctor_id") or chosen.get("id"),
                "doctor_name": chosen.get("name"),
                "Ngay": st.session_state.pending_booking["date"],
                "Gio": st.session_state.pending_booking["time"],
                "ChiNhanh": "Cơ sở chính",
                "DiaChi": "123 Nguyễn Văn A, Quận 1, TP.HCM"
            }
            st.session_state.booking_info = booking_info


            # dùng datetime trực tiếp vì đã import from datetime import datetime
            date_obj = datetime.strptime(booking_info["Ngay"], "%Y-%m-%d").date()
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
            date_str = date_obj.strftime("%d/%m/%Y")
            time_str = booking_info["Gio"][:5]

            reply = f"✅ Bạn đã đặt lịch với bác sĩ {chosen['name']} vào {time_str} {weekday}, {date_str}."
            add_message("assistant", reply)

            st.session_state.pending_booking = None
            st.session_state.ask_email = True
            toggle_chat_input(True)
            st.rerun()
        else:
            add_message("assistant", "❌ Mình chưa hiểu bạn muốn chọn bác sĩ nào.")

    else:
        # classify intent (rule-based with fallback)
        intent = classify_intent(user_input)
        print(f"[DEBUG] User input: {user_input} -> intent: {intent}")

        # ========== intent doctor_info ==========
        if intent == "doctor_info":
            doctors = get_all_doctors()  # list of dicts expected from db.py
            print(f"[DEBUG] Doctors fetched: {doctors}")

            chosen = choose_doctor(user_input, doctors)
            print(f"[DEBUG] Chosen doctor (before số X logic): {chosen}")

            # Hỏi danh sách bác sĩ (explicit)
            if any(k in user_input.lower() for k in ["danh sách", "những bác sĩ", "có bác sĩ nào", "liệt kê bác sĩ"]):
                reply = "👨‍⚕️ Danh sách bác sĩ:\n"
                for i, doc in enumerate(doctors, 1):
                    reply += f"{i}. {doc['name']} (Phòng {doc['room']})\n"
                add_message("assistant", reply)
                st.session_state.last_doctor_list = doctors
                print(f"[DEBUG] Saved last_doctor_list with {len(doctors)} doctors")

            # Hỏi lịch bác sĩ cụ thể
            else:
                user_lower = user_input.lower()
                m_idx = re.search(r"bác sĩ\s*(?:số\s*)?(\d+)", user_lower)
                if m_idx and st.session_state.last_doctor_list:
                    idx = int(m_idx.group(1)) - 1
                    if 0 <= idx < len(st.session_state.last_doctor_list):
                        chosen = st.session_state.last_doctor_list[idx]
                    print(f"[DEBUG] User referenced bác sĩ số {m_idx.group(1)} -> chosen: {chosen}")

                # determine date (if user provided date)
                date_extracted, time_extracted = extract_datetime(user_input)
                print(f"[DEBUG] Extracted datetime -> date: {date_extracted}, time: {time_extracted}")

                if date_extracted:
                    target_date = date_extracted
                else:
                    target_date = datetime.today().strftime("%Y-%m-%d")
                print(f"[DEBUG] Final target_date: {target_date}")

                if isinstance(chosen, list) and len(chosen) > 1:
                    reply = "❌ Có nhiều bác sĩ trùng tên, vui lòng chọn rõ hơn:\n"
                    for i, doc in enumerate(chosen, 1):
                        reply += f"{i}. {doc['name']} (Phòng {doc['room']})\n"
                    add_message("assistant", reply)
                elif chosen:
                    chosen_id = chosen.get("id") or chosen.get("doctor_id")
                    print(f"[DEBUG] Chosen doctor id: {chosen_id}")
                    slots = get_available_slots(chosen_id, target_date)
                    print(f"[DEBUG] Available slots for {chosen['name']} on {target_date}: {slots}")
                    reply = f"📅 Lịch trống của bác sĩ {chosen['name']} ngày {target_date}:\n"
                    reply += format_slots_human_readable(slots)
                    add_message("assistant", reply)
                else:
                    print("[DEBUG] No doctor chosen!")
                    add_message("assistant", "❌ Mình không tìm thấy bác sĩ nào khớp với yêu cầu.")

        # ========== intent chitchat ==========
        elif intent == "chitchat":
            print(f"[DEBUG] Intent chitchat for input: {user_input}")
            try:
                reply = model.generate_content(user_input).text
            except Exception as e:
                print(f"[DEBUG] Model chitchat error: {e}")
                reply = "Mình nghe bạn. Bạn muốn làm gì tiếp theo?"
            add_message("assistant", reply)

        # ========== intent booking_request ==========
        else:
            # booking_request: extract trước, rồi quyết định
            print(f"[DEBUG] Intent booking_request for input: {user_input}")
            date_extracted, time_extracted = extract_datetime(user_input)
            print(f"[DEBUG] Booking datetime extracted -> date: {date_extracted}, time: {time_extracted}")

            # Trường hợp đủ ngày + giờ -> thử tìm bác sĩ rảnh
            if date_extracted and time_extracted:
                if not validate_date(date_extracted):
                    reply = "❌ Ngày không hợp lệ!"
                    add_message("assistant", reply)
                else:
                    # chuẩn hoá giờ về HH:MM:SS
                    time_full = f"{time_extracted}:00" if len(time_extracted) == 5 else time_extracted
                    available = find_available_doctors(date_extracted, time_full)
                    print(f"[DEBUG] Available doctors: {available}")

                    if available:
                        reply = "✅ Các bác sĩ có thể khám:\n"
                        for doc in available:
                            reply += f"- {doc['name']} (Phòng {doc['room']})\n"
                        reply += "\n👉 Bạn muốn đặt với bác sĩ nào?"
                        st.session_state.pending_booking = {
                            "date": date_extracted,
                            "time": time_full,
                            "available": available
                        }
                        print(f"[DEBUG] Saved pending_booking: {st.session_state.pending_booking}")
                        add_message("assistant", reply)
                    else:
                        # Không có bác sĩ rảnh tại thời điểm đó -> mở form và lưu date/time để user chọn lại
                        add_message("assistant", f"❌ Rất tiếc, không có bác sĩ nào rảnh vào {time_extracted} ngày {date_extracted}.")
                        add_message("assistant", "👉 Bạn vui lòng chọn lại thời gian khác:")
                        st.session_state.show_booking_form = True
                        toggle_chat_input(True)

                        # lưu date + time để form hiển thị sẵn
                        booking_info = st.session_state.get("booking_info", {})
                        booking_info["Ngay"] = date_extracted
                        booking_info["Gio"] = time_full
                        st.session_state.booking_info = booking_info
                        print(f"[DEBUG] booking_info saved for form (no available doctors): {booking_info}")
                        st.rerun()

            else:
                # Thiếu ngày hoặc thiếu giờ -> mở form và lưu bất cứ gì đã extract được
                print("[DEBUG] Booking request missing date/time -> show booking form")
                st.session_state.show_booking_form = True
                toggle_chat_input(True)

                booking_info = st.session_state.get("booking_info", {})
                if date_extracted:
                    booking_info["Ngay"] = date_extracted
                if time_extracted:
                    # chuẩn hoá giờ về HH:MM:SS nếu cần
                    if len(time_extracted) == 5:
                        time_extracted = f"{time_extracted}:00"
                    booking_info["Gio"] = time_extracted
                st.session_state.booking_info = booking_info
                print(f"[DEBUG] booking_info saved for form: {booking_info}")

                st.rerun()




# ====== Form đặt lịch (giữ nguyên toàn bộ phần bạn có) ======
if st.session_state.show_booking_form:
    with st.chat_message("assistant"):
        st.write("📅 Vui lòng điền thông tin đặt lịch:")

        prev_info = st.session_state.booking_info

        with st.form("booking_form"):
            name = st.text_input("Tên bệnh nhân (không bắt buộc)", value=prev_info.get("name", ""))

            # Giới hạn ngày hợp lệ
            min_date = date.today() + timedelta(days=1)
            max_date = date.today() + timedelta(days=14)

            # Ngày mặc định
            if "date" in prev_info and prev_info["date"]:
                try:
                    default_date = datetime.strptime(prev_info["date"], "%Y-%m-%d").date()
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

            date_input_val = st.date_input(
                "Ngày khám",
                value=default_date,
                min_value=min_date,
                max_value=max_date
            )

            # Giờ mặc định
            if "time" in prev_info and prev_info["time"]:
                try:
                    default_time = datetime.strptime(prev_info["time"], "%H:%M:%S").time()
                except Exception:
                    try:
                        default_time = datetime.strptime(prev_info["time"], "%H:%M").time()
                    except Exception:
                        default_time = datetime.strptime("09:00", "%H:%M").time()
            else:
                default_time = datetime.strptime("09:00", "%H:%M").time()

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
            # date_input_val là datetime.date
            date_str = date_input_val.strftime("%Y-%m-%d")

            # time là datetime.time (từ selectbox)
            time_str = time.strftime("%H:%M:%S")

            available = find_available_doctors(date_str, time_str)

            if not available:
                add_message("assistant",
                    f"❌ Rất tiếc, không có bác sĩ nào rảnh vào {time.strftime('%H:%M')} ngày {date_str}.")
                add_message("assistant", "👉 Bạn vui lòng chọn lại thời gian khác:")

                st.session_state.show_booking_form = True
                toggle_chat_input(True)
                st.session_state.booking_info = {"Ngay": date_str, "Gio": time_str}
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
                    "Ngay": st.session_state.booking_info["Ngay"],
                    "Gio": st.session_state.booking_info["Gio"][:5],
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
if st.session_state.booking_done and not st.session_state.get("show_cancel_form", False):
    with st.chat_message("assistant"):
        st.write("👉 Nếu cần, bạn có thể chọn một trong các thao tác sau:")
        c1, c2 = st.columns(2)

        # ---------- Nút Đổi lịch ----------
        if c1.button("🔁 Đổi lịch"):
            st.session_state.show_booking_form = True
            st.session_state.booking_done = False
            toggle_chat_input(True)
            st.rerun()

        # ---------- Nút Huỷ lịch ----------
        if c2.button("❌ Huỷ lịch"):
            st.session_state.show_cancel_form = True
            st.rerun()

# ====== Form nhập lý do huỷ ======
if st.session_state.get("show_cancel_form", False):
    with st.form("cancel_form"):
        ly_do = st.text_area("Nhập lý do huỷ (không bắt buộc):", "")
        submit_cancel = st.form_submit_button("Xác nhận huỷ")

        if submit_cancel and st.session_state.user_email and st.session_state.booking_info:
            from datetime import datetime
            data = {
                "TenBenhVien": "Hehe Hospital",
                "MaDatLich": st.session_state.booking_info.get("MaDatLich"),
                "HoTen": st.session_state.booking_info.get("HoTen"),
                "Ngay": st.session_state.booking_info.get("Ngay"),
                "Gio": st.session_state.booking_info.get("Gio"),
                "ChiNhanh": st.session_state.booking_info.get("ChiNhanh"),
                "DiaChi": st.session_state.booking_info.get("DiaChi"),
                "LyDoHuy": ly_do if ly_do else "Người dùng yêu cầu huỷ lịch",
                "NgayHuy": datetime.now().strftime("%Y-%m-%d"),
                "Hotline": "0123456789",
                "EmailCSKH": "cskh@hehehospital.vn",
                "ZaloChatLink": "https://zalo.me/hehehospital",
                "LinkDoiHuy": "https://hehehospital.vn/lichkham",
                "GioLamViec": "7h30 - 17h00",
                "Website": "https://hehehospital.vn"
            }

            subject, body = write_cancel_email(data)
            send_email(
                st.session_state.user_email,
                from_email_default,
                password_default,
                subject,
                body
            )

            add_message("assistant", "🗑️ Lịch khám đã được huỷ.")
            for k in ["booking_info", "user_email", "show_cancel_form"]:
                st.session_state[k] = None
            st.session_state.booking_done = False
            toggle_chat_input(False)
            st.rerun()
