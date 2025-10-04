import os, datetime, re, json, logging, random, string
import gradio as gr
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

# Helpers từ project
from helpers import (
    generate_time_slots,
    send_email,
    write_confirm_email_v2,
    write_cancel_email_v2,
    choose_doctor
)
from db import find_available_doctors, get_all_doctors, get_available_slots, format_slots_human_readable
from email_settings import from_email_default, password_default, sdt

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# ====== Setup Gemini ======
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY_1"))
model = GenerativeModel("gemini-2.5-flash-lite")

# ====== Helper ======
def classify_intent(user_text: str) -> str:
    text = (user_text or "").lower().strip()
    booking_kw = ["đặt lịch", "muốn đặt", "book", "đăng ký", "muốn khám", "hẹn khám", "đặt khám"]
    doctor_kw = ["danh sách bác sĩ", "những bác sĩ", "có bác sĩ nào", "ai là bác sĩ",
                 "ngày khám", "lịch khám", "xem lịch", "còn trống", "rảnh", "lịch của", "lịch trống",
                 "bác sĩ số", "phòng khám"]
    chitchat_kw = ["xin chào", "hi", "hello", "cảm ơn", "thanks", "ok", "được", "chào"]

    if any(k in text for k in chitchat_kw):
        return "chitchat"
    if any(k in text for k in booking_kw):
        return "booking_request"
    if any(k in text for k in doctor_kw):
        return "doctor_info"

    try:
        prompt = f"""
        Bạn là hệ thống phân loại intent.
        Văn bản: "{user_text}"
        Nếu người dùng muốn đặt lịch khám => booking_request
        Nếu muốn hỏi bác sĩ, lịch khám, lịch trống => doctor_info
        Nếu không => chitchat
        """
        resp = model.generate_content(prompt)
        return resp.text.strip().lower()
    except:
        return "chitchat"


def extract_datetime(user_text):
    today = datetime.date.today()
    prompt = f"""
    Trích xuất ngày và giờ từ câu sau.
    Trả về JSON dạng: {{"date": "YYYY-MM-DD" hoặc null, "time": "HH:MM" hoặc null}}.
    Hôm nay là {today.strftime('%Y-%m-%d')}.
    Câu: "{user_text}"
    """
    try:
        resp = model.generate_content(prompt)
        m = re.search(r"\{.*\}", resp.text, re.S)
        if m:
            data = json.loads(m.group())
            return data.get("date"), data.get("time")
    except:
        pass
    return None, None


def parse_date_input(date_val):
    if isinstance(date_val, datetime.date):
        return date_val
    if date_val is None:
        return None
    s = str(date_val).strip()
    if "T" in s:
        s = s.split("T")[0]
    try:
        return datetime.date.fromisoformat(s)
    except:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except:
                continue
    return None


def _format_history(history):
    return [{"role": m["role"], "content": m["content"]} for m in history]

# ====== Handlers ======
def on_user_message(message, history, booking_info, booking_done, cancel_mode, last_doctors):
    history = history or []
    booking_info = booking_info or {}
    last_doctors = last_doctors or []

    history.append({"role": "user", "content": message})
    intent = classify_intent(message)

    # --- Chitchat ---
    if intent == "chitchat":
        try:
            reply = model.generate_content(message).text
        except:
            reply = "Mình nghe bạn. Bạn muốn làm gì tiếp theo?"
        history.append({"role": "assistant", "content": reply})
        return _format_history(history), history, booking_info, booking_done, cancel_mode, last_doctors, \
            gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False)

    # --- Doctor info ---
    if intent == "doctor_info":
        doctors = get_all_doctors()
        user_lower = message.lower()

        if any(k in user_lower for k in ["danh sách", "những bác sĩ", "có bác sĩ nào", "liệt kê bác sĩ"]):
            reply = "👨‍⚕️ Danh sách bác sĩ:\n"
            for i, doc in enumerate(doctors, 1):
                reply += f"{i}. {doc['name']} (Phòng {doc.get('room','-')})\n"
            history.append({"role": "assistant", "content": reply})
            last_doctors = doctors
            return _format_history(history), history, booking_info, booking_done, False, last_doctors, \
                gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False)

        m_idx = re.search(r"bác sĩ\s*(?:số\s*)?(\d+)", user_lower)
        chosen = None
        if m_idx and last_doctors:
            idx = int(m_idx.group(1)) - 1
            if 0 <= idx < len(last_doctors):
                chosen = last_doctors[idx]

        if not chosen:
            chosen = choose_doctor(message, doctors)
            if isinstance(chosen, list) and len(chosen) == 1:
                chosen = chosen[0]

        if not chosen:
            history.append({"role": "assistant", "content": "❌ Không tìm thấy bác sĩ phù hợp."})
            return _format_history(history), history, booking_info, booking_done, False, doctors, \
                gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False)

        date_extracted, _ = extract_datetime(message)
        target_date = date_extracted or datetime.date.today().strftime("%Y-%m-%d")
        slots = get_available_slots(chosen["id"], target_date) or []
        if not slots:
            reply = f"❌ Không có lịch trống cho bác sĩ {chosen['name']} ngày {target_date}."
        else:
            reply = f"📅 Lịch trống của bác sĩ {chosen['name']} ngày {target_date}:\n"
            reply += format_slots_human_readable(slots)
        history.append({"role": "assistant", "content": reply})
        last_doctors = [chosen]
        return _format_history(history), history, booking_info, booking_done, False, last_doctors, \
            gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False)

    # --- Booking ---
    if intent == "booking_request":
        d, t = extract_datetime(message)
        if d: booking_info["Ngay"] = d
        if t: booking_info["Gio"] = t if len(t) > 5 else f"{t}:00"
        history.append({"role": "assistant", "content": "📅 Mở form đặt lịch, vui lòng nhập thông tin."})
        return _format_history(history), history, booking_info, False, False, last_doctors, \
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # fallback
    reply = model.generate_content(message).text
    history.append({"role": "assistant", "content": reply})
    return _format_history(history), history, booking_info, booking_done, cancel_mode, last_doctors, \
        gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False)

# ====== Submit booking ======
def submit_booking(name, date_val, time_val, doctor_choice, email, history, booking_info):
    history = history or []
    booking_info = booking_info or {}

    # Validate name
    if not name or not str(name).strip():
        history.append({"role": "assistant", "content": "❌ Họ tên là bắt buộc. Vui lòng nhập lại."})
        return _format_history(history), history, booking_info, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # Parse date (compat with DatePicker or Textbox)
    date_obj = parse_date_input(date_val)
    if date_obj is None:
        history.append({"role": "assistant", "content": "❌ Ngày không hợp lệ. Vui lòng nhập ngày theo định dạng YYYY-MM-DD."})
        return _format_history(history), history, booking_info, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # Allowed range: tomorrow .. +14 days
    min_date = datetime.date.today() + datetime.timedelta(days=1)
    max_date = datetime.date.today() + datetime.timedelta(days=14)
    if date_obj < min_date or date_obj > max_date:
        history.append({"role": "assistant", "content": f"❌ Ngày phải trong khoảng {min_date.strftime('%d/%m/%Y')} — {max_date.strftime('%d/%m/%Y')}."})
        return _format_history(history), history, booking_info, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    date_str = date_obj.isoformat()

    # Normalize time
    if isinstance(time_val, str):
        t_str = time_val
    else:
        t_str = str(time_val)
    # accept "HH:MM" or "HH:MM:SS"
    if len(t_str) == 5:
        t_full = t_str + ":00"
    else:
        t_full = t_str

    # chosen doctor resolution
    chosen_doctor = None
    if doctor_choice and doctor_choice != "(Không chọn)":
        for d in get_all_doctors():
            label = f"{d['name']} (Phòng {d.get('room','-')})"
            if label == doctor_choice:
                chosen_doctor = d
                break

    # Check availability
    try:
        available = find_available_doctors(date_str, t_full)
    except Exception as e:
        logging.debug(f"find_available_doctors error: {e}")
        available = []

    if chosen_doctor:
        if chosen_doctor["id"] not in [d["id"] for d in available]:
            history.append({"role": "assistant", "content": f"❌ Bác sĩ {chosen_doctor['name']} không rảnh vào {t_full[:5]} {date_obj.strftime('%d/%m/%Y')}. Vui lòng chọn lại."})
            booking_info = {"HoTen": name, "Ngay": date_str, "Gio": t_full, "email": email}
            return _format_history(history), history, booking_info, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    if not available:
        history.append({"role": "assistant", "content": f"❌ Không có bác sĩ nào rảnh {t_full[:5]} {date_obj.strftime('%d/%m/%Y')}. Vui lòng chọn lại."})
        booking_info = {"HoTen": name, "Ngay": date_str, "email": email}
        return _format_history(history), history, booking_info, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

    # Create booking
    booking_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    booking_info = {
        "MaDatLich": booking_code,
        "HoTen": name,
        "Ngay": date_str,
        "Gio": t_full[:5],
        "doctor_id": chosen_doctor["id"] if chosen_doctor else None,
        "doctor_name": chosen_doctor["name"] if chosen_doctor else "(Chưa chọn)",
        "doctor_room": chosen_doctor["room"] if chosen_doctor else "(Chưa chọn)",
        "email": email,
        "ChiNhanh": "Cơ sở 1",
        "DiaChi": "123 Đường ABC, Quận 1, TP.HCM"
    }

    # Summary & email
    try:
        d_obj = date_obj
        weekday_map = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}
        weekday = weekday_map[d_obj.weekday()]
        date_human = d_obj.strftime("%d/%m/%Y")
    except Exception:
        weekday = ""
        date_human = booking_info["Ngay"]

    summary = (
        "🔔 Xác nhận thông tin đặt lịch:\n"
        f"- Mã đặt lịch: {booking_info['MaDatLich']}\n"
        f"- Họ tên: {booking_info['HoTen']}\n"
        f"- Ngày: {date_human} {f'({weekday})' if weekday else ''}\n"
        f"- Giờ: {booking_info['Gio']}\n"
        f"- Bác sĩ: {booking_info['doctor_name']} (Phòng {booking_info['doctor_room']})\n"
    )
    history.append({"role": "assistant", "content": summary})

    if email:
        try:
            subject, body = write_confirm_email_v2({**booking_info, "Hotline": sdt})
            send_email(email, from_email_default, password_default, subject, body)
            history.append({"role": "assistant", "content": "✅ Email xác nhận đã được gửi tới " + email})
        except Exception as e:
            logging.debug(f"send_email error: {e}")
            history.append({"role": "assistant", "content": "⚠️ Gửi email thất bại nhưng lịch đã được lưu cục bộ."})
    else:
        history.append({"role": "assistant", "content": "✅ Lịch khám đã được ghi nhận. (Không có email)"})

    return _format_history(history), history, booking_info, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)


# ====== Change / Cancel handlers ======
def on_click_change(history, booking_info):
    history = history or []
    booking_info = booking_info or {}
    history.append({"role": "assistant", "content": "🔁 Bạn đã chọn đổi lịch — mời chỉnh thông tin trong form."})
    return _format_history(history), history, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)


def on_click_cancel_mode(history):
    history = history or []
    history.append({"role": "assistant", "content": "❌ Vui lòng nhập lý do huỷ lịch và nhấn Xác nhận huỷ."})
    return _format_history(history), history, gr.update(visible=True)


def confirm_cancel(cancel_reason, history, booking_info):
    history = history or []
    booking_info = booking_info or {}
    if booking_info and booking_info.get("email"):
        from datetime import datetime
        data = {
            **booking_info,
            "LyDoHuy": cancel_reason if cancel_reason and cancel_reason.strip() else "Người dùng yêu cầu huỷ lịch",
            "NgayHuy": datetime.now().strftime("%Y-%m-%d"),
            "Hotline": sdt
        }
        try:
            subject, body = write_cancel_email_v2(data)
            send_email(booking_info["email"], from_email_default, password_default, subject, body)
            history.append({"role": "assistant", "content": "✅ Email huỷ lịch đã được gửi."})
        except Exception as e:
            logging.debug(f"send cancel email error: {e}")
            history.append({"role": "assistant", "content": "⚠️ Gửi email huỷ thất bại nhưng lịch đã được huỷ cục bộ."})
    history.append({"role": "assistant", "content": "🗑️ Lịch khám đã được huỷ."})
    return _format_history(history), history, {}, gr.update(visible=False)


# (submit_booking, on_click_change, on_click_cancel_mode, confirm_cancel)
# ... giữ nguyên như mình gửi ở bản trước (đã có validate ngày, mail confirm, cancel).
# ====== Build UI ======
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot Đặt lịch khám")

    history = gr.State([])
    booking_info = gr.State({})
    booking_done = gr.State(False)
    cancel_mode = gr.State(False)
    last_doctors = gr.State([])

    chatbot = gr.Chatbot(type="messages", label="")

    with gr.Group(visible=False) as form_col:
        name_input = gr.Textbox(label="Họ tên (*)")
        try:
            date_input = gr.DatePicker(label="Ngày khám (*)")
        except:
            date_input = gr.Textbox(label="Ngày khám (*)", placeholder="YYYY-MM-DD")
        time_slots = generate_time_slots("06:00", "18:00", 30)
        time_input = gr.Dropdown(choices=[t.strftime("%H:%M") for t in time_slots], label="Giờ khám (*)", value="09:00")
        doctor_dropdown = gr.Dropdown(choices=["(Không chọn)"] + [f"{d['name']} (Phòng {d.get('room','-')})" for d in get_all_doctors()], label="Bác sĩ (optional)")
        email_input = gr.Textbox(label="Email (optional)")
        submit_btn = gr.Button("Xác nhận đặt lịch")

    with gr.Row(visible=False) as post_booking_row:
        change_btn = gr.Button("🔁 Đổi lịch")
        cancel_btn = gr.Button("❌ Huỷ lịch")

    with gr.Group(visible=False) as cancel_col:
        cancel_reason_input = gr.Textbox(label="Lý do huỷ (optional)")
        confirm_cancel_btn = gr.Button("Xác nhận huỷ")

    # ✅ Textbox nhập chat để dưới cùng
    user_input = gr.Textbox(placeholder="Nhập tin nhắn...", lines=1)

    # Bindings (giữ nguyên từ code trước) ...
    # Bindings
    user_input.submit(
        fn=on_user_message,
        inputs=[user_input, history, booking_info, booking_done, cancel_mode, last_doctors],
        outputs=[chatbot, history, booking_info, booking_done, cancel_mode, last_doctors, form_col, post_booking_row, cancel_col],
        queue=False
    )

    submit_btn.click(
        fn=submit_booking,
        inputs=[name_input, date_input, time_input, doctor_dropdown, email_input, history, booking_info],
        outputs=[chatbot, history, booking_info, form_col, post_booking_row, cancel_col],
        queue=False
    )

    change_btn.click(
        fn=on_click_change,
        inputs=[history, booking_info],
        outputs=[chatbot, history, form_col, post_booking_row, cancel_col],
        queue=False
    )

    cancel_btn.click(
        fn=on_click_cancel_mode,
        inputs=[history],
        outputs=[chatbot, history, cancel_col],
        queue=False
    )

    confirm_cancel_btn.click(
        fn=confirm_cancel,
        inputs=[cancel_reason_input, history, booking_info],
        outputs=[chatbot, history, booking_info, cancel_col],
        queue=False
    )

demo.launch()
