from datetime import datetime, timedelta, date
import os, re, json, logging, random, string
import gradio as gr
from dotenv import load_dotenv
from google.generativeai import GenerativeModel, configure

# Helpers từ project
from helpers import (
    generate_time_slots,
    send_email,
    write_confirm_email_v2,
    write_cancel_email_v2,
    choose_doctor,
    is_valid_booking_date,
    map_time_to_slot
)
from db import find_available_doctors, get_all_doctors, get_available_slots, format_slots_human_readable
from email_settings import from_email_default, password_default, sdt

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# ====== Setup Gemini ======
load_dotenv()
configure(api_key=os.getenv("GEMINI_API_KEY_1"))
model = GenerativeModel("gemini-2.5-flash-lite")

from pyngrok import ngrok
NGROK = os.getenv("NGROK")
ngrok.set_auth_token(NGROK)

# ====== Helper functions ======
def _format_history(history):
    """
    Gradio Chatbot with type='messages' accepts list of dicts: {'role': 'user'/'assistant', 'content': '...'}
    We'll return that format.
    """
    return [{"role": m["role"], "content": m["content"]} for m in history]

def _add_assistant(history, text: str):
    """
    Append assistant message as a single bubble (no splitting by newline).
    """
    history.append({"role": "assistant", "content": str(text)})
    return history


def parse_date_input(date_val):
    """Parse date input từ Gradio DatePicker (timestamp float, date obj, ISO string, hoặc text)."""
    if date_val is None:
        return None

    # Nếu là timestamp float → chuyển thành date
    if isinstance(date_val, (float, int)):
        return datetime.fromtimestamp(date_val).date()

    # Nếu đã là date rồi
    if isinstance(date_val, date):
        return date_val

    # Nếu là datetime
    if isinstance(date_val, datetime):
        return date_val.date()

    # Nếu là string
    s = str(date_val).strip()
    if "T" in s:
        s = s.split("T")[0]
    if " " in s:
        s = s.split(" ")[0]

    try:
        return date.fromisoformat(s)
    except:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except:
                continue

    return None

def classify_intent(user_text: str) -> str:
    text = (user_text or "").lower().strip()
    booking_kw = ["đặt lịch", "muốn đặt", "book", "đăng ký", "muốn khám", "hẹn khám", "đặt khám"]
    doctor_kw = ["danh sách bác sĩ", "những bác sĩ", "có bác sĩ nào", "ai là bác sĩ",
                 "ngày khám", "lịch khám", "xem lịch", "còn trống", "rảnh", "lịch của", "lịch trống",
                 "bác sĩ số", "phòng khám"]
    chitchat_kw = ["xin chào", "hi", "hello", "cảm ơn", "thanks", "ok", "được", "chào"]

    if any(k in text for k in chitchat_kw): return "chitchat"
    if any(k in text for k in booking_kw): return "booking_request"
    if any(k in text for k in doctor_kw): return "doctor_info"

    # fallback: use model
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
    except Exception as e:
        logging.debug(f"classify_intent fallback error: {e}")
        return "chitchat"

def extract_datetime(user_text):
    """Dùng Gemini để lấy ngày + giờ từ câu nhập (fallback ok)."""
    today = date.today()
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
    except Exception as e:
        logging.debug(f"extract_datetime error: {e}")
    return None, None

# ====== Handlers (all returns include clearing user_input via gr.update(value="")) ======
def on_user_message(message, history, booking_info, booking_done, cancel_mode, last_doctors):
    history = history or []
    booking_info = booking_info or {}
    last_doctors = last_doctors or []
    booking_done = booking_done or False
    cancel_mode = cancel_mode or False

    # Append user bubble
    history.append({"role": "user", "content": message})

    intent = classify_intent(message)
    logging.debug(f"User input: {message} -> intent: {intent}")

    # CHITCHAT
    if intent == "chitchat":
        try:
            reply = model.generate_content(message).text
        except Exception as e:
            logging.debug(f"chitchat model error: {e}")
            reply = "Mình nghe bạn. Bạn muốn làm gì tiếp theo?"
        history = _add_assistant(history, reply)
        return (
            _format_history(history), history, booking_info, booking_done, cancel_mode, last_doctors,
            gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
            gr.update(value=""),     # clear user_input
            gr.update(value=""),     # name_input
            gr.update(value=None),   # date_input
            gr.update(value=None),   # time_input
            gr.update(value="(Không chọn)"),  # doctor_dropdown
            gr.update(value="")      # email_input
        )

    # === DOCTOR INFO ===
    if intent == "doctor_info":
        doctors = get_all_doctors()
        user_lower = (message or "").lower()

        # --- Trường hợp: user hỏi danh sách bác sĩ ---
        if any(k in user_lower for k in ["danh sách", "những bác sĩ", "có bác sĩ nào", "liệt kê bác sĩ"]):
            reply = "👨‍⚕️ Danh sách bác sĩ:\n"
            for i, doc in enumerate(doctors, 1):
                reply += f"{i}. {doc['name']} (Phòng {doc.get('room','-')})\n"
            history = _add_assistant(history, reply)
            last_doctors = doctors
            return (
                _format_history(history), history, booking_info, booking_done, False, last_doctors,
                gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
                gr.update(value=""),          # user_input
                gr.update(value=""),          # name_input
                gr.update(value=None),        # date_input
                gr.update(value=None),        # time_input
                gr.update(value="(Không chọn)"), # doctor_dropdown
                gr.update(value="")           # email_input
            )

        # --- Trường hợp: chọn "bác sĩ số X" ---
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

        # --- Trường hợp: nhiều bác sĩ trùng ---
        if isinstance(chosen, list) and len(chosen) > 1:
            reply = "❌ Có nhiều bác sĩ trùng tên, vui lòng chọn rõ hơn:\n"
            for i, doc in enumerate(chosen, 1):
                reply += f"{i}. {doc['name']} (Phòng {doc.get('room','-')})\n"
            history = _add_assistant(history, reply)
            last_doctors = chosen
            return (
                _format_history(history), history, booking_info, booking_done, False, last_doctors,
                gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
                gr.update(value=""), gr.update(value=""), gr.update(value=None),
                gr.update(value=None), gr.update(value="(Không chọn)"), gr.update(value="")
            )

        # --- Trường hợp: không tìm thấy ---
        if not chosen:
            history = _add_assistant(history, "❌ Mình không tìm thấy bác sĩ phù hợp.")
            last_doctors = doctors
            return (
                _format_history(history), history, booking_info, booking_done, False, last_doctors,
                gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
                gr.update(value=""), gr.update(value=""), gr.update(value=None),
                gr.update(value=None), gr.update(value="(Không chọn)"), gr.update(value="")
            )

        # --- Nếu đã chọn được bác sĩ ---
        date_extracted, _ = extract_datetime(message)
        target_date = date_extracted or date.today().strftime("%Y-%m-%d")
        try:
            slots = get_available_slots(chosen["id"], target_date) or []
        except Exception as e:
            logging.debug(f"get_available_slots error: {e}")
            slots = []

        if not slots:
            history = _add_assistant(
                history,
                f"❌ Không có lịch trống cho bác sĩ {chosen.get('name')} vào ngày {target_date}."
            )
        else:
            reply = f"📅 Lịch trống của bác sĩ {chosen.get('name')} ngày {target_date}:\n"
            try:
                reply += format_slots_human_readable(slots)
            except Exception:
                reply += ", ".join([s.get("time", str(s)) for s in slots])
            history = _add_assistant(history, reply)

        last_doctors = [chosen]
        return (
            _format_history(history), history, booking_info, booking_done, False, last_doctors,
            gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
            gr.update(value=""), gr.update(value=""), gr.update(value=None),
            gr.update(value=None), gr.update(value="(Không chọn)"), gr.update(value="")
        )

    # BOOKING REQUEST
    if intent == "booking_request":
        date_extracted, time_extracted = extract_datetime(message)

        if date_extracted:
            booking_info["Ngay"] = date_extracted
        if time_extracted:
            booking_info["Gio"] = f"{time_extracted}:00" if len(time_extracted) == 5 else time_extracted

        # Prefill values
        date_prefill = date_extracted
        time_prefill = None
        if time_extracted:
            time_prefill = map_time_to_slot(time_extracted, generate_time_slots("06:00", "18:00", 30))

        history = _add_assistant(history, "📅 Mình mở form đặt lịch — bạn vui lòng điền thông tin bên dưới.")
        return (
            _format_history(history), history, booking_info, False, False, last_doctors,
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),       # clear user_input
            gr.update(value=""),       # name_input
            gr.update(value=date_prefill),   # date_input
            gr.update(value=time_prefill),   # time_input (interval)
            gr.update(value="(Không chọn)"),
            gr.update(value="")
        )


    # === FALLBACK ===
    try:
        reply = model.generate_content(message).text
    except Exception as e:
        logging.debug(f"fallback generate error: {e}")
        reply = "Mình chưa rõ ý, bạn có thể nói lại được không?"
    history = _add_assistant(history, reply)
    return (
        _format_history(history), history, booking_info, booking_done, cancel_mode, last_doctors,
        gr.update(visible=False), gr.update(visible=booking_done), gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=""), gr.update(value=None), gr.update(value=None),
        gr.update(value="(Không chọn)"), gr.update(value="")
    )

# ====== Submit booking ======
def submit_booking(name, date_val, time_val, doctor_choice, email, history, booking_info):
    history = history or []
    booking_info = booking_info or {}

    # Validate name
    if not name or not str(name).strip():
        history = _add_assistant(history, "❌ Họ tên là bắt buộc. Vui lòng nhập lại.")
        return (_format_history(history), history, booking_info,
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=""))

    # Debug date_val trước khi parse
    print("🔍 [DEBUG] date_val:", date_val, "| type:", type(date_val))
    # Parse date
    date_obj = parse_date_input(date_val)
    # Debug kết quả sau khi parse
    print("🔍 [DEBUG] date_obj:", date_obj, "| type:", type(date_obj))

    # Validate ngày
    ok, msg = is_valid_booking_date(date_obj)
    if not ok:
        history = _add_assistant(history, msg)
        return (_format_history(history), history, booking_info,
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=""))

    date_str = date_obj.isoformat()

    # Normalize time (handle Dropdown label "HH:MM - HH:MM")
    t_str = str(time_val).strip() if time_val is not None else ""
    # Nếu dropdown trả về interval label "HH:MM - HH:MM", lấy phần start
    if " - " in t_str:
        start = t_str.split(" - ")[0].strip()
        # chuẩn hoá "9:00" -> "09:00"
        if len(start) == 4:
            start = start.zfill(5)
        t_full = start + ":00"   # "16:00" -> "16:00:00"
    else:
        # Có thể là "HH:MM" hoặc "HH:MM:SS"
        if len(t_str) == 5 and t_str.count(":") == 1:
            t_full = t_str + ":00"
        else:
            t_full = t_str  # giữ nguyên nếu đã là HH:MM:SS

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
            history = _add_assistant(
                history,
                f"❌ Bác sĩ {chosen_doctor['name']} không rảnh vào {t_full[:5]} {date_obj.strftime('%d/%m/%Y')}. Vui lòng chọn lại."
            )
            booking_info = {"HoTen": name, "Ngay": date_str, "Gio": t_full, "email": email}
            return (_format_history(history), history, booking_info,
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=""))

    if not available:
        history = _add_assistant(
            history,
            f"❌ Không có bác sĩ nào rảnh {t_full[:5]} {date_obj.strftime('%d/%m/%Y')}. Vui lòng chọn lại."
        )
        booking_info = {"HoTen": name, "Ngay": date_str, "email": email}
        return (_format_history(history), history, booking_info,
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=""))

    # ✅ Nếu người dùng không chọn bác sĩ nhưng có bác sĩ rảnh → tự động chọn
    if not chosen_doctor and available:
        chosen_doctor = available[0]
        history = _add_assistant(
            history,
            f"✅ Hệ thống đã tự động chọn bác sĩ {chosen_doctor['name']} (Phòng {chosen_doctor['room']}) cho bạn."
        )

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
        "DiaChi": "123 Đường ABC, Quận 1, TP.HCM",
        "TenBenhVien": "Hehe Hospital",
        "Hotline": "1900 123 456",
        "EmailCSKH": "cskh@hehehospital.vn",
        "ZaloChatLink": "https://zalo.me/hehehospital",
        "LinkDoiHuy": "https://hehehospital.vn/lichkham",
        "GioLamViec": "Thứ 2 - Thứ 7, 6:00 - 18:00",
        "Website": "https://hehehospital.vn"
    }

    # Summary & email
    try:
        weekday_map = {0: "Thứ Hai", 1: "Thứ Ba", 2: "Thứ Tư", 3: "Thứ Năm", 4: "Thứ Sáu", 5: "Thứ Bảy", 6: "Chủ Nhật"}
        weekday = weekday_map[date_obj.weekday()]
        date_human = date_obj.strftime("%d/%m/%Y")
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
    history = _add_assistant(history, summary)

    if email:
        try:
            subject, body = write_confirm_email_v2({**booking_info, "Hotline": sdt})
            send_email(email, from_email_default, password_default, subject, body)
            history = _add_assistant(history, "✅ Email xác nhận đã được gửi tới " + email)
        except Exception as e:
            logging.debug(f"send_email error: {e}")
            history = _add_assistant(history, "⚠️ Gửi email thất bại nhưng lịch đã được lưu cục bộ.")
    else:
        history = _add_assistant(history, "✅ Lịch khám đã được ghi nhận. (Không có email)")

    return (_format_history(history), history, booking_info,
            gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(value=""))

# ====== Change / Cancel handlers ======
def on_click_change(history, booking_info):
    history = history or []
    booking_info = booking_info or {}
    history = _add_assistant(history, "🔁 Bạn đã chọn đổi lịch — mời chỉnh thông tin trong form.")
    # show form, hide post_booking_row
    return (_format_history(history), history, gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value=""))

def on_click_cancel_mode(history):
    history = history or []
    history = _add_assistant(history, "❌ Vui lòng nhập lý do huỷ lịch và nhấn Xác nhận huỷ.")
    # return chat + show cancel_col
    return (_format_history(history), history, gr.update(visible=True), gr.update(value=""))

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
            history = _add_assistant(history, "✅ Email huỷ lịch đã được gửi.")
        except Exception as e:
            logging.debug(f"send cancel email error: {e}")
            history = _add_assistant(history, "⚠️ Gửi email huỷ thất bại nhưng lịch đã được huỷ cục bộ.")
    history = _add_assistant(history, "🗑️ Lịch khám đã được huỷ.")
    # clear booking_info and hide cancel form
    return (_format_history(history), history, {}, gr.update(visible=False), gr.update(value=""))

# ====== Build UI ======
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chatbot Đặt lịch khám")

    # States
    history = gr.State([])
    booking_info = gr.State({})
    booking_done = gr.State(False)
    cancel_mode = gr.State(False)
    last_doctor_list_state = gr.State([])

    chatbot = gr.Chatbot(type="messages", label="",group_consecutive_messages=False)

    # Form area (hidden by default)
    with gr.Group(visible=False) as form_col:
        name_input = gr.Textbox(label="Họ tên (*)")
        date_input = gr.DateTime(label="Ngày khám (*)", include_time=False)
        time_slots = generate_time_slots("06:00", "18:00", 30)
        time_input = gr.Dropdown(choices=time_slots,label="Giờ khám (*)",value="09:00 - 09:30")
        doctor_dropdown = gr.Dropdown(choices=["(Không chọn)"] + [f"{d['name']} (Phòng {d.get('room','-')})" for d in get_all_doctors()], label="Bác sĩ (optional)")
        email_input = gr.Textbox(label="Email (optional)")
        submit_btn = gr.Button("Xác nhận đặt lịch")

    # After booking: show options Change / Cancel (hidden default)
    with gr.Row(visible=False) as post_booking_row:
        change_btn = gr.Button("🔁 Đổi lịch")
        cancel_btn = gr.Button("❌ Huỷ lịch")

    # Cancel column (hidden default)
    with gr.Group(visible=False) as cancel_col:
        cancel_reason_input = gr.Textbox(label="Lý do huỷ (optional)")
        confirm_cancel_btn = gr.Button("Xác nhận huỷ")

    # Textbox nhập chat luôn ở dưới cùng
    user_input = gr.Textbox(placeholder="Nhập tin nhắn...", lines=1)

    # Bindings: NOTE we added user_input to outputs so we can clear it (gr.update(value=""))
    user_input.submit(
        fn=on_user_message,
        inputs=[user_input, history, booking_info, booking_done, cancel_mode, last_doctor_list_state],
        outputs=[
            chatbot, history, booking_info, booking_done, cancel_mode, last_doctor_list_state,
            form_col, post_booking_row, cancel_col, user_input,
            name_input, date_input, time_input, doctor_dropdown, email_input
        ],
        queue=False
    )


    submit_btn.click(
        fn=submit_booking,
        inputs=[name_input, date_input, time_input, doctor_dropdown, email_input, history, booking_info],
        outputs=[chatbot, history, booking_info, form_col, post_booking_row, cancel_col, user_input],
        queue=False
    )

    change_btn.click(
        fn=on_click_change,
        inputs=[history, booking_info],
        outputs=[chatbot, history, form_col, post_booking_row, cancel_col, user_input],
        queue=False
    )

    cancel_btn.click(
        fn=on_click_cancel_mode,
        inputs=[history],
        outputs=[chatbot, history, cancel_col, user_input],
        queue=False
    )

    confirm_cancel_btn.click(
        fn=confirm_cancel,
        inputs=[cancel_reason_input, history, booking_info],
        outputs=[chatbot, history, booking_info, cancel_col, user_input],
        queue=False
    )

demo.launch(share=True)
