import sqlite3
from datetime import datetime, timedelta

DB_PATH = r"C:\Users\Acer\OneDrive\Desktop\Intern\A2A_Hospital\Booking_Agent\schedule.db"

# ==========================
# Kết nối DB
# ==========================
def get_connection():
    return sqlite3.connect(DB_PATH)


# ==========================
# Bác sĩ
# ==========================
def get_all_doctors():
    """
    Trả về danh sách bác sĩ dưới dạng list[dict]:
    [
        {"id": 1, "name": "Bác sĩ A", "room": "101"},
        {"id": 2, "name": "Bác sĩ B", "room": "102"},
        ...
    ]
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, room FROM doctors")
        rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "room": r[2]} for r in rows]

# ==========================
# Lịch làm việc cố định
# ==========================
def get_work_shifts_by_day(day_of_week):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT doctor_id, start_time, end_time
            FROM work_shifts
            WHERE day_of_week = ?
        """, (day_of_week,))
        rows = cur.fetchall()
        shifts = [{"doctor_id": r[0], "start_time": r[1], "end_time": r[2]} for r in rows]
    print(f"[DEBUG] get_work_shifts_by_day({day_of_week}) -> {shifts}")
    return shifts


# ==========================
# Ngoại lệ ca làm việc
# ==========================
def get_shift_exceptions_by_date(doctor_id, date):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT start_time, end_time, is_available
            FROM shift_exceptions
            WHERE doctor_id = ? AND date = ?
        """, (doctor_id, date))
        rows = cur.fetchall()
        exceptions = [{"start_time": r[0], "end_time": r[1], "is_available": r[2]} for r in rows]
    print(f"[DEBUG] get_shift_exceptions_by_date(doc={doctor_id}, date={date}) -> {exceptions}")
    return exceptions


# ==========================
# Lịch hẹn
# ==========================
def get_appointments_by_date(doctor_id, date):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT start_time, end_time
            FROM appointments
            WHERE doctor_id = ? AND date = ? AND status = 'booked'
        """, (doctor_id, date))
        rows = cur.fetchall()
        appts = [{"start_time": r[0], "end_time": r[1]} for r in rows]
    print(f"[DEBUG] get_appointments_by_date(doc={doctor_id}, date={date}) -> {appts}")
    return appts


def to_time(value):
    """Chuyển string 'HH:MM:SS' hoặc 'HH:MM' thành datetime.time"""
    if isinstance(value, str):
        fmt = "%H:%M:%S" if len(value.split(":")) == 3 else "%H:%M"
        return datetime.strptime(value, fmt).time()
    return value  # đã là datetime.time thì giữ nguyên

# ==========================
# Check 1 bác sĩ có rảnh không
# ==========================
def is_doctor_available(doctor_id, date, time):
    desired_start = to_time(time)
    desired_end = (datetime.combine(datetime.today(), desired_start) + timedelta(minutes=30)).time()
    dow = date.weekday()  # 0=Mon -> 6=Sun

    print(f"[DEBUG] Checking availability for doctor={doctor_id}, date={date}, time={desired_start}-{desired_end}")

    # 1. Lấy lịch làm việc cố định
    shifts = get_work_shifts_by_day(dow)
    doctor_shifts = [
        {"start_time": to_time(s["start_time"]), "end_time": to_time(s["end_time"])}
        for s in shifts if s["doctor_id"] == doctor_id
    ]
    print(f"[DEBUG] Regular shifts for doctor={doctor_id}: {doctor_shifts}")

    has_regular_shift = any(s["start_time"] <= desired_start and s["end_time"] >= desired_end for s in doctor_shifts)
    if not has_regular_shift:
        print(f"[DEBUG] ❌ Doctor {doctor_id} không có ca cố định trùng khung giờ {desired_start}-{desired_end}")
        return False

    # 2. Check ngoại lệ
    exceptions = get_shift_exceptions_by_date(doctor_id, date.strftime("%Y-%m-%d"))
    for exc in exceptions:
        start, end, is_available = to_time(exc["start_time"]), to_time(exc["end_time"]), exc["is_available"]
        if start <= desired_start < end or start < desired_end <= end:
            if not is_available:
                print(f"[DEBUG] ❌ Doctor {doctor_id} bị ngoại lệ NGHỈ {start}-{end}")
                return False
            else:
                print(f"[DEBUG] ✅ Doctor {doctor_id} có ngoại lệ ĐƯỢC PHÉP {start}-{end}")
                return True

    # 3. Check trùng lịch hẹn
    appts = get_appointments_by_date(doctor_id, date.strftime("%Y-%m-%d"))
    for appt in appts:
        a_start, a_end = to_time(appt["start_time"]), to_time(appt["end_time"])
        if (a_start <= desired_start < a_end) or (a_start < desired_end <= a_end):
            print(f"[DEBUG] ❌ Doctor {doctor_id} đã có lịch hẹn {a_start}-{a_end}")
            return False

    print(f"[DEBUG] ✅ Doctor {doctor_id} AVAILABLE {desired_start}-{desired_end}")
    return True


# ==========================
# Tìm tất cả bác sĩ rảnh
# ==========================
def find_available_doctors(date_str, time_str):
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    print(f"[DEBUG] Finding available doctors on {date} from {time_str} to {(datetime.strptime(time_str, '%H:%M:%S') + timedelta(minutes=30)).time()}")

    available = []
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, room FROM doctors")
        doctors = cur.fetchall()

    for doc in doctors:
        doc_id, name, room = doc
        if is_doctor_available(doc_id, date, time_str):
            available.append({"id": doc_id, "name": name, "room": room})
        else:
            print(f"[DEBUG] Doctor {doc_id} ({name}) NOT available at {time_str} on {date}")

    print(f"[DEBUG] Available doctors result -> {available}")
    return available



# ==========================
# Lấy Lịch trống trong ngày bất kỳ của bác sĩ bất kỳ
# ==========================
def get_available_slots(doctor_id: str, date: str, step=30):
    """
    Trả về list các slot trống [(HH:MM, HH:MM), ...] cho doctor_id trong ngày date (YYYY-MM-DD).
    - Lấy work_shifts theo ngày trong tuần
    - Áp dụng shift_exceptions (nghỉ hoặc tăng ca)
    - Loại bỏ giờ đã có appointments
    """

    # ===== Helpers =====
    def time_to_dt(time_str):
        """parse time string -> datetime on given date"""
        fmt_candidates = ("%H:%M:%S", "%H:%M")
        for fmt in fmt_candidates:
            try:
                t = datetime.strptime(time_str, fmt).time()
                return datetime.combine(date_dt.date(), t)
            except Exception:
                continue
        raise ValueError(f"Unrecognized time format: {time_str}")

    def overlaps(a_s, a_e, b_s, b_e):
        """check overlap between 2 intervals"""
        return not (a_e <= b_s or a_s >= b_e)

    # ===== Chuẩn bị date, weekday =====
    try:
        date_dt = datetime.strptime(date, "%Y-%m-%d")
    except Exception as e:
        print(f"[get_available_slots] invalid date format: {date} -> {e}")
        return []

    day_of_week = date_dt.weekday()

    # ===== 1) lấy ca làm định kỳ =====
    raw_shifts = get_work_shifts_by_day(day_of_week)
    shifts = []
    for row in raw_shifts:
        if isinstance(row, dict):
            d_id = str(row.get("doctor_id") or row.get("id"))
            s, e = row.get("start_time"), row.get("end_time")
        else:
            d_id, s, e = str(row[0]), row[1], row[2]
        if d_id == str(doctor_id) and s and e:
            shifts.append((s, e))

    print(f"[get_available_slots] base shifts for doctor {doctor_id}: {shifts}")

    # ===== 2) áp dụng ngoại lệ =====
    raw_excs = get_shift_exceptions_by_date(doctor_id, date)
    for exc in raw_excs:
        if isinstance(exc, dict):
            s, e, avail = exc.get("start_time"), exc.get("end_time"), exc.get("is_available")
        else:
            s, e, avail = exc
        try:
            exc_s = time_to_dt(s)
            exc_e = time_to_dt(e)
        except Exception as ex:
            print(f"[get_available_slots] skip invalid exception: {exc} -> {ex}")
            continue

        # normalize avail
        try:
            avail_flag = int(avail)
        except Exception:
            avail_flag = 1 if str(avail).lower() in ("1", "true", "yes") else 0

        if avail_flag == 0:
            # xoá bất kỳ shift nào bị overlap với exception
            new_shifts = []
            for ss, ee in shifts:
                try:
                    shift_s, shift_e = time_to_dt(ss), time_to_dt(ee)
                except Exception:
                    continue
                if overlaps(shift_s, shift_e, exc_s, exc_e):
                    print(f"[get_available_slots] remove shift {ss}-{ee} due to exception {s}-{e}")
                else:
                    new_shifts.append((ss, ee))
            shifts = new_shifts
        else:
            # thêm ca tăng ca nếu chưa có
            if (s, e) not in shifts:
                shifts.append((s, e))

    print(f"[get_available_slots] shifts after exceptions: {shifts}")

    # ===== 3) lấy appointments =====
    raw_appts = get_appointments_by_date(doctor_id, date)
    appts_dt = []
    for ap in raw_appts:
        if isinstance(ap, dict):
            a_s, a_e = ap.get("start_time"), ap.get("end_time")
        else:
            a_s, a_e = ap
        try:
            appts_dt.append((time_to_dt(a_s), time_to_dt(a_e)))
        except Exception as e:
            print(f"[get_available_slots] skip invalid appt {ap}: {e}")

    print(f"[get_available_slots] appointments: {[(x.strftime('%H:%M'), y.strftime('%H:%M')) for x,y in appts_dt]}")

    # ===== 4) sinh slot trống =====
    available = []
    for s, e in shifts:
        try:
            shift_start, shift_end = time_to_dt(s), time_to_dt(e)
        except Exception as ex:
            print(f"[get_available_slots] invalid shift times {s}-{e}: {ex}")
            continue

        cur = shift_start
        while cur + timedelta(minutes=step) <= shift_end:
            slot_s, slot_e = cur, cur + timedelta(minutes=step)
            if not any(overlaps(slot_s, slot_e, a_s, a_e) for a_s, a_e in appts_dt):
                available.append((slot_s.strftime("%H:%M"), slot_e.strftime("%H:%M")))
            cur += timedelta(minutes=step)

    print(f"[get_available_slots] available slots: {available}")
    return available





def format_slots_human_readable(slots):
    """
    Nhận danh sách slot [(start_time, end_time), ...]
    và trả về chuỗi text dễ đọc cho người dùng.
    """
    if not slots:
        return "❌ Hiện không còn khung giờ trống nào trong ngày này."
    lines = ["🕒 Các khung giờ trống:"]
    for start, end in slots:
        lines.append(f"- {start} – {end}")
    return "\n".join(lines)
