import sqlite3
from datetime import datetime, timedelta
from db_setup import DB_NAME, create_tables


def seed_data(conn):
    cur = conn.cursor()

    # ==== Danh sách bác sĩ ====
    doctors = [
        ("10001", "Nguyễn Văn A", "C101"),
        ("10002", "Nguyễn Văn B", "C102"),
        ("10003", "Nguyễn Thanh C", "C103"),
        ("10004", "Soobin Hoàng D", "C104"),
    ]
    cur.executemany("""
        INSERT INTO doctors (id, name, room)
        VALUES (?, ?, ?)
    """, doctors)

    # ==== Work shifts (mỗi bác sĩ nghỉ 1 ngày/tuần) ====
    work_shifts = [
        # --- Bác sĩ A (10001), nghỉ Chủ Nhật (6) ---
        ("morning", "10001", "0", "06:00:00", "11:30:00"),
        ("afternoon", "10001", "0", "13:00:00", "18:00:00"),
        ("morning", "10001", "1", "06:00:00", "11:30:00"),
        ("morning", "10001", "2", "06:00:00", "11:30:00"),
        ("afternoon", "10001", "2", "13:00:00", "18:00:00"),
        ("morning", "10001", "3", "06:00:00", "11:30:00"),
        ("afternoon", "10001", "4", "13:00:00", "18:00:00"),
        ("morning", "10001", "5", "06:00:00", "11:30:00"),

        # --- Bác sĩ B (10002), nghỉ Thứ Hai (0) ---
        ("morning", "10002", "1", "06:00:00", "11:30:00"),
        ("afternoon", "10002", "1", "13:00:00", "18:00:00"),
        ("morning", "10002", "2", "06:00:00", "11:30:00"),
        ("afternoon", "10002", "3", "13:00:00", "18:00:00"),
        ("morning", "10002", "4", "06:00:00", "11:30:00"),
        ("afternoon", "10002", "5", "13:00:00", "18:00:00"),
        ("morning", "10002", "6", "06:00:00", "11:30:00"),

        # --- Bác sĩ C (10003), nghỉ Thứ Tư (2) ---
        ("morning", "10003", "0", "06:00:00", "11:30:00"),
        ("afternoon", "10003", "0", "13:00:00", "18:00:00"),
        ("morning", "10003", "1", "06:00:00", "11:30:00"),
        ("afternoon", "10003", "3", "13:00:00", "18:00:00"),
        ("morning", "10003", "4", "06:00:00", "11:30:00"),
        ("morning", "10003", "5", "06:00:00", "11:30:00"),
        ("afternoon", "10003", "5", "13:00:00", "18:00:00"),
        ("morning", "10003", "6", "06:00:00", "11:30:00"),

        # --- Bác sĩ D (10004), nghỉ Thứ Sáu (4) ---
        ("morning", "10004", "0", "06:00:00", "11:30:00"),
        ("afternoon", "10004", "1", "13:00:00", "18:00:00"),
        ("morning", "10004", "2", "06:00:00", "11:30:00"),
        ("morning", "10004", "3", "06:00:00", "11:30:00"),
        ("afternoon", "10004", "3", "13:00:00", "18:00:00"),
        ("morning", "10004", "5", "06:00:00", "11:30:00"),
        ("afternoon", "10004", "6", "13:00:00", "18:00:00"),
    ]
    cur.executemany("""
        INSERT INTO work_shifts (id, doctor_id, day_of_week, start_time, end_time)
        VALUES (?, ?, ?, ?, ?)
    """, work_shifts)

    # ==== Ngoại lệ mới (trong 14 ngày tới) ====
    shift_exceptions = [
        # 📌 Ngày toàn viện nghỉ (tất cả bác sĩ không làm việc)
        ("10001", "2025-10-05", "00:00:00", "23:59:59", 0),
        ("10002", "2025-10-05", "00:00:00", "23:59:59", 0),
        ("10003", "2025-10-05", "00:00:00", "23:59:59", 0),
        ("10004", "2025-10-05", "00:00:00", "23:59:59", 0),

        # 📌 Tất cả nghỉ buổi sáng ngày 2025-10-10
        ("10001", "2025-10-10", "06:00:00", "11:30:00", 0),
        ("10002", "2025-10-10", "06:00:00", "11:30:00", 0),
        ("10003", "2025-10-10", "06:00:00", "11:30:00", 0),
        ("10004", "2025-10-10", "06:00:00", "11:30:00", 0),

        # 📌 Một số tăng ca / nghỉ đột xuất riêng lẻ
        ("10001", "2025-10-04", "18:00:00", "20:00:00", 1),  # tăng ca
        ("10002", "2025-10-07", "17:00:00", "19:00:00", 1),
        ("10003", "2025-10-09", "06:00:00", "09:00:00", 1),  # tăng ca
        ("10004", "2025-10-11", "14:00:00", "15:00:00", 0),  # nghỉ đột xuất
    ]
    cur.executemany("""
        INSERT INTO shift_exceptions (doctor_id, date, start_time, end_time, is_available)
        VALUES (?, ?, ?, ?, ?)
    """, shift_exceptions)

    # ==== Cuộc hẹn mẫu (tất cả đều bận cùng lúc) ====
    now_iso = datetime.now().isoformat()
    appointments = [
        # 📌 Trường hợp đặc biệt 1: 2025-10-06, tất cả bận 10:00–10:30
        ("00001", "10001", "Thịnh",  "2025-10-06", "10:00:00", "10:30:00", "booked",   now_iso),
        ("00002", "10002", "Minh",   "2025-10-06", "10:00:00", "10:30:00", "booked",   now_iso),
        ("00003", "10003", "An",     "2025-10-06", "10:00:00", "10:30:00", "booked",   now_iso),
        ("00004", "10004", "Hòa",    "2025-10-06", "10:00:00", "10:30:00", "booked",   now_iso),

        # 📌 Trường hợp đặc biệt 2: 2025-10-13, tất cả bận 15:00–15:30
        ("00005", "10001", "Hải",    "2025-10-13", "15:00:00", "15:30:00", "booked",   now_iso),
        ("00006", "10002", "Lan",    "2025-10-13", "15:00:00", "15:30:00", "booked",   now_iso),
        ("00007", "10003", "Phúc",   "2025-10-13", "15:00:00", "15:30:00", "booked",   now_iso),
        ("00008", "10004", "Quân",   "2025-10-13", "15:00:00", "15:30:00", "booked",   now_iso),

        # 📅 Một số lịch hẹn rải rác để kiểm thử
        ("00009", "10001", "Mai",    "2025-10-08", "09:00:00", "09:30:00", "booked",   now_iso),
        ("00010", "10002", "Bảo",    "2025-10-12", "14:00:00", "14:30:00", "canceled", now_iso),
        ("00011", "10003", "Long",   "2025-10-14", "16:00:00", "16:30:00", "booked",   now_iso),
        ("00012", "10004", "Trang",  "2025-10-15", "08:00:00", "08:30:00", "canceled", now_iso),
    ]
    cur.executemany("""
        INSERT INTO appointments (id, doctor_id, patient_name, date, start_time, end_time, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, appointments)

    conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_NAME)
    create_tables(conn)
    seed_data(conn)
    conn.close()
