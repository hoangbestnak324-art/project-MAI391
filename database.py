import sqlite3
import json
import numpy as np

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Tối ưu hóa hiệu năng truy vấn SQLite bằng WAL mode & Synchronous NORMAL
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    # Bảng lưu thông tin sinh viên và vector khuôn mặt
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_code TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)
    
    # Bảng lưu lịch sử điểm danh
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_code) REFERENCES students (student_code)
        )
    """)
    
    # Tạo Index tăng tốc truy vấn điểm danh theo ngày & sinh viên
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_time ON attendance(student_code, timestamp);")
    
    conn.commit()
    conn.close()

def save_student(student_code: str, name: str, embedding: np.ndarray):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    emb_list = embedding.tolist()
    emb_json = json.dumps(emb_list)
    
    cursor.execute("""
        INSERT OR REPLACE INTO students (student_code, name, embedding)
        VALUES (?, ?, ?)
    """, (student_code, name, emb_json))
    
    conn.commit()
    conn.close()

def update_student(old_code: str, new_code: str, new_name: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    if old_code != new_code:
        cursor.execute("UPDATE students SET student_code = ?, name = ? WHERE student_code = ?", (new_code, new_name, old_code))
        cursor.execute("UPDATE attendance SET student_code = ? WHERE student_code = ?", (new_code, old_code))
    else:
        cursor.execute("UPDATE students SET name = ? WHERE student_code = ?", (new_name, old_code))
        
    conn.commit()
    conn.close()

def delete_student(student_code: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE student_code = ?", (student_code,))
    cursor.execute("DELETE FROM students WHERE student_code = ?", (student_code,))
    conn.commit()
    conn.close()

def load_all_students():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT student_code, name, embedding FROM students")
    rows = cursor.fetchall()
    conn.close()
    
    students_data = {}
    for row in rows:
        student_code, name, emb_json = row
        embedding = np.array(json.loads(emb_json), dtype=np.float32)
        students_data[student_code] = {
            "name": name,
            "embedding": embedding
        }
    return students_data

def log_attendance(student_code: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM attendance 
        WHERE student_code = ? AND date(timestamp, 'localtime') = date('now', 'localtime')
    """, (student_code,))
    
    count = cursor.fetchone()[0]
    
    is_new = False
    if count == 0:
        cursor.execute("INSERT INTO attendance (student_code) VALUES (?)", (student_code,))
        is_new = True
        
    conn.commit()
    conn.close()
    return is_new

def reset_today_attendance():
    """
    Xóa toàn bộ dữ liệu điểm danh trong ngày hôm nay.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        DELETE FROM attendance 
        WHERE date(timestamp, 'localtime') = date('now', 'localtime')
    """)
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count

def get_today_attendance():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT a.id, a.student_code, s.name, time(a.timestamp, 'localtime') as time
        FROM attendance a
        JOIN students s ON a.student_code = s.student_code
        WHERE date(a.timestamp, 'localtime') = date('now', 'localtime')
        ORDER BY a.timestamp DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        results.append({
            "id": row[0],
            "student_code": row[1],
            "name": row[2],
            "time": row[3]
        })
    return results

init_db()
