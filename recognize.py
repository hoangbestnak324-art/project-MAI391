import cv2
import pickle
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import csv
import os
from datetime import datetime

# -------------------------------------------------------------
# 1. KHỞI TẠO FILE ĐIỂM DANH CSV THEO NGÀY
# -------------------------------------------------------------
today_str = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"DiemDanh_{today_str}.csv"

# Nếu file chưa tồn tại -> Tạo mới và ghi dòng tiêu đề (Header)
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["MSSV_HoTen", "Ngay", "ThoiGian"])
    print(f"📄 Đã tạo file điểm danh mới: {csv_filename}")

# Tập hợp (set) dùng để lưu danh sách những người đã điểm danh trong phiên làm việc
attended_students = set()

# Đọc lại các SV đã điểm danh trước đó trong ngày (nếu khởi động lại chương trình)
if os.path.exists(csv_filename):
    with open(csv_filename, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader, None) # Bỏ qua header
        for row in reader:
            if row:
                attended_students.add(row[0])

# -------------------------------------------------------------
# 2. KHỞI TẠO MÔ HÌNH INSIGHTFACE & DATABASE
# -------------------------------------------------------------
print("🔄 Đang khởi tạo mô hình InsightFace...")
app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

try:
    with open("database.pkl", "rb") as f:
        known_embeddings = pickle.load(f)
    print(f"✅ Đã tải Database với {len(known_embeddings)} sinh viên.")
except Exception as e:
    print(f"❌ Lỗi không đọc được database.pkl: {e}")
    exit()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

THRESHOLD = 0.3
cap = cv2.VideoCapture(0)

print("🚀 Đang mở Webcam... Nhấn phím 'q' trên cửa sổ video để thoát.")

frame_count = 0
SKIP_FRAMES = 3 
cached_faces = [] 

# -------------------------------------------------------------
# 3. VÒNG LẶP XỬ LÝ VIDEO & ĐIỂM DANH
# -------------------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể kết nối Webcam!")
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES == 0:
        cached_faces = app.get(frame)

    for face in cached_faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        current_embedding = face.normed_embedding

        max_similarity = -1.0
        identity = "Unknown"

        for name, saved_embedding in known_embeddings.items():
            sim = cosine_similarity(current_embedding, saved_embedding)
            if sim > max_similarity:
                max_similarity = sim
                if sim >= THRESHOLD:
                    identity = name

        # Nếu nhận diện thành công
        if identity != "Unknown":
            color = (0, 255, 0) # Xanh lá
            text = f"{identity} ({max_similarity:.2f})"

            # THỰC HIỆN ĐIỂM DANH (Chỉ ghi nếu chưa điểm danh)
            if identity not in attended_students:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                # Ghi vào file CSV
                with open(csv_filename, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([identity, date_str, time_str])

                # Thêm vào danh sách đã điểm danh để không ghi trùng
                attended_students.add(identity)
                print(f"🎉 [ĐIỂM DANH THÀNH CÔNG] {identity} lúc {time_str}")

        else:
            color = (0, 0, 255) # Đỏ
            text = f"Unknown ({max_similarity:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Attendance - Terminal Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Đã đóng Webcam an toàn.")