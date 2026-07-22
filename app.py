import os
import cv2
import numpy as np
import json
import asyncio
import time
import base64
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

from database import init_db, load_all_students, save_student, log_attendance, get_today_attendance

init_db()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Models
print("Loading AI Models ...")
yolo_model = YOLO('yolov8n-face.pt')
face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("Models loaded")

# Cache known students as NumPy matrix for fast vectorized cosine similarity
known_students = {}
known_codes = []
known_names = []
known_matrix = None
last_notification_time = {}
COOLDOWN_SECONDS = 6.0

def refresh_student_cache():
    global known_students, known_codes, known_names, known_matrix
    known_students = load_all_students()
    known_codes = list(known_students.keys())
    known_names = [known_students[c]["name"] for c in known_codes]
    if len(known_codes) > 0:
        known_matrix = np.array([known_students[c]["embedding"] for c in known_codes], dtype=np.float32)
    else:
        known_matrix = None

refresh_student_cache()
THRESHOLD = 0.30

# WebSocket connections manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

def process_attendance_event(identity_code: str, identity_name: str):
    """
    Ghi nhận điểm danh và phát tín hiệu WebSocket tức thì về giao diện web
    """
    now = time.time()
    last_time = last_notification_time.get(identity_code, 0)
    
    if now - last_time >= COOLDOWN_SECONDS:
        is_new = log_attendance(identity_code)
        last_notification_time[identity_code] = now
        time_str = time.strftime("%H:%M:%S")
        
        asyncio.create_task(manager.broadcast(json.dumps({
            "type": "new_attendance",
            "student_code": identity_code,
            "name": identity_name,
            "is_new": is_new,
            "time": time_str
        })))

async def generate_frames():
    refresh_student_cache()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_skip = 2
    frame_count = 0
    cached_identities = []

    try:
        while True:
            if not cap.isOpened():
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera Not Available / In Use", (80, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                await asyncio.sleep(1.0)
                continue

            success, frame = cap.read()
            if not success or frame is None:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % frame_skip == 0:
                cached_identities = []

                # 1. Trích xuất khuôn mặt & vector 512D trực tiếp từ frame bằng InsightFace
                faces = face_app.get(frame)

                if len(faces) > 0:
                    for f in faces:
                        x1, y1, x2, y2 = map(int, f.bbox)
                        current_embedding = f.normed_embedding

                        identity_code = "Unknown"
                        identity_name = "Unknown"
                        max_similarity = -1.0

                        if known_matrix is not None and len(known_matrix) > 0:
                            similarities = np.dot(known_matrix, current_embedding)
                            best_idx = np.argmax(similarities)
                            max_similarity = similarities[best_idx]

                            if max_similarity >= THRESHOLD:
                                identity_code = known_codes[best_idx]
                                identity_name = known_names[best_idx]

                        cached_identities.append((x1, y1, x2, y2, identity_code, identity_name, float(max_similarity)))

                        if identity_code != "Unknown":
                            process_attendance_event(identity_code, identity_name)
                else:
                    # 2. Fallback: Phát hiện bằng YOLOv8-face
                    results = yolo_model(frame, verbose=False, imgsz=320)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            h, w, _ = frame.shape
                            padding = 30
                            px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
                            px2, py2 = min(w, x2 + padding), min(h, y2 + padding)

                            face_crop = frame[py1:py2, px1:px2]
                            if face_crop.size == 0:
                                continue

                            faces_crop = face_app.get(face_crop)
                            if len(faces_crop) > 0:
                                current_embedding = faces_crop[0].normed_embedding

                                identity_code = "Unknown"
                                identity_name = "Unknown"
                                max_similarity = -1.0

                                if known_matrix is not None and len(known_matrix) > 0:
                                    similarities = np.dot(known_matrix, current_embedding)
                                    best_idx = np.argmax(similarities)
                                    max_similarity = similarities[best_idx]

                                    if max_similarity >= THRESHOLD:
                                        identity_code = known_codes[best_idx]
                                        identity_name = known_names[best_idx]

                                cached_identities.append((x1, y1, x2, y2, identity_code, identity_name, float(max_similarity)))

                                if identity_code != "Unknown":
                                    process_attendance_event(identity_code, identity_name)

            # Vẽ bounding box & thông tin nhận diện lên frame
            for (x1, y1, x2, y2, code, name, sim) in cached_identities:
                color = (0, 255, 0) if code != "Unknown" else (0, 0, 255)
                text = f"{name} ({sim:.2f})" if code != "Unknown" else f"Unknown ({sim:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            await asyncio.sleep(0.001)
    finally:
        if cap is not None and cap.isOpened():
            cap.release()

@app.get("/")
async def index(request: Request):
    refresh_student_cache()
    attendance = get_today_attendance()
    return templates.TemplateResponse(request=request, name="index.html", context={"attendance": attendance})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        manager.disconnect(websocket)

@app.post("/api/register_student")
async def register_student(
    student_code: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(None),
    image_data: str = Form(None)
):
    """
    API đăng ký khuôn mặt mới cho sinh viên
    """
    try:
        student_code = student_code.strip()
        name = name.strip()

        if not student_code or not name:
            return JSONResponse({"success": False, "message": "Vui lòng nhập đầy đủ Mã số sinh viên và Họ tên"}, status_code=400)

        img_np = None

        # 1. Đọc ảnh từ file tải lên
        if file and file.filename:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 2. Đọc ảnh từ snapshot webcam base64
        elif image_data:
            if "," in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None or img_np.size == 0:
            return JSONResponse({"success": False, "message": "Không đọc được dữ liệu hình ảnh"}, status_code=400)

        # 3. Trích xuất vector 512D bằng InsightFace
        faces = face_app.get(img_np)
        embedding = None

        if len(faces) > 0:
            embedding = faces[0].normed_embedding
        else:
            # Fallback YOLOv8 + InsightFace crop
            results = yolo_model(img_np, verbose=False, imgsz=320)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w, _ = img_np.shape
                    px1, py1 = max(0, x1 - 30), max(0, y1 - 30)
                    px2, py2 = min(w, x2 + 30), min(h, y2 + 30)
                    crop = img_np[py1:py2, px1:px2]
                    if crop.size > 0:
                        fc = face_app.get(crop)
                        if len(fc) > 0:
                            embedding = fc[0].normed_embedding
                            break
            if embedding is None:
                return JSONResponse({"success": False, "message": "Không tìm thấy khuôn mặt rõ ràng trong ảnh! Vui lòng chọn/chụp lại ảnh khác."}, status_code=400)

        # 4. Lưu ảnh khuôn mặt vào Dataset/<student_code>/
        save_dir = os.path.join("Dataset", student_code)
        os.makedirs(save_dir, exist_ok=True)
        img_filename = f"img_{int(time.time())}.jpg"
        cv2.imwrite(os.path.join(save_dir, img_filename), img_np)

        # 5. Cập nhật SQLite & Bộ nhớ RAM Cache
        save_student(student_code, name, embedding)
        refresh_student_cache()

        return JSONResponse({
            "success": True,
            "message": f"Đã đăng ký thành công cho sinh viên: {name} ({student_code})"
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi hệ thống: {str(e)}"}, status_code=500)

@app.get("/api/students")
async def get_students_api():
    """
    Lấy danh sách tất cả sinh viên đã đăng ký trong hệ thống
    """
    students_dict = load_all_students()
    result = []
    for code, info in students_dict.items():
        result.append({
            "student_code": code,
            "name": info["name"]
        })
    return JSONResponse({"success": True, "students": result})

@app.put("/api/students/{old_code}")
async def update_student_api(old_code: str, request: Request):
    """
    Cập nhật thông tin sinh viên (MSSV / Họ Tên).
    """
    try:
        body = await request.json()
        new_code = body.get("student_code", "").strip()
        new_name = body.get("name", "").strip()

        if not new_code or not new_name:
            return JSONResponse({"success": False, "message": "Thông tin không được để trống"}, status_code=400)

        from database import update_student
        update_student(old_code, new_code, new_name)

        if old_code != new_code and os.path.exists(os.path.join("Dataset", old_code)):
            import shutil
            old_dir = os.path.join("Dataset", old_code)
            new_dir = os.path.join("Dataset", new_code)
            if os.path.exists(new_dir):
                shutil.rmtree(new_dir)
            os.rename(old_dir, new_dir)

        refresh_student_cache()
        return JSONResponse({"success": True, "message": f"Đã cập nhật sinh viên {new_name} ({new_code}) thành công"})
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi: {str(e)}"}, status_code=500)

@app.delete("/api/students/{student_code}")
async def delete_student_api(student_code: str):
    """
    Xóa sinh viên khỏi hệ thống.
    """
    try:
        import shutil
        from database import delete_student
        delete_student(student_code)

        dataset_dir = os.path.join("Dataset", student_code)
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)

        refresh_student_cache()
        return JSONResponse({"success": True, "message": f"Đã xóa thành công sinh viên {student_code}"})
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi: {str(e)}"}, status_code=500)

@app.post("/api/attendance/reset")
async def reset_attendance_api():
    """
    API Reset/Xóa toàn bộ lịch sử điểm danh hôm nay.
    """
    try:
        from database import reset_today_attendance
        deleted_count = reset_today_attendance()

        global last_notification_time
        last_notification_time.clear()

        asyncio.create_task(manager.broadcast(json.dumps({
            "type": "reset_attendance"
        })))

        return JSONResponse({
            "success": True,
            "message": f"Đã reset danh sách điểm danh hôm nay ({deleted_count} sinh viên)"
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi: {str(e)}"}, status_code=500)


@app.get("/api/attendance/export_csv")
async def export_attendance_csv():
    """
    API Xuất danh sách sinh viên đã điểm danh hôm nay ra file CSV với chuẩn UTF-8 BOM
    """
    try:
        from fastapi.responses import Response
        import io
        import csv

        attendance = get_today_attendance()

        output = io.StringIO()
        # Ghi UTF-8 BOM để MS Excel mở file không bị lỗi phông chữ Tiếng Việt
        output.write('\ufeff')
        writer = csv.writer(output)

        # Header bảng CSV
        writer.writerow(["STT", "Mã số sinh viên", "Họ và tên", "Thời gian điểm danh", "Ngày điểm danh"])

        today_str = time.strftime("%Y-%m-%d")
        for idx, item in enumerate(attendance, start=1):
            writer.writerow([
                idx,
                item.get("student_code", ""),
                item.get("name", ""),
                item.get("time", ""),
                today_str
            ])

        csv_data = output.getvalue()
        filename = f"DanhSachDiemDanh_{today_str}.csv"

        return Response(
            content=csv_data.encode('utf-8'),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi xuất CSV: {str(e)}"}, status_code=500)



def recalculate_student_embedding(student_code: str, name: str = None):
    """
    Tính toán lại vector đặc trưng trung bình (mean embedding) chuẩn hóa cho sinh viên
    dựa trên tất cả các ảnh có trong thư mục Dataset/<student_code>/, sau đó cập nhật CSDL SQLite và RAM Cache.
    """
    save_dir = os.path.join("Dataset", student_code)
    if not os.path.exists(save_dir):
        return 0

    if not name:
        all_st = load_all_students()
        if student_code in all_st:
            name = all_st[student_code]["name"]
        else:
            name = student_code

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_files = [f for f in os.listdir(save_dir) if f.lower().endswith(valid_extensions)]

    vectors = []
    for img_file in img_files:
        img_path = os.path.join(save_dir, img_file)
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            continue

        faces = face_app.get(img)
        if len(faces) > 0:
            vectors.append(faces[0].normed_embedding)
        else:
            results = yolo_model(img, verbose=False, imgsz=320)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w, _ = img.shape
                    px1, py1 = max(0, x1 - 30), max(0, y1 - 30)
                    px2, py2 = min(w, x2 + 30), min(h, y2 + 30)
                    crop = img[py1:py2, px1:px2]
                    if crop.size > 0:
                        fc = face_app.get(crop)
                        if len(fc) > 0:
                            vectors.append(fc[0].normed_embedding)
                            break

    if len(vectors) > 0:
        mean_vector = np.mean(vectors, axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)
        save_student(student_code, name, mean_vector)
        refresh_student_cache()
        return len(vectors)
    return 0


@app.get("/api/students/{student_code}/faces")
async def get_student_faces(student_code: str):
    """
    Lấy danh sách các ảnh gương mặt hiện tại của sinh viên từ thư mục Dataset.
    """
    student_code = student_code.strip()
    save_dir = os.path.join("Dataset", student_code)
    if not os.path.exists(save_dir):
        return JSONResponse({"success": True, "faces": [], "count": 0})

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(save_dir) if f.lower().endswith(valid_extensions)]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(save_dir, x)), reverse=True)

    faces = [
        {
            "filename": f,
            "url": f"/api/students/{student_code}/faces/{f}"
        }
        for f in files
    ]
    return JSONResponse({"success": True, "faces": faces, "count": len(faces)})


@app.get("/api/students/{student_code}/faces/{filename}")
async def get_student_face_file(student_code: str, filename: str):
    """
    Trả về file ảnh gương mặt cụ thể của sinh viên.
    """
    file_path = os.path.join("Dataset", student_code, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File không tồn tại"}, status_code=404)
    return FileResponse(file_path)


@app.post("/api/students/{student_code}/faces")
async def add_student_face(
    student_code: str,
    file: UploadFile = File(None),
    image_data: str = Form(None)
):
    """
    Thêm 1 mẫu gương mặt mới cho sinh viên đã tồn tại và cập nhật lại các vector trên CSDL.
    """
    try:
        student_code = student_code.strip()
        all_st = load_all_students()
        if student_code not in all_st:
            return JSONResponse({"success": False, "message": f"Sinh viên {student_code} không tồn tại trong CSDL!"}, status_code=404)

        name = all_st[student_code]["name"]
        img_np = None

        if file and file.filename:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif image_data:
            if "," in image_data:
                image_data = image_data.split(",")[1]
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_np is None or img_np.size == 0:
            return JSONResponse({"success": False, "message": "Không đọc được dữ liệu hình ảnh!"}, status_code=400)

        # Kiểm tra nhận diện gương mặt
        faces = face_app.get(img_np)
        has_face = len(faces) > 0

        if not has_face:
            results = yolo_model(img_np, verbose=False, imgsz=320)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w, _ = img_np.shape
                    px1, py1 = max(0, x1 - 30), max(0, y1 - 30)
                    px2, py2 = min(w, x2 + 30), min(h, y2 + 30)
                    crop = img_np[py1:py2, px1:px2]
                    if crop.size > 0:
                        fc = face_app.get(crop)
                        if len(fc) > 0:
                            has_face = True
                            break

        if not has_face:
            return JSONResponse({"success": False, "message": "Không tìm thấy gương mặt rõ ràng trong ảnh! Vui lòng thử ảnh khác."}, status_code=400)

        save_dir = os.path.join("Dataset", student_code)
        os.makedirs(save_dir, exist_ok=True)
        filename = f"img_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), img_np)

        count = recalculate_student_embedding(student_code, name)

        return JSONResponse({
            "success": True,
            "message": f"Đã thêm gương mặt thành công cho {name}! (Tổng số: {count} ảnh)",
            "filename": filename,
            "count": count
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi: {str(e)}"}, status_code=500)


@app.delete("/api/students/{student_code}/faces/{filename}")
async def delete_student_face(student_code: str, filename: str):
    """
    Xóa 1 mẫu gương mặt chỉ định của sinh viên và tính toán lại vector CSDL.
    """
    try:
        student_code = student_code.strip()
        file_path = os.path.join("Dataset", student_code, filename)

        if not os.path.exists(file_path):
            return JSONResponse({"success": False, "message": "File ảnh không tồn tại!"}, status_code=404)

        os.remove(file_path)

        all_st = load_all_students()
        name = all_st[student_code]["name"] if student_code in all_st else student_code
        count = recalculate_student_embedding(student_code, name)

        return JSONResponse({
            "success": True,
            "message": f"Đã xóa ảnh gương mặt {filename}! (Còn lại: {count} ảnh)",
            "remaining_count": count
        })
    except Exception as e:
        return JSONResponse({"success": False, "message": f"Lỗi: {str(e)}"}, status_code=500)

