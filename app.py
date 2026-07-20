import cv2
import numpy as np
import json
import asyncio
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis

from database import init_db, load_all_students, log_attendance, get_today_attendance

# Initialize database
init_db()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load Models
print("🔄 Loading AI Models...")
yolo_model = YOLO('yolov8n-face.pt')
face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Models loaded!")

# Load known students
known_students = load_all_students()
THRESHOLD = 0.3

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

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

async def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_skip = 2
    frame_count = 0
    cached_identities = []
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_count += 1
            if frame_count % frame_skip == 0:
                cached_identities = []
                # 1. Phát hiện khuôn mặt bằng YOLOv8
                results = yolo_model(frame, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        h, w, _ = frame.shape
                        padding = 10
                        px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
                        px2, py2 = min(w, x2 + padding), min(h, y2 + padding)
                        
                        face_crop = frame[py1:py2, px1:px2]
                        if face_crop.size == 0:
                            continue
                            
                        # 2. Trích xuất vector 512D bằng InsightFace
                        faces = face_app.get(face_crop)
                        if len(faces) > 0:
                            current_embedding = faces[0].normed_embedding
                        else:
                            faces_full = face_app.get(frame)
                            if len(faces_full) > 0:
                                current_embedding = faces_full[0].normed_embedding
                            else:
                                continue
                                
                        # 3. So khớp với CSDL
                        max_similarity = -1.0
                        identity_code = "Unknown"
                        identity_name = "Unknown"
                        
                        for student_code, data in known_students.items():
                            sim = cosine_similarity(current_embedding, data["embedding"])
                            if sim > max_similarity:
                                max_similarity = sim
                                if sim >= THRESHOLD:
                                    identity_code = student_code
                                    identity_name = data["name"]
                        
                        cached_identities.append((x1, y1, x2, y2, identity_code, identity_name, float(max_similarity)))
                        
                        # 4. Ghi nhận điểm danh & broadcast WebSocket
                        if identity_code != "Unknown":
                            is_new = log_attendance(identity_code)
                            if is_new:
                                # Gửi update cho tất cả client
                                asyncio.create_task(manager.broadcast(json.dumps({
                                    "type": "new_attendance",
                                    "student_code": identity_code,
                                    "name": identity_name
                                })))
            
            # Vẽ bounding box
            for (x1, y1, x2, y2, code, name, sim) in cached_identities:
                color = (0, 255, 0) if code != "Unknown" else (0, 0, 255)
                text = f"{name} ({sim:.2f})" if code != "Unknown" else f"Unknown ({sim:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Non-blocking yield for async loop
            await asyncio.sleep(0.01)

@app.get("/")
async def index(request: Request):
    # Lấy danh sách điểm danh hôm nay
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
    except WebSocketDisconnect:
        manager.disconnect(websocket)
