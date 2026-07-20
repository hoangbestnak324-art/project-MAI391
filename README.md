# Hệ Thống Nhận Diện Khuôn Mặt & Điểm Danh Sinh Viên (VisionSync)

Hệ thống điểm danh tự động thời gian thực sử dụng **YOLOv8-face** (phát hiện khuôn mặt), **InsightFace ArcFace** (trích xuất vector đặc trưng 512D), **SQLite** (lưu trữ) và **FastAPI + WebSockets** (giao diện web).

---

## 🛠️ 1. Yêu cầu & Cài đặt môi trường

> **📌 Lưu ý về Python:**  
> Python toàn cục trên máy bạn có thể là **Python 3.14**, nhưng các thư viện AI (`onnxruntime`, `insightface`) chưa hỗ trợ Python 3.14.  
> Vì vậy, dự án sử dụng môi trường ảo **`.venv`** chạy trên **Python 3.12** để đảm bảo tương thích 100%.

### Bước 1: Cài đặt Python 3.12 (nếu chưa có)
- **macOS (Homebrew):**
  ```bash
  brew install python@3.12
  ```

### Bước 2: Tạo & Kích hoạt Môi trường ảo (`.venv`)
```bash
# Tạo venv bằng Python 3.12
/usr/local/bin/python3.12 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate
```

### Bước 3: Cài đặt các thư viện cần thiết
```bash
pip install opencv-python==4.9.0.80 "scipy<1.14" "numpy<2" ultralytics insightface onnxruntime fastapi uvicorn jinja2 python-multipart ipykernel
```

---

## 🚀 2. Hướng dẫn chạy dự án

### Bước 1: Trích xuất Vector & Tạo Database (`database.db`)
Thêm ảnh sinh viên vào thư mục `Dataset/<Mã_Sinh_Viên>/` (ví dụ: `Dataset/SV001/img1.jpg`).
- **Cách 1 (VS Code Jupyter):**
  Mở file `MAI.ipynb`, chọn Kernel **`.venv (Python 3.12)`** và bấm **Run All**.
- **Cách 2 (Terminal):**
  ```bash
  source .venv/bin/activate
  python -c "import database; print('Database ready!')"
  ```

### Bước 2: Khởi chạy Web Server Điểm Danh
```bash
source .venv/bin/activate
uvicorn app:app --reload --port 8000
```

### Bước 3: Trải nghiệm
Mở trình duyệt truy cập: **[http://localhost:8000](http://localhost:8000)** để xem live stream điểm danh qua Webcam.
