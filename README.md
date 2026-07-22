# "Face Regconition Attendence System with Yolov8 and ArcFace" - Hệ thống nhận diện khuôn mặt & điểm danh

Hệ thống điểm danh tự động thời gian thực sử dụng **YOLOv8-face** (phát hiện khuôn mặt), **InsightFace ArcFace** (trích xuất vector đặc trưng 512D), **SQLite** (lưu trữ) và **FastAPI + WebSockets** (giao diện web).

---

## 📌 Lưu ý quan trọng về phiên bản Python
> Dự án khuyến nghị sử dụng **Python 3.12** với **NumPy < 2.0**.  
> Nếu máy bạn đang dùng Python 3.14 (hoặc phiên bản khác), hãy tạo môi trường ảo **`.venv`** bằng Python 3.12 như hướng dẫn bên dưới để tránh lỗi thư viện AI (`onnxruntime`, `insightface`).

---

## I. Yêu cầu & Cài đặt môi trường

### Đối với HĐH macOS

#### Bước 1: Cài đặt Python 3.12 (qua Homebrew)
```bash
brew install python@3.12
```

#### Bước 2: Tạo & Kích hoạt Môi trường ảo (`.venv`)
```bash
# Tạo venv bằng Python 3.12
/usr/local/bin/python3.12 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate
```

#### Bước 3: Cài đặt các thư viện
```bash
pip install opencv-python==4.9.0.80 "scipy<1.14" "numpy<2" ultralytics insightface onnxruntime fastapi uvicorn jinja2 python-multipart ipykernel
```

---

### Đối với HĐH Windows

#### Bước 1: Cài đặt Python 3.12
- **Tải từ trang chủ:** Tải bản cài đặt Python 3.12 từ [python.org](https://www.python.org/downloads/release/python-3128/) *(lưu ý tích chọn **Add python.exe to PATH** khi cài)*.
- **Hoặc bằng PowerShell (Admin):**
  ```powershell
  winget install Python.Python.3.12
  ```

#### Bước 2: Tạo & Kích hoạt Môi trường ảo (`.venv`)
Mở Command Prompt (cmd) hoặc PowerShell tại thư mục dự án:
```cmd
:: Tạo venv bằng Python 3.12
py -3.12 -m venv .venv

:: Kích hoạt venv trên Command Prompt (cmd):
.venv\Scripts\activate.bat

:: HOẶC Kích hoạt venv trên PowerShell:
.venv\Scripts\Activate.ps1
```

#### Bước 3: Cài đặt các thư viện
```cmd
pip install opencv-python==4.9.0.80 "scipy<1.14" "numpy<2" ultralytics insightface onnxruntime fastapi uvicorn jinja2 python-multipart ipykernel
```

---

## II. Hướng dẫn chạy dự án

### Bước 1: Trích xuất Vector & Tạo Database (`database.db`)
Thêm ảnh sinh viên vào thư mục `Dataset/<Mã_Sinh_Viên>/` (ví dụ: `Dataset/SV001/img1.jpg`).
- **Cách 1 (VS Code Jupyter):**
  1. Mở file `MAI.ipynb` trong VS Code.
  2. Bấm góc trên bên phải chọn Kernel: **`.venv (Python 3.12)`**.
  3. Chọn **Run All**.
- **Cách 2 (Terminal / Command Prompt):**
  - **macOS:** `source .venv/bin/activate`
  - **Windows:** `.venv\Scripts\activate`
  - Run command:
    ```bash
    python -c "import database; print('Database ready!')"
    ```

### Bước 2: Khởi chạy Web Server Điểm Danh
Lưu ý đảm bảo đã kích hoạt môi trường ảo (`.venv`):
- **macOS:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

Chạy lệnh:
```bash
uvicorn app:app --reload --port 8000
```

### Bước 3: Trải nghiệm
Mở trình duyệt bất kỳ và truy cập địa chỉ:  
👉 **[http://localhost:8000](http://localhost:8000)** để xem live stream điểm danh qua Webcam.
