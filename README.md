# People Counting in a Room
## YOLOv8 + OpenCV + CustomTkinter

### Nhóm thực hiện
| MSSV | Họ và tên | Email |
| --- | --- | --- |
| 2312607 | Nguyễn Ngọc Hân | 2312607@dlu.edu.vn |
| 2312697 | Nguyễn Thị Trường Nga | 2312697@dlu.edu.vn |


> Ứng dụng đếm số người trong phòng từ input video, hiển thị theo thời gian thực và cảnh báo khi vượt ngưỡng.

---

## 1) Mô tả bài toán

Chương trình đọc video đầu vào, dùng mô hình `YOLOv8` để phát hiện đối tượng trong từng frame, sau đó:
- Chỉ giữ lại đối tượng lớp `person`
- Vẽ khung nhận diện quanh từng người
- Đếm số người trong frame hiện tại
- Hiển thị số lượng + FPS + cảnh báo trên màn hình video
- Tùy chọn lưu ảnh định kỳ theo số giây cấu hình

---

## 2) Tính năng chính

- Nhận diện người theo thời gian thực (`person` only)
- Hiển thị:
  - `People Count`
  - `Threshold`
  - `FPS`
- Cảnh báo khi số người vượt ngưỡng:
  - `WARNING: Too many people!`
- Lưu screenshot theo chu kỳ (`Screenshot interval (s)`)
- Giao diện trực quan bằng `CustomTkinter`
- Thoát bằng `ESC` hoặc đóng cửa sổ video (`X`)

---

## 3) Cấu trúc dự án

```text
CameraTracking/
├── main.py                # Giao diện UI, validate input, gọi xử lý
├── detection_model.py     # YOLOv8 + OpenCV (detect/count/FPS/warning/save image)
├── requirements.txt       # Danh sách thư viện
├── videos/
│   └── room.mp4           # Video mẫu
├── screenshots/           # Tự tạo khi bật lưu ảnh
└── README.md
```

---

## 4) Yêu cầu môi trường

- Python 3.x
- Hệ điều hành: Windows / Linux / macOS
- Có file video đầu vào (`.mp4`, `.avi`, `.mov`, `.mkv`, ...)

---

## 5) Cài đặt và chạy (Windows PowerShell)

### Bước 1: Tạo môi trường ảo

```powershell
cd D:\CameraTracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Bước 2: Cài thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install --no-cache-dir -r requirements.txt
```

### Bước 3: Chạy chương trình

```powershell
python main.py
```

---

## 6) Hướng dẫn sử dụng giao diện

Sau khi chạy `main.py`, điền các trường trong UI:

- `Video file`: chọn video cần xử lý
- `YOLO model`: mặc định `yolov8n.pt` (khuyến nghị giữ mặc định)
- `Threshold`: ngưỡng số người để cảnh báo
- `Screenshot interval (s)`: chu kỳ chụp ảnh, ví dụ `5`, `10`
- `Screenshot folder`: thư mục lưu ảnh, mặc định `screenshots`
- `Save screenshot continuously by interval`: bật để lưu ảnh định kỳ

Cuối cùng nhấn **Start Detection** để chạy.

- **About**: mở cửa sổ hướng dẫn chức năng và ý nghĩa từng mục trong app (không cần mở README).

---

## 7) Kết quả đầu ra

Trong cửa sổ video:
- Khung nhận diện xanh quanh từng người
- Nhãn dạng `PERSON | xx.x%`
- Dòng thống kê `People Count`, `Threshold`, `FPS`
- Cảnh báo đỏ khi vượt ngưỡng

Khi bật lưu ảnh định kỳ:
- Ảnh được lưu trong thư mục `screenshots/`
- Tên file dạng: `output_YYYYMMDD_HHMMSS_xxxxxx.jpg`

---

## 8) Lỗi thường gặp và cách xử lý

### `ModuleNotFoundError: No module named 'customtkinter'`

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir -r requirements.txt
```

### Lỗi NumPy `_multiarray_umath`

Nguyên nhân phổ biến: môi trường ảo bị lẫn package từ phiên bản Python khác.

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir -r requirements.txt
```

### Lỗi Torch `WinError 126 ... shm.dll`

- Cài [Microsoft VC++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- Khởi động lại máy rồi chạy lại chương trình

### `pip` không nhận lệnh

Thay vì `pip`, dùng:

```powershell
python -m pip install -r requirements.txt
```

---

## 9) Thư viện sử dụng

- `ultralytics`
- `opencv-python`
- `customtkinter`
