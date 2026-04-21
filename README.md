# People Counting / Cafe Analytics  
## YOLOv8 + OpenCV + CustomTkinter

### Nhóm thực hiện
| MSSV | Họ và tên | Email |
| --- | --- | --- |
| 2312607 | Nguyễn Ngọc Hân | 2312607@dlu.edu.vn |
| 2312697 | Nguyễn Thị Trường Nga | 2312697@dlu.edu.vn |

> Ứng dụng đếm số người trong phòng / quán cafe từ video hoặc camera, hiển thị theo thời gian thực, gán ID, đo thời gian ngồi và ghi lại thống kê (SQLite + CSV reports).

---

## 1) Mô tả bài toán

Ứng dụng đọc video hoặc stream từ camera, sử dụng `YOLOv10` để phát hiện lớp `person`, sau đó:
- Chỉ giữ lại đối tượng lớp `person`.
- Gán **ID tracking ổn định** cho từng người (ByteTrack/BoT-SORT của Ultralytics).
- Vẽ khung + nhãn `ID | conf | time(s)` quanh mỗi người.
- Đếm số người trong khung hình (có thể chỉ trong **ROI quán cafe**).
- Tính **thời gian ngồi** của từng khách (session: vào – ra).
- Lưu session vào **SQLite** và xuất **báo cáo theo ngày/giờ**.
- Tùy chọn chụp screenshot theo chu kỳ.

Ứng dụng phù hợp cho:
- Demo đếm người trong phòng học / phòng họp.
- Thử nghiệm phân tích hành vi khách trong quán cafe (thời gian ngồi, giờ cao điểm).

---

## 2) Tính năng chính

- **Nhận diện người real-time** bằng `YOLOv10` (mặc định class `person`).
- **Tracking ID** ổn định (ByteTrack/BoT-SORT qua Ultralytics).
- **Overlay thông tin** trên video:
  - `ID`, `% confidence`, `thời gian ngồi (s)` cho từng người.
  - `People Count`, `Threshold`, `FPS`.
  - Cảnh báo khi `People Count > Threshold`: `WARNING: Too many people!`.
- **ROI (Region of Interest)**:
  - Chỉ tính người bên trong vùng ROI (quán cafe).
  - Người ngoài ROI được vẽ xám và không tính vào thống kê.
- **Lưu dữ liệu**:
  - SQLite (`data/cafe_analytics.sqlite3`) – bảng `sessions` (track_id, time_in, time_out, duration…).
  - CSV raw session log (`data/sessions_log.csv`).
  - CSV model stats (`output/model_stats.csv`) – so sánh các phiên bản YOLO.
  - CSV report theo ngày/giờ (`output/report_day.csv`, `output/report_hour.csv`).
- **UI CustomTkinter**:
  - Chọn video / webcam / camera IP.
  - Chọn model YOLO (preset `yolov10m/s/n/l/x`).
  - Thiết lập ngưỡng cảnh báo, screenshot interval, thư mục ảnh.
  - Nút `Start`, `About`, `Exit`.
- Thoát video bằng **ESC** hoặc đóng cửa sổ OpenCV (`X`).

---

## 3) Cấu trúc dự án

```text
CameraTracking/
├── main.py                  # UI CustomTkinter: form cấu hình + gọi PeopleCounterModel
├── detection_model.py       # Pipeline YOLOv10 + tracker + ROI + logging
├── core/
│   ├── __init__.py
│   ├── tracking.py          # Track, SimpleIOUTracker, UltralyticsTrackerAdapter
│   └── sessions.py          # PersonSession, SessionManager, CsvSessionLogger
├── analytics/
│   ├── __init__.py
│   ├── model_stats.py       # ModelRunStats, CsvModelStatsLogger
│   ├── storage_sqlite.py    # SQLiteStore, SessionRow (bảng sessions)
│   └── report_sqlite.py     # Script xuất báo cáo từ SQLite
├── videos/                  # Video mẫu / camera file (tự thêm)
├── screenshots/             # Ảnh chụp màn hình (nếu bật)
├── data/
│   ├── cafe_analytics.sqlite3  # SQLite DB (tự tạo)
│   └── sessions_log.csv        # Log session raw (tự tạo nếu bật CSV)
├── output/
│   ├── model_stats.csv      # Thống kê mỗi lần chạy YOLO (tự tạo)
│   ├── report_day.csv       # Báo cáo theo ngày (tùy chọn)
│   └── report_hour.csv      # Báo cáo theo giờ (tùy chọn)
├── requirements.txt         # Danh sách thư viện
└── README.md
```

---

## 4) Yêu cầu môi trường

- Python 3.x (khuyến nghị 3.10+).
- Hệ điều hành: Windows / Linux / macOS.
- Đã cài:
  - `ultralytics` (kéo theo PyTorch).
  - `opencv-python`.
  - `customtkinter`.
- Có **video đầu vào** (`.mp4`, `.avi`, `.mov`, `.mkv`, ...) hoặc **camera/webcam**.

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

### Bước 3: Chạy chương trình (UI)

```powershell
python main.py
```

---

## 6) Hướng dẫn sử dụng UI

Sau khi chạy `main.py`, điền các trường trong UI:

- **Video file**
  - File video: `videos/room.mp4`, ...
  - Webcam: nhập `0` (hoặc `1` nếu nhiều camera).
  - Camera IP/điện thoại: URL `http://ip:port/video` hoặc `rtsp://...`.
- **YOLO model**
  - Đường dẫn file `.pt` (vd: `yolov8n.pt`) hoặc dùng **combo preset** bên phải để chọn nhanh `yolov8n/s/m/l/x`.
- **Threshold**
  - Ngưỡng số người cho phép; khi `People Count > Threshold` sẽ hiển thị cảnh báo đỏ.
- **Screenshot interval (s)**
  - Mỗi `N` giây sẽ chụp 1 frame nếu bật lưu ảnh.
- **Screenshot folder**
  - Thư mục lưu ảnh (tương đối project), mặc định: `screenshots`.
- **Save screenshot continuously by interval**
  - Bật/tắt lưu ảnh định kỳ.

Cuối cùng nhấn **Start** để chạy.  
Nút **About** mở hướng dẫn chi tiết trong một cửa sổ phụ; **Exit** đóng UI.

---

## 7) Kết quả hiển thị & dữ liệu đầu ra

### 7.1 Trên video (OpenCV window)

- Khung xanh quanh từng người, nhãn:  
  `| ID <id> | <time>s | <conf>% |`
- Dòng thống kê góc trái trên:
  - `People Count: <số khách trong ROI>`.
  - `Threshold: <ngưỡng>`.
  - `FPS: <tốc độ>`.
- Nếu bật ROI:
  - Vẽ vùng `ROI (Cafe Area)` màu tím trong khung hình.
  - Người ngoài ROI được vẽ xám, có label `(out ROI)` và **không tính** vào `People Count`.
- Nếu vượt ngưỡng:
  - Banner đỏ góc phải: `WARNING: Too many people!`.

### 7.2 Ảnh chụp màn hình

- Nếu bật **Save screenshot continuously by interval**:
  - Ảnh lưu trong thư mục `screenshots/`.
  - Tên file: `output_YYYYMMDD_HHMMSS_xxxxxx.jpg`.

### 7.3 Dữ liệu & thống kê

- **SQLite DB**: `data/cafe_analytics.sqlite3`
  - Bảng `sessions`: mỗi dòng là 1 lượt khách (theo tracking ID).
- **Log CSV thô**: `data/sessions_log.csv`
  - `track_id,time_in,time_out,duration_seconds`.
- **Thống kê model**: `output/model_stats.csv`
  - Mỗi lần bấm Start xong sẽ thêm 1 dòng: model, FPS trung bình, số frame, avg/max people...
- **Báo cáo** (tùy chọn, qua script):
  - `output/report_day.csv`: tổng lượt + thời gian trung bình theo ngày.
  - `output/report_hour.csv`: tổng lượt + thời gian trung bình theo giờ.

---

## 8) Báo cáo từ SQLite (Cafe Analytics)

Chạy các lệnh sau trong thư mục project:

### Báo cáo theo ngày

```powershell
python -m analytics.report_sqlite --mode day --out report_day.csv
```

### Báo cáo theo giờ trong 1 ngày cụ thể

```powershell
python -m analytics.report_sqlite --mode hour --day 2026-04-07 --out report_hour.csv
```

Các file CSV sẽ nằm trong thư mục `output/`.

---

## 9) Lỗi thường gặp và cách xử lý

### `ModuleNotFoundError: No module named 'customtkinter'`

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir -r requirements.txt
```

### Lỗi NumPy `_multiarray_umath`

Môi trường ảo bị lẫn package từ bản Python khác.

```powershell
deactivate
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-cache-dir -r requirements.txt
```

### Lỗi Torch `WinError 126 ... shm.dll`

- Cài [Microsoft VC++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- Khởi động lại máy rồi chạy lại chương trình.

### `pip` không nhận lệnh

```powershell
python -m pip install -r requirements.txt
```

---

## 10) Thư viện sử dụng

- `ultralytics` – YOLOv8 + ByteTrack/BoT-SORT.
- `opencv-python` – đọc video, vẽ overlay, hiển thị.
- `customtkinter` – UI desktop hiện đại.

---

## 11) Nguồn video gợi ý

### Video stock (tải nhanh, dùng thử)

Gợi ý từ khóa: `people in room`, `office people`, `crowd`, `meeting room`, `walking people`.

| Nguồn | Link | Ghi chú |
| --- | --- | --- |
| **Pexels** | https://www.pexels.com/videos/ | Nhiều clip miễn phí; xem license từng video. |
| **Pixabay** | https://pixabay.com/videos/ | Tìm `crowd`, `indoor`, `people walking`. |
| **Coverr** | https://coverr.co/videos | Video ngắn, nhiều cảnh trong nhà / văn phòng. |
| **Mixkit** | https://mixkit.co/free-stock-video/people/ | Category people / crowd. |

### Dataset nghiên cứu (video dài, có nhãn)

| Dataset | Mô tả ngắn | Link |
| --- | --- | --- |
| **PIROPO** | Người trong phòng, camera thường / 360° | https://www.gti.ssr.upm.es/research/gti-data/databases |
| **IndoorCrowd** | Cảnh trong nhà, nhiều người (Hugging Face) | https://huggingface.co/datasets/sebnae/IndoorCrowd |
| **MOT Challenge** | Đường phố, nhiều người đi bộ (hay dùng cho tracking) | https://motchallenge.net/ |
| **Oxford Town Centre** | Camera cố định, đông người ngoài trời | Tìm *Oxford Town Centre dataset* trên Google / GitHub. |

### Cách dùng video trong project

1. Tải file `.mp4` (hoặc chuyển sang `.mp4` nếu cần).
2. Đặt vào thư mục `videos/` (ví dụ `videos/room.mp4`).
3. Trong UI, nhập đường dẫn video hoặc bấm **Browse** để chọn file.

---
