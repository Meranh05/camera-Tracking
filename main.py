import os
import sys
import traceback
import customtkinter as ctk
from tkinter import filedialog

from detection_model import DetectionConfig, PeopleCounterModel

# giao diện (CustomTkinter). Logic đọc video + YOLO nằm trong detection_model.

ABOUT_TEXT = """People Counting in a Room
Giới Thiệu:
 Chương trình đếm số người trong phòng từ video, hiển thị theo thời gian thực và cảnh báo khi vượt ngưỡng.
 Dùng YOLOv8 + OpenCV + CustomTkinter

Hướng Dẫn:
1. CHỨC NĂNG CHÍNH
- Đọc video, phát hiện chỉ lớp NGƯỜI (person) bằng YOLOv8.
- Vẽ khung (bounding box) và hiển thị độ tin cậy.
- Đếm số người ở mỗi frame, hiện People Count / Threshold / FPS.
- Cảnh báo khi số người VƯỢT ngưỡng: WARNING: Too many people!
- Tùy chọn lưu ảnh màn hình ĐỊNH KỲ theo số giây bạn nhập.
- Thoát video: ESC hoặc đóng cửa sổ preview (nút X).

2. Các chức năng trong UI
- Video file — Đường dẫn file video. Có thể Browse để chọn.
- YOLO model — File trọng số (vd: yolov8n.pt). Lần đầu có thể tải tự động.
- Threshold — Số người tối đa cho phép; nếu Count > Threshold thì cảnh báo.
- Screenshot interval (s) — Cứ sau N giây lưu 1 ảnh (khi bật lưu ảnh).
- Screenshot folder — Thư mục con trong project lưu ảnh (vd: screenshots).
- Save screenshot… — Bật để lưu ảnh theo chu kỳ; tắt thì không lưu.

Start  — Chạy xử lý (mở cửa sổ video OpenCV).
"""

class PeopleCountingApp(ctk.CTk):
    """Cửa sổ chính: nhập cấu hình (video, model, ngưỡng…) và bấm chạy đếm người."""

    def __init__(self):
        # Khởi tạo: tiêu đề cửa sổ, theme, các biến gắn với ô nhập; rồi vẽ form và căn kích thước.
        super().__init__()
        self.withdraw()
        self.title("People Counting - YOLOv8")
        # Khởi tạo chiều cao
        self.geometry("960x200")
        self.minsize(760, 360)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")  

        self.video_path_var = ctk.StringVar(value="videos/room.mp4")
        self.model_path_var = ctk.StringVar(value="yolov8n.pt")
        self.threshold_var = ctk.StringVar(value="5") #Số người tối đa cho phép; nếu Count > Threshold thì cảnh báo.
        self.cooldown_var = ctk.StringVar(value="5") #Cứ sau N giây lưu 1 ảnh (khi bật lưu ảnh).
        self.output_dir_var = ctk.StringVar(value="screenshots") #Thư mục lưu ảnh (tương đối thư mục project)
        self.save_shot_var = ctk.BooleanVar(value=False) #Bật tắt để lưu ảnh màn hình theo chu kỳ

        self._build_ui()  # Tạo toàn bộ nút và ô nhập
        self._fit_window_to_content()  # Cho cửa sổ vừa nội dung (tránh cắt chữ)
        self.deiconify()

    def _build_ui(self) -> None:
        """Dựng giao diện: tiêu đề, khung nhập liệu, nút Start / About / Exit, dòng trạng thái."""
        self.grid_columnconfigure(0, weight=1)
        # Không cho hàng form giãn theo chiều dọc — tránh khoảng trống thừa trong khung.

        header = ctk.CTkLabel(
            self,
            text="People Counting in a Room", 
            font=ctk.CTkFont(size=28, weight="bold"),
        )
        header.grid(row=0, column=0, padx=24, pady=(16, 6), sticky="nw")

        self._card = ctk.CTkFrame(self, corner_radius=14)
        card = self._card
        card.grid(row=1, column=0, padx=24, pady=(4, 6), sticky="ew")
        card.grid_columnconfigure(1, weight=1)

        btn_font_bold = ctk.CTkFont(size=13, weight="bold")

        ctk.CTkLabel(card, text="Video file").grid(row=0, column=0, padx=14, pady=12, sticky="w")
        ctk.CTkEntry(card, textvariable=self.video_path_var).grid(row=0, column=1, padx=10, pady=12, sticky="ew")
        ctk.CTkButton(
            card,
            text="Browse",
            width=100,
            font=btn_font_bold,
            fg_color="#6366f1",
            hover_color="#4f46e5",
            command=self._pick_video,
        ).grid(row=0, column=2, padx=(0, 14), pady=12)

        ctk.CTkLabel(card, text="YOLO model").grid(row=1, column=0, padx=14, pady=12, sticky="w")
        ctk.CTkEntry(card, textvariable=self.model_path_var).grid(row=1, column=1, padx=10, pady=12, sticky="ew")

        ctk.CTkLabel(card, text="Threshold").grid(row=2, column=0, padx=14, pady=12, sticky="w")
        ctk.CTkEntry(card, textvariable=self.threshold_var, width=140).grid(row=2, column=1, padx=10, pady=12, sticky="w")

        ctk.CTkLabel(card, text="Screenshot interval (s)").grid(row=3, column=0, padx=14, pady=12, sticky="w")
        ctk.CTkEntry(card, textvariable=self.cooldown_var, width=140).grid(row=3, column=1, padx=10, pady=12, sticky="w")

        ctk.CTkLabel(card, text="Screenshot folder").grid(row=4, column=0, padx=14, pady=12, sticky="w")
        ctk.CTkEntry(card, textvariable=self.output_dir_var).grid(row=4, column=1, padx=10, pady=12, sticky="ew")

        ctk.CTkSwitch(
            card,
            text="Save screenshot continuously by interval",
            variable=self.save_shot_var,
            onvalue=True,
            offvalue=False,
        ).grid(row=5, column=1, padx=10, pady=(4, 10), sticky="w")

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, padx=24, pady=(6, 6), sticky="ew")

        primary_fg = "#2563eb"
        primary_hover = "#1d4ed8"

        ctk.CTkButton(
            btn_frame,
            text="Start",
            width=170,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=primary_fg,
            hover_color=primary_hover,
            command=self._start,
        ).pack(side="left")
        ctk.CTkButton(
            btn_frame,
            text="About",
            width=100,
            font=btn_font_bold,
            fg_color=primary_fg,
            hover_color=primary_hover,
            command=self._show_about,
        ).pack(side="left", padx=(10, 0))
        ctk.CTkButton(
            btn_frame,
            text="Exit",
            width=120,
            font=btn_font_bold,
            fg_color=primary_fg,
            hover_color=primary_hover,
            command=self.destroy,
        ).pack(side="left", padx=10)

        self.status_label = ctk.CTkLabel(
            self,
            text="Status: Ready",
            font=ctk.CTkFont(size=13),
            text_color="gray70",
        )
        self.status_label.grid(row=3, column=0, padx=24, pady=(0, 10), sticky="w")

    #Cửa sổ phụ chứa hướng dẫn 
    def _show_about(self) -> None:
        """Hiện cửa sổ phụ chứa hướng dẫn (nội dung ABOUT_TEXT) và nút đóng."""
        win = ctk.CTkToplevel(self)
        win.title("About — Hướng dẫn")
        win.geometry("640x520")
        win.minsize(480, 360)
        win.transient(self)
        win.grab_set()

        title = ctk.CTkLabel(
            win,
            text="Giới thiệu & Hướng dẫn",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(padx=16, pady=(16, 8), anchor="w")

        textbox = ctk.CTkTextbox(win, font=ctk.CTkFont(size=13), wrap="word")
        textbox.pack(fill="both", expand=True, padx=16, pady=8)
        textbox.insert("1.0", getattr(sys.modules[__name__], "ABOUT_TEXT", ""))
        textbox.configure(state="disabled")

        btn_row = ctk.CTkFrame(win, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkButton(btn_row, text="Đóng", width=120, command=win.destroy).pack(side="right")

    #Xử lý căn chỉnh cửa sổ cho phù hợp với nội dung
    def _fit_window_to_content(self) -> None:
        """Căn cửa sổ sát nội dung."""
        self.update_idletasks()
        self.update()

        pad_x = 24
        pad_bottom = 14
        min_w, min_h = 760, 360

        needed_w = max(self._card.winfo_reqwidth() + pad_x * 2 + 8, min_w)
        bottom = self.status_label.winfo_y() + self.status_label.winfo_height()
        needed_h = max(bottom + pad_bottom, min_h)

        self.geometry(f"{needed_w}x{needed_h}")

    #Xử lý chọn file video
    def _pick_video(self) -> None:
        """Mở hộp thoại chọn file video; nếu chọn thì gán đường dẫn vào ô Video file."""
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.video_path_var.set(path)

    #Xử lý cập nhật dòng 'Status: …' dưới cùng (cả khi detection_model gọi callback).
    def _set_status(self, text: str) -> None:
        """Cập nhật dòng 'Status: …' dưới cùng (cả khi detection_model gọi callback)."""
        self.status_label.configure(text=f"Status: {text}")
        self.update_idletasks()


    def _start(self) -> None:
        """Đọc giá trị từ form → kiểm tra → tải YOLO → chạy video và nhận diện (cửa OpenCV)."""
        try:
            video_path = self.video_path_var.get().strip()
            model_path = self.model_path_var.get().strip()
            threshold = int(self.threshold_var.get().strip())
            cooldown = float(self.cooldown_var.get().strip())
            output_dir = self.output_dir_var.get().strip() or "screenshots"
            save_shot = bool(self.save_shot_var.get())

            if not video_path:
                raise ValueError("Please choose a video file.")
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if threshold < 0:
                raise ValueError("Threshold must be >= 0.")
            if cooldown < 0:
                raise ValueError("Cooldown seconds must be >= 0.")

            self._set_status("Loading model...")
            model = PeopleCounterModel(model_path=model_path)  # Tải file .pt (YOLO)

            config = DetectionConfig(  # Gói mọi tham số cho một lần chạy video
                video_path=video_path,
                model_path=model_path,
                threshold=threshold,
                save_warning_shot=save_shot,
                output_dir=output_dir,
                cooldown_seconds=cooldown,
            )

            self._set_status("Running detection (press ESC in video window to stop)...")
            model.run(config, on_info=lambda msg: self._set_status(msg))  # Vòng lặp frame; ESC hoặc đóng cửa = dừng
            self._set_status("Finished.")
        except Exception as exc:
            self._set_status(f"Error: {exc}")
            print("[ERROR]", exc)
            print(traceback.format_exc())


def main() -> None:
    """Tạo app và chạy mainloop Tk."""
    app = PeopleCountingApp()
    app.mainloop()


if __name__ == "__main__":
    main()
