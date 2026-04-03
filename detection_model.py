import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
from ultralytics import YOLO

# Module: mở video, chạy YOLO chỉ lớp người, vẽ khung, đếm, cảnh báo, tùy chọn lưu ảnh.


@dataclass
class DetectionConfig:
    """Một bộ tham số cho một lần chạy: đường dẫn video, ngưỡng cảnh báo, lưu ảnh, kích thước hiển thị…"""

    video_path: str  # Đường dẫn file video đầu vào
    model_path: str = "yolov8n.pt"  # File trọng số YOLO
    threshold: int = 5  # Số người tối đa cho phép; vượt thì hiện cảnh báo đỏ
    save_warning_shot: bool = False  # Bật thì lưu ảnh màn hình theo chu kỳ
    output_dir: str = "screenshots"  # Thư mục lưu ảnh (tương đối thư mục project)
    cooldown_seconds: float = 2.0  # Khoảng cách tối thiểu giữa hai lần lưu ảnh (giây)
    inference_size: int = 512  # Kích thước ảnh đưa vào YOLO (imgsz)
    conf_threshold: float = 0.35  # Ngưỡng độ tin cậy tối thiểu để giữ detection
    display_width: int = 1280  # Kích thước cửa sổ hiển thị (rộng)
    display_height: int = 720  # Kích thước cửa sổ hiển thị (cao)
    process_every_n_frames: int = 3  # Chỉ chạy suy luận mỗi N frame
    max_grab_frames: int = 1  # Số frame grab thêm trước read (giảm trễ)


class PeopleCounterModel:
    """Bọc YOLOv8: phát hiện người (class 0), đếm số box mỗi frame, hiển thị bằng OpenCV."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """Nạp model từ file .pt; lưu thư mục project để ghép đường dẫn output; fuse() giúp suy luận nhanh hơn một chút."""
        self.model = YOLO(model_path)
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        # Gộp Conv+BN (nếu được) để tăng tốc suy luận nhẹ.
        try:
            self.model.fuse()
        except Exception:
            pass

    @staticmethod
    def _draw_person_box(frame, x1: int, y1: int, x2: int, y2: int, confidence: float) -> None:
        """Vẽ khung và chữ 'PERSON | % tin cậy' lên frame tại (x1,y1)-(x2,y2) — dễ nhìn trên video."""
        green_outer = (0, 255, 60)
        green_inner = (0, 170, 0)

        # Double-stroke rectangle for a stronger highlighted effect.
        cv2.rectangle(frame, (x1, y1), (x2, y2), green_outer, 4)
        cv2.rectangle(frame, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), green_inner, 2)

        label = f"PERSON | {confidence * 100:.1f}%"
        font_scale = 0.72
        font_thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_x1 = max(0, x1)
        label_y1 = max(0, y1 - th - 20)
        label_x2 = label_x1 + tw + 18
        label_y2 = label_y1 + th + 14

        # Solid background + border to make text pop in all lighting conditions.
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), green_outer, -1)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (20, 70, 20), 2)
        cv2.putText(
            frame,
            label,
            (label_x1 + 9, label_y2 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _fit_frame_for_display(frame, max_w: int = 1280, max_h: int = 720):
        """Thu nhỏ frame giữ tỷ lệ, thêm viền xám để vừa khung max_w×max_h (letterbox) trước khi imshow."""
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

        # Fast path: frame already fits target display.
        if w <= max_w and h <= max_h:
            return frame

        scale = min(max_w / w, max_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        pad_x = max_w - new_w
        pad_y = max_h - new_h
        left = pad_x // 2
        right = pad_x - left
        top = pad_y // 2
        bottom = pad_y - top
        return cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

    @staticmethod
    def _save_warning_screenshot(frame, output_dir: str) -> str:
        """Lưu ảnh frame hiện tại thành JPG trong output_dir; tên file có timestamp; trả về đường dẫn file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(output_dir, f"output_{timestamp}.jpg")
        ok = cv2.imwrite(file_path, frame)
        if not ok:
            raise RuntimeError(f"Could not save screenshot to: {file_path}")
        return file_path

    def run(
        self,
        config: DetectionConfig,
        on_info: Optional[callable] = None,
    ) -> None:
        """Đọc video từng frame: chạy YOLO mỗi N frame, vẽ box/overlay/FPS, cảnh báo khi vượt ngưỡng,
        tùy chọn lưu ảnh theo chu kỳ. on_info: gửi thông báo (vd. status) lên giao diện.
        Thoát: hết video, ESC, hoặc đóng cửa sổ OpenCV."""
        if not os.path.isfile(config.video_path):
            raise FileNotFoundError(f"Video file not found: {config.video_path}")
        if config.threshold < 0:
            raise ValueError("Threshold must be >= 0.")
        if config.cooldown_seconds < 0:
            raise ValueError("Cooldown seconds must be >= 0.")
        if config.inference_size <= 0:
            raise ValueError("Inference size must be > 0.")
        if config.display_width <= 0 or config.display_height <= 0:
            raise ValueError("Display width/height must be > 0.")
        if not (0.0 <= config.conf_threshold <= 1.0):
            raise ValueError("Confidence threshold must be in range [0, 1].")
        if config.process_every_n_frames <= 0:
            raise ValueError("process_every_n_frames must be > 0.")
        if config.max_grab_frames < 0:
            raise ValueError("max_grab_frames must be >= 0.")
        output_dir = config.output_dir.strip() or "screenshots"
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_dir, output_dir)

        if config.save_warning_shot:
            os.makedirs(output_dir, exist_ok=True)
            if on_info is not None:
                on_info(f"Screenshot folder ready: {output_dir}")

        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.video_path}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Giảm độ trễ buffer; thử MJPG để giảm tải giải mã (tùy backend).
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        window_name = "People Counting - YOLOv8 + OpenCV"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.display_width, config.display_height)
        prev_time = time.time()
        smooth_fps = 0.0
        last_screenshot_time = time.time()
        window_opened = False
        frame_index = 0
        cached_boxes = []
        cached_person_count = 0

        while True:
            # Bỏ qua frame thừa (nếu cấu hình) để stream gần thời gian thực, hạn chế tích trễ.
            for _ in range(config.max_grab_frames):
                if not cap.grab():
                    break

            ret, frame = cap.read()
            if not ret:
                break

            # Chỉ mỗi N frame mới chạy YOLO (các frame giữa dùng lại kết quả cũ) — nhanh hơn, giảm tải GPU/CPU.
            run_inference = (frame_index % config.process_every_n_frames == 0)
            if run_inference:
                results = self.model(
                    frame,
                    verbose=False,
                    classes=[0],
                    imgsz=config.inference_size,
                    conf=config.conf_threshold,
                )
                result = results[0]
                cached_boxes = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0].item())
                        cached_boxes.append((x1, y1, x2, y2, confidence))
                cached_person_count = len(cached_boxes)

            person_count = cached_person_count
            for x1, y1, x2, y2, confidence in cached_boxes:
                self._draw_person_box(frame, x1, y1, x2, y2, confidence)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            smooth_fps = fps if smooth_fps == 0.0 else (smooth_fps * 0.85 + fps * 0.15)
            prev_time = now

            cv2.putText(
                frame,
                f"People Count: {person_count}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Threshold: {config.threshold}",
                (12, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FPS: {smooth_fps:.2f}",
                (12, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if person_count > config.threshold:
                warning_text = "WARNING: Too many people!"
                warning_scale = 1.05
                warning_thickness = 3
                (tw, th), _ = cv2.getTextSize(
                    warning_text, cv2.FONT_HERSHEY_SIMPLEX, warning_scale, warning_thickness
                )
                pad_x, pad_y = 14, 10
                box_x2 = frame.shape[1] - 14
                box_x1 = max(10, box_x2 - tw - (pad_x * 2))
                box_y1 = 14
                box_y2 = box_y1 + th + (pad_y * 2)

                # Red warning badge at top-right for better visibility.
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 255), -1)
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), 2)
                cv2.putText(
                    frame,
                    warning_text,
                    (box_x1 + pad_x, box_y2 - pad_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    warning_scale,
                    (255, 255, 255),
                    warning_thickness,
                    cv2.LINE_AA,
                )

            # Save screenshot continuously every N seconds when enabled.
            if config.save_warning_shot and now - last_screenshot_time >= config.cooldown_seconds:
                screenshot_path = self._save_warning_screenshot(frame, output_dir)
                last_screenshot_time = now
                if on_info is not None:
                    on_info(f"Saved screenshot: {screenshot_path}")

            # If user closed window previously, stop before drawing again.
            if window_opened:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break

            display_frame = self._fit_frame_for_display(
                frame,
                max_w=config.display_width,
                max_h=config.display_height,
            )
            try:
                cv2.imshow(window_name, display_frame)
                window_opened = True
            except cv2.error:
                # Window may already be destroyed by user action.
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

            # Some backends report close event after waitKey.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
