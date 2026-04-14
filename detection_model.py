import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import cv2
from ultralytics import YOLO

from core.sessions import CsvSessionLogger, SessionManager
from core.tracking import Track, UltralyticsTrackerAdapter
from analytics.model_stats import CsvModelStatsLogger, ModelRunStats
from analytics.storage_sqlite import SessionRow, SQLiteStore

# Module: mở video, chạy YOLO chỉ lớp người, vẽ khung, đếm, cảnh báo, tùy chọn lưu ảnh + tracking, thời gian ngồi.


@dataclass
class DetectionConfig:
    """Một bộ tham số cho một lần chạy: đường dẫn video, ngưỡng cảnh báo, lưu ảnh, tracking, kích thước hiển thị…"""

    video_path: str  # File video, URL (rtsp/http), hoặc index webcam (vd "0")
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
    track_every_n_frames: int = 1  # Chạy tracker mỗi N frame, frame giữa dùng cache để tăng FPS
    performance_mode: str = "balanced"  # fast | balanced | quality
    # Cấu hình cho tracking / session thời gian ngồi
    max_track_lost_seconds: float = 10.0  # Mất dấu quá lâu thì coi như rời quán
    min_session_duration_seconds: float = 5.0  # Chỉ log session nếu ngồi tối thiểu bấy nhiêu giây
    enable_csv_logging: bool = True  # Ghi log session ra file CSV cục bộ
    # Tracker (Ultralytics)
    tracker_type: str = "bytetrack"  # "bytetrack" hoặc "botsort"
    # ROI: nếu bật, chỉ tính khách khi tâm bbox nằm trong ROI
    use_roi: bool = False
    roi_x1: int = 0
    roi_y1: int = 0
    roi_x2: int = 0
    roi_y2: int = 0
    # SQLite persistence
    enable_sqlite_logging: bool = True
    sqlite_path: str = "cafe_analytics.sqlite3"


class PeopleCounterModel:
    """Bọc YOLOv8: phát hiện người (class 0), tracking, đếm, hiển thị bằng OpenCV và log thời gian ngồi."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        """Nạp model từ file .pt; lưu thư mục project để ghép đường dẫn output; fuse() giúp suy luận nhanh hơn một chút."""
        self.model = YOLO(model_path)
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            self.model.fuse()
        except Exception:
            pass

        # Tracker & session manager
        self.tracker: Optional[UltralyticsTrackerAdapter] = None
        self.session_manager = SessionManager()
        # Thư mục dữ liệu và output thống kê
        self.data_dir = os.path.join(self.project_dir, "data")
        self.output_dir = os.path.join(self.project_dir, "output")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Log session chi tiết coi như dữ liệu (data)
        self.sessions_log_path = os.path.join(self.data_dir, "sessions_log.csv")
        self.session_logger = CsvSessionLogger(self.sessions_log_path)
        # Thống kê model là output
        self.model_stats_logger = CsvModelStatsLogger(os.path.join(self.output_dir, "model_stats.csv"))
        self.sqlite_store: Optional[SQLiteStore] = None

    @staticmethod
    def _draw_person_box(
        frame,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        confidence: float,
        track_id: Optional[int] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        #màu khung nhận diện
        box_color = (0, 230, 90)
        text_color = (255, 255, 255)
        bg_color = (25, 25, 25)

        # Khung mỏng để giảm rối khi đông người.
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 1)
        
        # Nhãn tối giản: chỉ ID + thời gian (bỏ % để đỡ chồng chữ).
        parts = [f"ID {track_id}" if track_id is not None else "P"]
        if duration_seconds is not None:
            parts.append(f"{int(duration_seconds)}s")
        label = " | ".join(parts)

        font_scale = 0.62
        font_thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_x1 = max(0, x1)
        y_stagger = 0 if track_id is None else (track_id % 3) * 10
        label_y1 = max(0, y1 - th - 10 - y_stagger)
        label_x2 = label_x1 + tw + 10
        label_y2 = label_y1 + th + 8

        # Nếu không đủ chỗ trên đầu box thì đặt label vào trong box.
        if label_y1 <= 2:
            label_y1 = min(frame.shape[0] - th - 8, y1 + 2 + y_stagger)
            label_y2 = label_y1 + th + 8

        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), bg_color, -1)
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), box_color, 1)
        # Vẽ chữ 2 lớp (đen + trắng) cho dễ đọc trên nền nhiễu.
        cv2.putText(
            frame,
            label,
            (label_x1 + 5, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            font_thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            label,
            (label_x1 + 5, label_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def _in_rect_roi(x: int, y: int, roi: Tuple[int, int, int, int]) -> bool:
        rx1, ry1, rx2, ry2 = roi
        if rx2 <= rx1 or ry2 <= ry1:
            return False
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    @staticmethod
    def _fit_frame_for_display(frame, max_w: int = 1280, max_h: int = 720):
        """Thu nhỏ frame giữ tỷ lệ, thêm viền xám để vừa khung max_w×max_h (letterbox) trước khi imshow."""
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame

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

    @staticmethod
    def _open_capture(source: str) -> cv2.VideoCapture:
        """Mở video source: file path / URL / camera index string (vd '0')."""
        src: Union[str, int]
        s = source.strip()
        if s.isdigit():
            src = int(s)
        else:
            src = s
        return cv2.VideoCapture(src)

    def run(
        self,
        config: DetectionConfig,
        on_info: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Đọc video từng frame: chạy YOLO + tracking, vẽ box/overlay/FPS, cảnh báo, log thời gian ngồi.

        on_info: gửi thông báo (vd. status) lên giao diện.
        Thoát: hết video, ESC, hoặc đóng cửa sổ OpenCV.
        """
        # Reset tracker / sessions cho mỗi lần chạy mới
        tracker_yaml = "bytetrack.yaml" if config.tracker_type.lower() == "bytetrack" else "botsort.yaml"
        self.tracker = UltralyticsTrackerAdapter(self.model, tracker=tracker_yaml)
        self.session_manager = SessionManager(
            lost_timeout_seconds=config.max_track_lost_seconds,
            min_duration_seconds=config.min_session_duration_seconds,
        )
        self.session_manager.reset()

        # SQLite store
        if config.enable_sqlite_logging:
            data_dir = os.path.join(self.project_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            sqlite_path = config.sqlite_path.strip() or "cafe_analytics.sqlite3"
            if not os.path.isabs(sqlite_path):
                sqlite_path = os.path.join(data_dir, sqlite_path)
            self.sqlite_store = SQLiteStore(sqlite_path)
            if on_info is not None:
                on_info(f"SQLite ready: {sqlite_path}")
        else:
            self.sqlite_store = None

        # Nếu là đường dẫn file thì check tồn tại; còn URL/camera index thì bỏ qua check này.
        src_str = config.video_path.strip()
        looks_like_file = not (src_str.isdigit() or "://" in src_str)
        if looks_like_file and not os.path.isfile(src_str):
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
        if config.track_every_n_frames <= 0:
            raise ValueError("track_every_n_frames must be > 0.")
        if config.performance_mode not in {"fast", "balanced", "quality"}:
            raise ValueError("performance_mode must be one of: fast, balanced, quality.")
        output_dir = config.output_dir.strip() or "screenshots"
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.project_dir, output_dir)

        if config.save_warning_shot:
            os.makedirs(output_dir, exist_ok=True)
            if on_info is not None:
                on_info(f"Screenshot folder ready: {output_dir}")

        # Preset hiệu năng để dễ cân bằng FPS và chất lượng.
        effective_imgsz = config.inference_size
        effective_conf = config.conf_threshold
        effective_track_every = config.track_every_n_frames
        if config.performance_mode == "fast":
            effective_imgsz = min(effective_imgsz, 416)
            effective_conf = max(effective_conf, 0.40)
            effective_track_every = max(effective_track_every, 2)
        elif config.performance_mode == "quality":
            effective_imgsz = max(effective_imgsz, 640)
            effective_conf = min(effective_conf, 0.30)
            effective_track_every = 1

        cap = self._open_capture(config.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {config.video_path}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        window_name = "People Counting - YOLOv8 + OpenCV"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.display_width, config.display_height)
        start_time = time.time()
        prev_time = start_time
        smooth_fps = 0.0
        last_screenshot_time = start_time
        window_opened = False
        frame_index = 0
        cached_boxes: List[Tuple[int, int, int, int, float]] = []
        total_frames = 0
        people_sum = 0
        people_max = 0
        roi = (config.roi_x1, config.roi_y1, config.roi_x2, config.roi_y2)
        cached_tracks: List[Track] = []
        cached_conf_by_id = {}

        while True:
            for _ in range(config.max_grab_frames):
                if not cap.grab():
                    break

            ret, frame = cap.read()
            if not ret:
                break

            # Với ByteTrack/BoT-SORT: nên track liên tục mỗi frame để ID ổn định.
            # Nếu muốn tăng tốc, có thể tăng process_every_n_frames và tự nội suy, nhưng sẽ giảm ổn định ID.
            tracks: List[Track] = []
            conf_by_id = {}
            run_tracking = (frame_index % effective_track_every == 0)
            if run_tracking and self.tracker is not None:
                tracks, conf_by_id = self.tracker.update(
                    frame=frame,
                    now=time.time(),
                    imgsz=effective_imgsz,
                    conf=effective_conf,
                    classes=[0],
                )
                cached_tracks = tracks
                cached_conf_by_id = conf_by_id
            else:
                tracks, conf_by_id = cached_tracks, cached_conf_by_id

            now = time.time()

            # Count trong ROI (nếu bật) và cập nhật session theo ROI.
            in_roi_by_id = {}
            roi_count = 0
            if config.use_roi:
                for tr in tracks:
                    cx, cy = self._bbox_center(tr.bbox)
                    inside = self._in_rect_roi(cx, cy, roi)
                    in_roi_by_id[tr.track_id] = inside
                    if inside:
                        roi_count += 1
            else:
                for tr in tracks:
                    in_roi_by_id[tr.track_id] = True
                roi_count = len(tracks)

            closed_sessions = self.session_manager.update(tracks, now, in_roi_by_id=in_roi_by_id)
            if config.enable_csv_logging:
                for sess in closed_sessions:
                    self.session_logger.log(sess, on_info=on_info)
            if self.sqlite_store is not None:
                for sess in closed_sessions:
                    self.sqlite_store.insert_session(
                        SessionRow(
                            track_id=sess.track_id,
                            time_in=sess.time_in,
                            time_out=sess.last_seen,
                            duration_seconds=sess.duration_seconds,
                            model_path=config.model_path,
                            source=config.video_path,
                        )
                    )

            person_count = roi_count
            total_frames += 1
            people_sum += person_count
            people_max = max(people_max, person_count)

            for tr in tracks:
                x1, y1, x2, y2 = tr.bbox
                sess = self.session_manager.sessions.get(tr.track_id)
                duration = sess.duration_seconds if sess is not None else None
                det_conf = float(conf_by_id.get(tr.track_id, 0.0))

                if in_roi_by_id.get(tr.track_id, True):
                    self._draw_person_box(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        det_conf,
                        track_id=tr.track_id,
                        duration_seconds=duration,
                    )
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 120), 2)
                    cv2.putText(
                        frame,
                        f"ID {tr.track_id} (out ROI)",
                        (max(0, x1), max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (120, 120, 120),
                        2,
                        cv2.LINE_AA,
                    )

            fps = 1.0 / max(now - prev_time, 1e-6)
            smooth_fps = fps if smooth_fps == 0.0 else (smooth_fps * 0.85 + fps * 0.15)
            prev_time = now

            # Khối thông tin góc trái với nền đậm để dễ đọc trên mọi background.
            def _panel_text(y: int, text: str, color: Tuple[int, int, int]) -> None:
                cv2.putText(
                    frame,
                    text,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.88,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    text,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.88,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            _panel_text(42, f"People Count: {person_count}", (255, 255, 255))
            _panel_text(80, f"Threshold: {config.threshold}", (255, 255, 0))
            _panel_text(118, f"FPS: {smooth_fps:.2f}", (0, 255, 255))

            if config.use_roi and (roi[2] > roi[0]) and (roi[3] > roi[1]):
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    "ROI (Cafe Area)",
                    (roi[0] + 6, roi[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
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

            if config.save_warning_shot and now - last_screenshot_time >= config.cooldown_seconds:
                screenshot_path = self._save_warning_screenshot(frame, output_dir)
                last_screenshot_time = now
                if on_info is not None:
                    on_info(f"Saved screenshot: {screenshot_path}")

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
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        # Thống kê theo model để so sánh các phiên bản YOLO / cấu hình khác nhau.
        total_time = max(time.time() - start_time, 1e-6)
        if total_frames > 0:
            avg_fps_overall = total_frames / total_time
            avg_people = people_sum / total_frames
            stats = ModelRunStats(
                model_path=config.model_path,
                inference_size=effective_imgsz,
                conf_threshold=effective_conf,
                process_every_n_frames=effective_track_every,
                max_track_lost_seconds=config.max_track_lost_seconds,
                started_at=start_time,
                duration_seconds=total_time,
                total_frames=total_frames,
                avg_fps=avg_fps_overall,
                avg_people=avg_people,
                max_people=people_max,
            )
            self.model_stats_logger.log(stats, on_info=on_info)
