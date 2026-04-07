from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2


@dataclass
class Track:
    """1 track đại diện cho 1 người đang được theo dõi trong camera."""

    track_id: int
    bbox: Tuple[int, int, int, int]
    first_seen: float
    last_seen: float


class SimpleIOUTracker:
    """Tracker đơn giản dựa trên IoU.

    - Nhận list bbox người mỗi frame.
    - Ghép với track cũ bằng IoU cao nhất.
    - Không đủ tốt -> tạo track mới.
    """

    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def update(self, detections: List[Tuple[int, int, int, int]], now: float) -> List[Track]:
        """Cập nhật danh sách track từ các bbox detect mới."""
        updated_tracks: Dict[int, Track] = {}

        for det_box in detections:
            best_iou = 0.0
            best_id: Optional[int] = None
            for tid, track in self.tracks.items():
                score = self._iou(det_box, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_threshold:
                old = self.tracks[best_id]
                updated_tracks[best_id] = Track(
                    track_id=best_id,
                    bbox=det_box,
                    first_seen=old.first_seen,
                    last_seen=now,
                )
            else:
                tid = self._next_id
                self._next_id += 1
                updated_tracks[tid] = Track(
                    track_id=tid,
                    bbox=det_box,
                    first_seen=now,
                    last_seen=now,
                )

        self.tracks = updated_tracks
        return list(self.tracks.values())


class UltralyticsTrackerAdapter:
    """Adapter dùng tracker có sẵn của Ultralytics (ByteTrack/BoT-SORT) để lấy track_id ổn định.

    Ultralytics sẽ quản lý trạng thái tracker khi gọi `model.track(..., persist=True)` lặp theo frame.
    """

    def __init__(self, yolo_model, tracker: str = "bytetrack.yaml"):
        self.model = yolo_model
        self.tracker = tracker

    def reset(self) -> None:
        """Reset tracker state (best-effort)."""
        return

    def update(
        self,
        frame,
        now: float,
        imgsz: int,
        conf: float,
        classes: List[int],
    ) -> Tuple[List[Track], Dict[int, float]]:
        """Chạy track trên 1 frame và trả về Track list + map track_id->confidence."""
        results = self.model.track(
            source=frame,
            persist=True,
            verbose=False,
            tracker=self.tracker,
            imgsz=imgsz,
            conf=conf,
            classes=classes,
        )
        result = results[0]

        tracks: List[Track] = []
        conf_by_id: Dict[int, float] = {}

        if result.boxes is None:
            return tracks, conf_by_id

        boxes = result.boxes
        ids = getattr(boxes, "id", None)
        if ids is None:
            return tracks, conf_by_id

        for i in range(len(boxes)):
            tid = int(ids[i].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
            c = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            tracks.append(Track(track_id=tid, bbox=(x1, y1, x2, y2), first_seen=now, last_seen=now))
            conf_by_id[tid] = c

        return tracks, conf_by_id

