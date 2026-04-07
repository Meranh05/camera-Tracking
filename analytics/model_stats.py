import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class ModelRunStats:
    """Thống kê 1 lần chạy model YOLO."""

    model_path: str
    inference_size: int
    conf_threshold: float
    process_every_n_frames: int
    max_track_lost_seconds: float
    started_at: float
    duration_seconds: float
    total_frames: int
    avg_fps: float
    avg_people: float
    max_people: int


class CsvModelStatsLogger:
    """Ghi thống kê từng lần chạy model ra CSV để dễ so sánh các phiên bản YOLO."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def log(self, stats: ModelRunStats, on_info: Optional[callable] = None) -> None:
        header = (
            "model_path,inference_size,conf_threshold,process_every_n_frames,"
            "max_track_lost_seconds,started_at,duration_seconds,total_frames,avg_fps,"
            "avg_people,max_people\n"
        )
        started_iso = datetime.fromtimestamp(stats.started_at).isoformat()
        line = (
            f"{os.path.basename(stats.model_path)},{stats.inference_size},"
            f"{stats.conf_threshold:.3f},{stats.process_every_n_frames},"
            f"{stats.max_track_lost_seconds:.1f},{started_iso},"
            f"{stats.duration_seconds:.3f},{stats.total_frames},"
            f"{stats.avg_fps:.2f},{stats.avg_people:.2f},{stats.max_people}\n"
        )
        file_exists = os.path.isfile(self.csv_path)
        try:
            with open(self.csv_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write(header)
                f.write(line)
            if on_info is not None:
                on_info(
                    f"Logged model run stats for {os.path.basename(stats.model_path)} "
                    f"(avg_fps={stats.avg_fps:.2f}, frames={stats.total_frames})."
                )
        except OSError:
            if on_info is not None:
                on_info("Could not write model_stats.csv.")

