from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .tracking import Track


@dataclass
class PersonSession:
    """1 lượt khách ngồi trong quán (theo track_id)."""

    track_id: int
    time_in: float
    last_seen: float
    closed: bool = False

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.last_seen - self.time_in)


class SessionManager:
    """Quản lý session dựa trên track_id.

    - Track xuất hiện lần đầu -> mở session (time_in).
    - Track biến mất quá lost_timeout -> đóng session (time_out=last_seen).
    """

    def __init__(
        self,
        lost_timeout_seconds: float = 10.0,
        min_duration_seconds: float = 5.0,
    ):
        self.lost_timeout_seconds = lost_timeout_seconds
        self.min_duration_seconds = min_duration_seconds
        self.sessions: Dict[int, PersonSession] = {}
        self._in_roi: Dict[int, bool] = {}

    def reset(self) -> None:
        self.sessions = {}
        self._in_roi = {}

    def update(
        self,
        tracks: List[Track],
        now: float,
        in_roi_by_id: Optional[Dict[int, bool]] = None,
    ) -> List[PersonSession]:
        """Update sessions; trả về danh sách session vừa đóng (đủ min_duration)."""
        active_ids = {tr.track_id for tr in tracks}
        for tr in tracks:
            in_roi = True if in_roi_by_id is None else bool(in_roi_by_id.get(tr.track_id, False))
            self._in_roi[tr.track_id] = in_roi

            sess = self.sessions.get(tr.track_id)
            if sess is None:
                if in_roi:
                    self.sessions[tr.track_id] = PersonSession(
                        track_id=tr.track_id,
                        time_in=now,
                        last_seen=now,
                    )
            else:
                if in_roi:
                    sess.last_seen = now

        closed: List[PersonSession] = []
        for tid, sess in list(self.sessions.items()):
            if sess.closed:
                continue
            if tid in active_ids and self._in_roi.get(tid, True):
                continue
            if now - sess.last_seen >= self.lost_timeout_seconds:
                sess.closed = True
                if sess.duration_seconds >= self.min_duration_seconds:
                    closed.append(sess)
        return closed


class CsvSessionLogger:
    """Ghi session ra file CSV cục bộ."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def log(self, session: PersonSession, on_info: Optional[Callable[[str], None]] = None) -> None:
        header = "track_id,time_in,time_out,duration_seconds\n"
        line = (
            f"{session.track_id},"
            f"{datetime.fromtimestamp(session.time_in).isoformat()},"
            f"{datetime.fromtimestamp(session.last_seen).isoformat()},"
            f"{session.duration_seconds:.1f}\n"
        )
        file_exists = False
        try:
            import os

            file_exists = os.path.isfile(self.csv_path)
        except Exception:
            file_exists = False

        try:
            with open(self.csv_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write(header)
                f.write(line)
            if on_info is not None:
                on_info(
                    f"Session closed: track {session.track_id}, duration {session.duration_seconds:.1f}s (logged)."
                )
        except OSError:
            if on_info is not None:
                on_info("Could not write sessions_log.csv.")

