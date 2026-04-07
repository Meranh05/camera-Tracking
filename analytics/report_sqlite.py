import argparse
import csv
import os

from .storage_sqlite import SQLiteStore


def main() -> None:
    p = argparse.ArgumentParser(description="Cafe analytics report from SQLite.")
    p.add_argument("--db", default="data/cafe_analytics.sqlite3", help="Path to sqlite DB.")
    p.add_argument("--mode", choices=["day", "hour"], default="day", help="Report mode.")
    p.add_argument("--day", default=None, help="Filter YYYY-MM-DD for hour mode.")
    p.add_argument("--out", default=None, help="Optional CSV output path.")
    args = p.parse_args()

    db_path = args.db
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", db_path)

    store = SQLiteStore(db_path)
    if args.mode == "day":
        rows = store.report_by_day()
        header = ["day", "total_sessions", "avg_duration_seconds"]
    else:
        rows = store.report_by_hour(day=args.day)
        header = ["hour", "total_sessions", "avg_duration_seconds"]

    if args.out:
        out_path = args.out
        if not os.path.isabs(out_path):
            reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            out_path = os.path.join(reports_dir, out_path)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"Wrote report to {out_path}")
    else:
        print(",".join(header))
        for r in rows:
            print(",".join(str(x) for x in r))


if __name__ == "__main__":
    main()

