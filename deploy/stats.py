"""
Derives live stats from log files on disk. No database needed.
"""

import json
import os
from collections import Counter
from datetime import datetime, timezone, timedelta


def get_stats(log_dir: str) -> dict:
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")
    active_cutoff = now - timedelta(minutes=5)

    total_sessions = set()
    today_sessions = set()
    active_sessions = set()
    total_messages = 0
    today_messages = 0
    model_counts: Counter = Counter()

    if not os.path.isdir(log_dir):
        return _empty()

    for date_dir in os.listdir(log_dir):
        day_path = os.path.join(log_dir, date_dir)
        if not os.path.isdir(day_path):
            continue

        is_today = date_dir == today_str

        for fname in os.listdir(day_path):
            if not fname.endswith(".jsonl"):
                continue

            session_id = fname[:-6]
            total_sessions.add(session_id)
            if is_today:
                today_sessions.add(session_id)

            fpath = os.path.join(day_path, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        total_messages += 1
                        if is_today:
                            today_messages += 1

                        model_counts[entry.get("model_id", "unknown")] += 1

                        # Active = last message within 5 minutes
                        ts_str = entry.get("ts", "")
                        try:
                            ts = datetime.fromisoformat(ts_str)
                            if ts >= active_cutoff:
                                active_sessions.add(session_id)
                        except ValueError:
                            pass

            except OSError:
                continue

    top_model = model_counts.most_common(1)[0][0] if model_counts else None

    return {
        "total_sessions": len(total_sessions),
        "today_sessions": len(today_sessions),
        "active_sessions": len(active_sessions),   # active in last 5 min
        "total_messages": total_messages,
        "today_messages": today_messages,
        "top_model": top_model,
        "model_breakdown": dict(model_counts),
        "as_of": now.isoformat(),
    }


def _empty() -> dict:
    return {
        "total_sessions": 0,
        "today_sessions": 0,
        "active_sessions": 0,
        "total_messages": 0,
        "today_messages": 0,
        "top_model": None,
        "model_breakdown": {},
        "as_of": datetime.now(timezone.utc).isoformat(),
    }
