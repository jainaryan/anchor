"""
Writes anonymized chat logs to logs/YYYY-MM-DD/{session_id}.jsonl
One JSON line per conversation turn.
"""

import json
import os
import re
from datetime import datetime, timezone


class ChatLogger:
    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
    _PHONE_RE = re.compile(r"\b\+?[\d][\d\s\-\(\)\.]{8,}\d\b")

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _scrub(self, text: str) -> str:
        text = self._EMAIL_RE.sub("[EMAIL]", text)
        text = self._PHONE_RE.sub("[PHONE]", text)
        return text

    def log(
        self,
        session_id: str,
        model_id: str,
        model_name: str,
        messages: list[dict],
        assistant_response: str,
    ):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_dir = os.path.join(self.log_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session": session_id,
            "model_id": model_id,
            "model_name": model_name,
            "history": [
                {"role": m["role"], "content": self._scrub(m["content"])}
                for m in messages
            ],
            "response": self._scrub(assistant_response),
        }

        path = os.path.join(day_dir, f"{session_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
