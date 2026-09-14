import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id=None):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_file = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent, action, **kwargs):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }

        with self.log_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, default=str) + "\n")

    def get_logs(self):
        if not self.log_file.exists():
            return []

        with self.log_file.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]
