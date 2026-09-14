import json
import time
from datetime import datetime, timezone
from pathlib import Path


TRACES_DIR = Path("data/traces")
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class AgentTrace:
    def __init__(self, agent_name, run_id):
        self.agent_name = agent_name
        self.run_id = run_id
        self._start_time = time.time()
        self.trace = {
            "agent": agent_name,
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "input": {},
            "plan": "",
            "tool_calls": [],
            "reasoning_steps": [],
            "output": {},
            "duration_seconds": None,
            "status": "running",
        }

    def set_input(self, **kwargs):
        self.trace["input"] = kwargs
        return self

    def set_plan(self, plan_str):
        self.trace["plan"] = plan_str
        return self

    def set_output(self, **kwargs):
        self.trace["output"] = kwargs
        return self

    def extract_from_messages(self, messages):
        for message in messages:
            message_type = message.__class__.__name__
            content = getattr(message, "content", "")

            if message_type == "HumanMessage":
                self.trace["reasoning_steps"].append(
                    {"type": "task_input", "content": content}
                )
            elif message_type == "AIMessage":
                tool_calls = getattr(message, "tool_calls", []) or []
                self.trace["tool_calls"].extend(tool_calls)
                self.trace["reasoning_steps"].append(
                    {"type": "ai_reasoning", "content": content}
                )
                if not self.trace["plan"] and isinstance(content, str) and content.strip():
                    self.trace["plan"] = content
            elif message_type == "ToolMessage":
                self.trace["reasoning_steps"].append(
                    {"type": "tool_result", "content": content}
                )

        return self

    def complete(self, status="success"):
        self.trace["duration_seconds"] = time.time() - self._start_time
        self.trace["status"] = status
        trace_path = TRACES_DIR / f"trace_{self.agent_name}_{self.run_id[:8]}.json"
        with trace_path.open("w", encoding="utf-8") as file:
            json.dump(self.trace, file, indent=2, default=str)

        print(
            f"Agent trace complete: {self.agent_name} "
            f"({status}, {self.trace['duration_seconds']:.2f}s)"
        )
        return self

    def fail(self, error):
        self.trace["error"] = str(error)
        return self.complete(status="failed")
