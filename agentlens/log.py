import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import petname
from pydantic import BaseModel

current_run_dir: ContextVar[Path] = ContextVar("current_run_dir")


class RunLog(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None


class RunHistory:
    def __init__(self, runs_dir: Path):
        self._runs_dir = runs_dir
        self.runs = self.load()

    def file_path(self) -> Path:
        path = self._runs_dir / "run_history.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def load(self) -> List[RunLog]:
        path = self.file_path()
        if not path.exists():
            return []
        with open(path) as f:
            return [RunLog(**entry) for entry in json.load(f)]

    def save(self):
        with open(self.file_path(), "w") as f:
            json.dump([run.model_dump() for run in self.runs], f, indent=2)

    def create_id(self) -> str:
        counter = 0
        while counter < 100:
            id = petname.generate(words=3, separator="_")
            if id not in {run.run_id for run in self.runs}:
                return id
            counter += 1
        raise Exception("Failed to create a unique run id")

    def add_run(self) -> RunLog:
        run = RunLog(
            run_id=self.create_id(),
            status="started",
            start_time=datetime.now().isoformat(),
        )
        self.runs.append(run)
        self.save()
        return run

    def complete_run(self, log: RunLog):
        if log.status != "failed":
            log.status = "completed"
        log.end_time = datetime.now().isoformat()
        self.save()


@contextmanager
def create_run_log(runs_dir: Path):
    history = RunHistory(runs_dir)
    log = history.add_run()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{timestamp}_{log.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    token = current_run_dir.set(run_dir)

    print(f"Starting run {log.run_id}")
    try:
        yield log
    except Exception:
        log.status = "failed"
        print(f"Run {log.run_id} failed")
        raise
    finally:
        history.complete_run(log)
        current_run_dir.reset(token)
        if log.status != "failed":
            print(f"Run {log.run_id} completed")


def get_run_dir() -> Path:
    try:
        return current_run_dir.get()
    except LookupError:
        raise RuntimeError("No active run context - this can only be called during a run")
