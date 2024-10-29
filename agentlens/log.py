import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from agentlens.utils import create_readable_id


class RunLog(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None


@contextmanager
def create_run_log(run_dir: Path):
    log = RunLogs(run_dir)
    run = log.start_run()
    try:
        yield run
    except Exception:
        run.status = "failed"
        raise
    finally:
        log.finish_run(run)


class RunLogs:
    def __init__(self, run_dir: Path):
        self._run_dir = run_dir
        self.runs = self.load()

    def file_path(self) -> Path:
        path = self._run_dir / "run_log.json"
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
        while True and counter < 100:
            id = create_readable_id()
            if id not in {run.run_id for run in self.runs}:
                return id
            counter += 1  # just in case
        raise Exception("Failed to create a unique run id")

    def start_run(self) -> RunLog:
        run = RunLog(
            run_id=self.create_id(),
            status="started",
            start_time=datetime.now().isoformat(),
        )
        self.runs.append(run)
        self.save()
        return run

    def finish_run(self, run: RunLog):
        if run.status != "failed":  # Only update status if not already failed
            run.status = "completed"
        run.end_time = datetime.now().isoformat()
        self.save()
