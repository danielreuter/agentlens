import json
from datetime import datetime

import pytest

from agentlens import AI


def test_run_creates_log_entries(ai: AI, simple_dataset, simple_dataset_tasks):
    # Run the task
    results = ai.run(
        main=lambda row: simple_dataset_tasks["simple"](row.x),
        dataset=simple_dataset,
    )

    # Verify results
    assert results == [2, 4, 6]

    # Check run log
    run_log = ai._run_log
    assert len(run_log.runs) == 1

    run = run_log.runs[0]
    assert run.status == "completed"
    assert run.id is not None
    assert run.end_time is not None
    assert run.start_time is not None

    # Verify timestamps
    start_time = datetime.fromisoformat(run.start_time)
    end_time = datetime.fromisoformat(run.end_time)
    assert end_time > start_time

    # Verify log file was created
    log_file = run_log.file_path()
    assert log_file.exists()

    # Verify log file contents
    with open(log_file) as f:
        log_data = json.load(f)
    assert len(log_data) == 1
    assert log_data[0]["status"] == "completed"


def test_run_handles_errors(ai: AI, simple_dataset, simple_dataset_tasks):
    with pytest.raises(ValueError):
        ai.run(
            main=lambda row: simple_dataset_tasks["failing"](row.x),
            dataset=simple_dataset,
        )

    # Check run log still exists but shows failure
    run_log = ai._run_log
    assert len(run_log.runs) == 1

    run = run_log.runs[0]
    assert run.status == "started"  # Should remain "started" since it didn't complete
    assert run.start_time is not None
    assert run.end_time is None


def test_multiple_runs(ai: AI, simple_dataset, simple_dataset_tasks):
    # Run the task multiple times
    for _ in range(3):
        ai.run(
            main=lambda row: simple_dataset_tasks["simple"](row.x),
            dataset=simple_dataset,
        )

    # Verify run logs
    run_log = ai._run_log
    assert len(run_log.runs) == 3

    # Verify all runs completed successfully
    for run in run_log.runs:
        assert run.status == "completed"
        assert run.start_time is not None
        assert run.end_time is not None
