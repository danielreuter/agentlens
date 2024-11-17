from pathlib import Path

import pytest

import agentlens
from agentlens.client import Lens


# Fixtures
@pytest.fixture
def lens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lens:
    """Create a temporary Lens instance with tmp directories"""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Patch the lens directories
    monkeypatch.setattr(agentlens.lens, "_runs_dir", runs_dir)
    return agentlens.lens
