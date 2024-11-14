import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from agentlens import AI, OpenAIProvider
from agentlens.hooks import GeneratorHook, MockMiss
from agentlens.lens import Lens

# Initialize lens with dummy directories at module level
il: Lens = Lens(
    dataset_dir="dummy_data",
    runs_dir="dummy_runs",
)


# Test contexts
@dataclass
@il.context
class Counter:
    value: int = 0


@dataclass
@il.context
class Messages:
    items: list[str] = field(default_factory=list)


@dataclass
class UnregisteredContext:
    value: int


# Test tasks
@il.task()
async def simple_task(x: int) -> int:
    return x * 2


@il.task()
async def complex_task(a: int, b: str, c: bool = True) -> str:
    return f"{a}-{b}-{c}"


@il.task()
async def task_with_kwargs(*, name: str, value: int) -> dict:
    return {"name": name, "value": value}


@il.task()
async def task_with_defaults(x: int, y: str = "default") -> str:
    return f"{x}-{y}"


@il.task()
async def task_raises_error() -> None:
    raise ValueError("Expected error")


@il.task()
async def nested_task(x: int) -> int:
    return await simple_task(x)


@il.task()
async def increment_counter() -> int:
    counter = il[Counter]
    counter.value += 1
    return counter.value


@il.task()
async def add_message(msg: str) -> str:
    messages = il[Messages]
    messages.items.append(msg)
    return msg


@il.task()
async def get_counter_value() -> int:
    counter = il[Counter]
    return counter.value


# Test mocks
async def mock_simple(x: int) -> int:
    return x * 3


async def mock_partial_args(a: int) -> str:
    return f"mocked-{a}"


async def mock_with_miss(x: int) -> int:
    if x > 10:
        raise MockMiss()
    return x * 4


# Test hooks
@il.hook(simple_task)
def hook_simple(x: int) -> GeneratorHook[int]:
    result = yield {}
    assert isinstance(result, int)
    return None


@il.hook(complex_task)
def hook_modify_args(a: int, b: str) -> GeneratorHook[str]:
    result = yield {"a": a * 2, "b": b.upper()}
    assert isinstance(result, str)
    return None


@il.hook(complex_task)
def hook_partial_args(a: int) -> GeneratorHook[str]:
    result = yield {"a": a + 1}
    assert isinstance(result, str)
    return None


@il.hook(task_with_kwargs)
def hook_kwargs_only(name: str) -> GeneratorHook[dict]:
    result = yield {"name": name.upper()}
    assert isinstance(result, dict)
    return None


@il.hook(simple_task)
def hook_modify_result(x: int) -> GeneratorHook[int]:
    yield {}


@il.hook(simple_task)
def hook_multiply(x: int) -> GeneratorHook[int]:
    yield {"x": x * 2}
    return None


@il.hook(simple_task)
def hook_add(x: int) -> GeneratorHook[int]:
    yield {"x": x + 1}
    return None


@il.hook(increment_counter)
def log_increment() -> GeneratorHook[int]:
    messages = il[Messages]
    result = yield {}
    messages.items.append(f"Counter incremented to {result}")


# Fixtures
@pytest.fixture
def ls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lens:
    """Create a temporary Lens instance with tmp directories"""
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir()
    runs_dir.mkdir()

    # Patch the lens directories
    monkeypatch.setattr(il, "_dataset_dir", data_dir)
    monkeypatch.setattr(il, "_runs_dir", runs_dir)
    return il


@pytest.fixture
def ai():
    return AI(
        providers=[
            OpenAIProvider(
                api_key=os.getenv("OPENAI_API_KEY"),
                max_connections={
                    "DEFAULT": 10,
                    "gpt-4o-mini": 30,
                },
            )
        ],
    )
