from dataclasses import dataclass, field

import agentlens.evaluation as ev
from agentlens import lens, task


@dataclass
class Counter:
    value: int = 0


@dataclass
class Messages:
    items: list[str] = field(default_factory=list)


# Test tasks
@task
async def simple_task(x: int) -> int:
    return x * 2


@task
async def complex_task(a: int, b: str, c: bool = True) -> str:
    return f"{a}-{b}-{c}"


@task
async def task_with_kwargs(*, name: str, value: int) -> dict:
    return {"name": name, "value": value}


@task
async def task_with_defaults(x: int, y: str = "default") -> str:
    return f"{x}-{y}"


@task
async def task_raises_error() -> None:
    raise ValueError("Expected error")


@task
async def nested_task(x: int) -> int:
    return await simple_task(x)


@task
async def increment_counter() -> int:
    counter = lens[Counter]
    counter.value += 1
    return counter.value


@task
async def add_message(msg: str) -> str:
    messages = lens[Messages]
    messages.items.append(msg)
    return msg


@task
async def get_counter_value() -> int:
    counter = lens[Counter]
    return counter.value


# Test mocks
async def mock_simple(x: int) -> int:
    return x * 3


async def mock_partial_args(a: int) -> str:
    return f"mocked-{a}"


async def mock_with_miss(x: int) -> int:
    if x > 10:
        raise ev.MockMiss()
    return x * 4


# Test hooks
@ev.hook(simple_task)
def hook_simple(x: int) -> ev.HookGenerator[int]:
    result = yield {}
    assert isinstance(result, int)
    return None


@ev.hook(complex_task)
def hook_modify_args(a: int, b: str) -> ev.HookGenerator[str]:
    result = yield {"a": a * 2, "b": b.upper()}
    assert isinstance(result, str)
    return None


@ev.hook(complex_task)
def hook_partial_args(a: int) -> ev.HookGenerator[str]:
    result = yield {"a": a + 1}
    assert isinstance(result, str)
    return None


@ev.hook(task_with_kwargs)
def hook_kwargs_only(name: str) -> ev.HookGenerator[dict]:
    result = yield {"name": name.upper()}
    assert isinstance(result, dict)
    return None


@ev.hook(simple_task)
def hook_modify_result(x: int) -> ev.HookGenerator[int]:
    yield {}


@ev.hook(simple_task)
def hook_multiply(x: int) -> ev.HookGenerator[int]:
    yield {"x": x * 2}
    return None


@ev.hook(simple_task)
def hook_add(x: int) -> ev.HookGenerator[int]:
    yield {"x": x + 1}
    return None


@ev.hook(increment_counter)
def log_increment() -> ev.HookGenerator[int]:
    messages = lens[Messages]
    result = yield {}
    messages.items.append(f"Counter incremented to {result}")
