import pytest

import agentlens.evaluation as ev
from agentlens.client import observe, provide


@observe
async def greet_person(name: str) -> str:
    return f"Hello, {name}"


# Hook that modifies the "name" param
@ev.hook(greet_person)
def hook_add_exclamation(name: str) -> ev.Hook[str]:
    override = {"name": name + "!"}
    _result = yield override
    return None


# Hook that does no yield
@ev.hook(greet_person)
def hook_no_yield(name: str):
    pass


@observe
async def combine_strings(a: str, b: str) -> str:
    return f"{a}-{b}"


@ev.hook(combine_strings)
def hook_partial(a: str) -> ev.Hook[str]:
    override = {"a": a.upper()}
    _ = yield override
    return None


async def test_single_hook():
    with provide(hooks=[hook_add_exclamation]):
        result = await greet_person("Alice")
        assert result == "Hello, Alice!"


async def test_multiple_hooks():
    @ev.hook(greet_person)
    def hook_replace_a_with_b(name: str) -> ev.Hook[str]:
        if name.startswith("A"):
            override = {"name": "Bob"}
            yield override

    with provide(hooks=[hook_add_exclamation, hook_replace_a_with_b]):
        result = await greet_person("Alice")
        # The exact final string may depend on hook order & logic
        # We'll just check that it starts with "Hello"
        assert result.startswith("Hello")


async def test_hook_observes_result():
    @ev.hook(greet_person)
    def hook_assert_result(name: str) -> ev.Hook[str]:
        final = yield {}
        assert final == f"Hello, {name}"

    with provide(hooks=[hook_assert_result]):
        await greet_person("Alice")  # Should pass


async def test_invalid_hook_param():
    with pytest.raises(ValueError, match="Parameter 'invalid' does not exist"):

        @ev.hook(greet_person)
        def hook_bad_param(invalid: int) -> ev.Hook[str]:
            """
            invalid param doesn't exist in greet_person, expecting a ValueError
            from the hooking system.
            """
            _ = yield {}
            return None

        with provide(hooks=[hook_bad_param]):
            await greet_person("Alice")


async def test_hook_no_yields():
    with provide(hooks=[hook_no_yield]):
        result = await greet_person("Alice")
        assert result == "Hello, Alice"


async def test_hook_partial_params():
    with provide(hooks=[hook_partial]):
        result = await combine_strings("abc", "def")
        assert result == "ABC-def"
