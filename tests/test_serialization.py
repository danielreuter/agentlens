import pytest

from agentlens.serialization import (
    TaskInput,
    TaskOutput,
    serialize_task_input,
    serialize_task_output,
)
from tests.conftest import Order, UserProfile


def test_user_profile_serialization():
    user_profile = UserProfile(
        name="John Doe", age=30, is_active=True, score=4.5, bio="Python developer"
    )
    dumped = user_profile.model_dump()
    deserialized = UserProfile.model_validate(dumped)
    assert user_profile == deserialized


def test_complex_order_serialization(complex_order):
    dumped = complex_order.model_dump()
    deserialized = Order.model_validate(dumped)
    assert complex_order == deserialized


def test_task_input_serialization():
    user = UserProfile(name="Test", age=25, is_active=True, score=4.0)
    task_input = TaskInput(args=[1, "test", user], kwargs={"flag": True})
    task_input.model_dump_json()


def test_task_output_serialization():
    result = UserProfile(name="Test", age=25, is_active=True, score=4.0)
    task_output = TaskOutput(return_value=result)
    task_output.model_dump_json()


def test_serialize_task_input_helper():
    user = UserProfile(name="Test", age=25, is_active=True, score=4.0)
    result = serialize_task_input(1, "test", user, flag=True)
    assert result == {"args": [1, "test", user.model_dump()], "kwargs": {"flag": True}}


def test_serialize_task_output_helper():
    user = UserProfile(name="Test", age=25, is_active=True, score=4.0)
    result = serialize_task_output(user)
    assert result == {"return_value": user.model_dump()}


def test_task_input_with_invalid_type(non_serializable):
    with pytest.raises(ValueError):
        TaskInput(args=[non_serializable], kwargs={}).model_dump()


def test_task_output_with_invalid_type(non_serializable):
    with pytest.raises(ValueError):
        TaskOutput(return_value=non_serializable).model_dump()


def test_empty_task_input():
    task_input = TaskInput(args=[], kwargs={})
    serialized = task_input.model_dump_json()
    deserialized = TaskInput.model_validate_json(serialized)
    assert task_input == deserialized


def test_nested_serialization():
    nested_input = TaskInput(
        args=[TaskInput(args=[1, 2], kwargs={"inner": True}), "test"], kwargs={"outer": False}
    )
    serialized = nested_input.model_dump_json()
    deserialized = TaskInput.model_validate_json(serialized)
    assert nested_input == deserialized
