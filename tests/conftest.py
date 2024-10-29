import asyncio
import os
from typing import Any, Dict, List, Optional, Union

import pytest
from pydantic import BaseModel

from agentlens import AI, Dataset, OpenAIProvider
from agentlens.dataset import Row


@pytest.fixture()
def ai(tmp_path):
    """Global AI instance for testing"""
    return AI(
        dataset_dir=tmp_path,  # Use temporary directory for test datasets
        run_dir=tmp_path,  # Use temporary directory for test runs
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


class MockInput(BaseModel):
    value: int
    multiplier: float


class MockOutput(BaseModel):
    result: float
    steps: list[str]


# Basic tasks for testing
@pytest.fixture
def ai_fixture(ai):
    # Simple task that squares a number
    @ai.task()
    async def square(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * x

    # Task that uses a Pydantic model
    @ai.task()
    async def multiply_with_logging(input: MockInput) -> MockOutput:
        await asyncio.sleep(0.1)
        result = input.value * input.multiplier
        return MockOutput(result=result, steps=[f"Multiplied {input.value} by {input.multiplier}"])

    # Cached task
    @ai.task(cache=True)
    async def expensive_operation(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    # Task without input capture
    @ai.task(capture_input=False)
    async def sensitive_operation(secret: str) -> str:
        await asyncio.sleep(0.1)
        return f"Processed: {secret}"

    # Task without output capture
    @ai.task(capture_output=False)
    async def private_result(x: int) -> str:
        await asyncio.sleep(0.1)
        return f"Secret result: {x}"

    # Composed tasks
    @ai.task()
    async def complex_calculation(x: int) -> float:
        squared = await square(x)
        doubled = await expensive_operation(squared)
        input_model = MockInput(value=doubled, multiplier=0.5)
        result = await multiply_with_logging(input_model)
        return result.result

    # New tasks using conftest models
    @ai.task()
    async def process_user_profile(user: UserProfile) -> dict:
        await asyncio.sleep(0.1)
        return {
            "summary": f"{user.name} ({user.age})",
            "status": "active" if user.is_active else "inactive",
            "performance_rating": user.score,
        }

    @ai.task(cache=True)
    async def calculate_order_total(order: Order) -> float:
        await asyncio.sleep(0.1)
        return sum(item.quantity * item.unit_price for item in order.items)

    @ai.task()
    async def validate_contact_info(contact: ContactInfo) -> list[str]:
        await asyncio.sleep(0.1)
        validations = []
        if contact.email:
            validations.append("Email present")
        if contact.phone:
            validations.append("Phone present")
        validations.append(f"Has {len(contact.addresses)} addresses")
        return validations

    return {
        **locals(),
    }


# Basic model classes
class UserProfile(BaseModel):
    name: str
    age: int
    is_active: bool
    score: float
    bio: Optional[str] = None


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    is_primary: bool


class ContactInfo(BaseModel):
    email: str
    phone: Optional[str]
    addresses: List[Address]


class OrderItem(BaseModel):
    product_id: str
    quantity: int
    unit_price: float
    notes: Optional[str] = None


class Order(BaseModel):
    order_id: str
    customer: UserProfile
    items: List[OrderItem]
    shipping_address: Address
    total: float
    metadata: Dict[str, Union[str, int, float, bool]]


# Test fixtures for primitive values
@pytest.fixture
def primitive_values():
    return [
        "hello",
        42,
        3.14,
        True,
        None,
        ["a", "b", "c"],
        {"x": 1, "y": 2},
        (1, "two", 3.0),
    ]


# Test fixtures for Data objects
@pytest.fixture
def user_profile():
    return UserProfile(name="John Doe", age=30, is_active=True, score=4.5, bio="Python developer")


@pytest.fixture
def address():
    return Address(street="123 Main St", city="Springfield", zip_code="12345", is_primary=True)


@pytest.fixture
def contact_info(address):
    return ContactInfo(email="john@example.com", phone="+1234567890", addresses=[address, address])


@pytest.fixture
def order_item():
    return OrderItem(product_id="PROD123", quantity=2, unit_price=9.99, notes="Gift wrap")


@pytest.fixture
def complex_order(user_profile, address, order_item):
    return Order(
        order_id="ORD123",
        customer=user_profile,
        items=[order_item, order_item],
        shipping_address=address,
        total=19.98,
        metadata={"source": "web", "priority": 1, "discount": 0.1, "gift": True},
    )


# Complex container fixtures
@pytest.fixture
def nested_tuple(user_profile, address):
    return (user_profile, 42, "string", (address, True, 3.14), [1, 2, 3], {"key": "value"})


@pytest.fixture
def nested_list(user_profile, contact_info, order_item):
    return [user_profile, [contact_info, "nested"], {"key": order_item}, (1, 2, 3), None, True]


@pytest.fixture
def nested_dict(user_profile, address, order_item, contact_info):
    return {
        "user": user_profile,
        "numbers": [1, 2, 3],
        "mixed": (address, "string", 42),
        "nested": {"order": order_item, "contact": contact_info, "flags": [True, False, None]},
    }


class NonSerializableObject:
    """A class that can't be serialized to JSON"""

    def __init__(self):
        self.x = lambda y: y + 1  # Functions aren't JSON serializable


@pytest.fixture
def non_serializable():
    return NonSerializableObject()


class InvalidData(BaseModel):
    name: str
    func: Any  # Will hold a lambda function


@pytest.fixture
def invalid_data():
    return InvalidData(name="test", func=lambda x: x + 1)


@pytest.fixture
def inactive_user_profile():
    return UserProfile(name="Jane Smith", age=25, is_active=False, score=3.2, bio="Data Analyst")


@pytest.fixture
def minimal_user_profile():
    return UserProfile(name="Bob Brown", age=18, is_active=True, score=0.0)


# Additional Address fixtures
@pytest.fixture
def secondary_address():
    return Address(street="456 Oak Avenue", city="Metropolis", zip_code="67890", is_primary=False)


@pytest.fixture
def international_address():
    return Address(street="10 Downing Street", city="London", zip_code="SW1A 2AA", is_primary=True)


# Additional Contact fixtures
@pytest.fixture
def minimal_contact_info(address):
    return ContactInfo(email="minimal@example.com", addresses=[address])


@pytest.fixture
def multi_address_contact(address, secondary_address, international_address):
    return ContactInfo(
        email="multi@example.com",
        phone="+9876543210",
        addresses=[address, secondary_address, international_address],
    )


# Additional Order fixtures
@pytest.fixture
def bulk_order_item():
    return OrderItem(product_id="BULK001", quantity=100, unit_price=5.00, notes="Wholesale order")


@pytest.fixture
def digital_order_item():
    return OrderItem(
        product_id="DIGITAL123", quantity=1, unit_price=29.99, notes="Digital download"
    )


@pytest.fixture
def minimal_order(minimal_user_profile, address, order_item):
    return Order(
        order_id="MIN001",
        customer=minimal_user_profile,
        items=[order_item],
        shipping_address=address,
        total=9.99,
        metadata={"source": "mobile"},
    )


@pytest.fixture
def bulk_order(user_profile, address, bulk_order_item):
    return Order(
        order_id="BULK001",
        customer=user_profile,
        items=[bulk_order_item],
        shipping_address=address,
        total=500.00,
        metadata={
            "source": "b2b",
            "priority": 2,
            "discount": 0.15,
            "gift": False,
            "shipping_method": "freight",
        },
    )


# Additional complex container fixtures
@pytest.fixture
def mixed_data_types(
    user_profile,
    inactive_user_profile,
    address,
    international_address,
    order_item,
    digital_order_item,
):
    return {
        "users": [user_profile, inactive_user_profile],
        "addresses": {"domestic": address, "international": international_address},
        "items": [order_item, digital_order_item],
        "stats": {
            "counts": [1, 2, 3, 4],
            "averages": {"price": 19.99, "quantity": 3},
            "flags": {"processed": True, "shipped": False},
        },
    }


@pytest.fixture
def edge_case_values():
    return {
        "empty": {"list": [], "dict": {}, "string": "", "tuple": ()},
        "special": {
            "zero": 0,
            "negative": -1,
            "max_int": 2**31 - 1,
            "min_int": -(2**31),
            "infinity": float("inf"),
            "nan": float("nan"),
        },
    }


# Add these new fixtures
class SimpleRow(Row):
    x: int


@pytest.fixture
def simple_dataset(ai: AI):
    @ai.dataset(name="simple")
    class SimpleDataset(Dataset[SimpleRow]): ...

    dataset = SimpleDataset()
    dataset.extend([SimpleRow(x=1), SimpleRow(x=2), SimpleRow(x=3)])
    dataset.save()
    return dataset


@pytest.fixture
def simple_dataset_tasks(ai: AI):
    @ai.task()
    async def task(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2

    @ai.task()
    async def failing(x: int) -> int:
        raise ValueError("Failed")

    return {
        "simple": task,
        "failing": failing,
    }
