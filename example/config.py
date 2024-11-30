import os

from dotenv import load_dotenv

from agentlens.provider_anthropic import Anthropic
from agentlens.provider_openai import OpenAI

load_dotenv()

openai = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_connections_default=10,
    max_connections={
        "o1-preview": 2,
        "gpt-4o-mini": 30,
    },
)

anthropic = Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"],
    max_connections_default=10,
    max_connections={
        "claude-3-5-sonnet": 5,
    },
)
