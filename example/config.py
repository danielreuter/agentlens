import os
from pathlib import Path

from dotenv import load_dotenv

from agentlens import AI, OpenAIProvider
from agentlens.lens import Lens
from agentlens.provider_anthropic import AnthropicProvider

load_dotenv()

ROOT_DIR = Path(__file__).parent

ls = Lens(
    runs_dir=ROOT_DIR / "runs",  # where to store runs
    dataset_dir=ROOT_DIR / "datasets",  # where to store datasets
)

ai = AI(
    providers=[
        OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            max_connections={
                "DEFAULT": 10,
                "o1-preview": 2,
                "gpt-4o-mini": 30,
            },
        ),
        AnthropicProvider(
            api_key="...",  # your Anthropic API key
            max_connections={
                "DEFAULT": 10,
                "claude-3-5-sonnet": 5,
            },
        ),
    ],
)
