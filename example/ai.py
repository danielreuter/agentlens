import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse

from agentlens import AI, OpenAIProvider

load_dotenv()

ROOT_DIR = Path(__file__).parent


langfuse = Langfuse(
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)

ai = AI(
    run_dir=ROOT_DIR / "runs",  # where to store runs
    dataset_dir=ROOT_DIR / "datasets",  # where to store datasets
    langfuse=langfuse,
    providers=[
        OpenAIProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            max_connections={
                "DEFAULT": 10,
                "o1-preview": 2,
                "gpt-4o-mini": 30,
            },
        )
    ],
)
