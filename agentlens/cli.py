import asyncio
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import typer
from typing_extensions import Annotated

from agentlens.console import RunConsole
from agentlens.trace import Run

app = typer.Typer()
run_app = typer.Typer()
app.add_typer(run_app, name="run")


def wrap_function(func: Callable[[], Any], runs: list[Run]) -> Callable[[], Any]:
    """Wrap the target function to capture its run context"""

    async def wrapped():
        if asyncio.iscoroutinefunction(func):
            await func()
        else:
            func()

    return wrapped


@run_app.callback(invoke_without_command=True)
def run(
    file_path: Annotated[str, typer.Argument(help="Path to the Python file to run")],
    function_name: Annotated[str, typer.Argument(help="Name of the function to run")],
):
    """Run a Python function with AgentLens console visualization"""
    try:
        # Convert file path to module path
        path = Path(file_path)
        if not path.suffix == ".py":
            raise ValueError("File must be a Python file")

        # Convert path/to/file.py to path.to.file
        module_path = str(path.with_suffix("")).replace("/", ".").replace("\\", ".")

        # Import the module
        module = import_module(module_path)

        # Get the function
        func = module.__dict__.get(function_name)
        if func is None or not callable(func):
            available_functions = [name for name, item in module.__dict__.items() if callable(item)]
            typer.echo(
                f"\nFunction '{function_name}' not found or not callable. Available functions: {', '.join(available_functions)}",
                err=True,
            )
            raise typer.Exit(1)

        # Parse args into sys.argv for the function's CLI parser
        sys.argv = [file_path]

        # Create a list to store runs
        runs: list[Run] = []

        # Create and run the console app
        wrapped_func = wrap_function(func, runs)
        app = RunConsole(runs=runs, execute_callback=wrapped_func)
        app.run()

    except ImportError as e:
        typer.echo(f"Failed to import module: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
