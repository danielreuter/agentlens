import sys
from importlib import import_module
from pathlib import Path
from typing import List

import typer

app = typer.Typer()

# Create a subcommand group for the "run" command
run_app = typer.Typer()
app.add_typer(run_app, name="run")


# Move the run function to be under the run_app group
@run_app.callback(invoke_without_command=True)
def run(
    file_path: str,
    function_name: str,
    args: List[str] = typer.Argument(None, help="Additional arguments to pass to the function"),
):
    try:
        # Convert file path to module path
        path = Path(file_path)
        if not path.suffix == ".py":
            raise ValueError("File must be a Python file")

        # Convert path/to/file.py to path.to.file
        module_path = str(path.with_suffix("")).replace("/", ".").replace("\\", ".")

        # Import the module normally, which will execute all dependencies
        module = import_module(module_path)

        # # Debug: Print available functions with their full repr
        # print("\nAvailable functions in module:")
        # for name, item in module.__dict__.items():
        #     if callable(item):
        #         print(f"- {name}: {repr(item)}")

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
        sys.argv = [file_path, *args] if args else [file_path]
        func()

    except ImportError as e:
        typer.echo(f"Failed to import module: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
