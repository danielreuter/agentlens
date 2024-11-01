from __future__ import annotations

from logging import getLogger
from typing import Any, Callable, ClassVar

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Tree

from agentlens.trace import Observation, Run

logger = getLogger("agentlens.console")


class RunTree(Tree):
    COMPONENT_CLASSES: ClassVar[set[str]] = {"run-tree"}

    def __init__(self, runs: list[Run]):
        super().__init__("Runs", id="run-tree")
        self.runs = runs

    def on_mount(self) -> None:
        self.show_root = False
        self.guide_depth = 3
        self.root.expand()

    def render_trace(self, trace: Observation, parent_node: Any) -> None:
        node = parent_node.add(f"{trace.get_status_icon()} {trace.name}", expand=True)
        node.add_leaf(f"⏰ Started at: {trace.start_time.strftime('%H:%M:%S')}")

        if trace.get_status() == "completed":
            node.add_leaf(f"✨ Completed in {trace.get_duration()}")
        elif trace.get_status() == "failed":
            node.add_leaf(f"❌ Failed: {trace.error}")

        for child in trace.children:
            self.render_trace(child, node)

    def refresh_tree(self) -> None:
        # todo -- add a tab at the top for the run key
        self.root.remove_children()
        for i, run in enumerate(self.runs):
            run_node = self.root.add(f"Run {i}", expand=True)
            self.render_trace(run.observation, run_node)
        self.refresh()


class RunConsole(App):
    CSS = """
    ScrollableContainer {
        width: 100%;
        height: 100%;
        dock: left;
        border: solid green;
    }

    .run-tree {
        width: 100%;
        height: 100%;
        background: #2c3e50;
        color: #ecf0f1;
        padding: 1;
    }
    """

    def __init__(self, runs: list[Run], execute_callback: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._runs = runs
        self._execute_callback = execute_callback

    def compose(self) -> ComposeResult:
        self.trace_tree = RunTree(self._runs)
        with ScrollableContainer():
            yield self.trace_tree

    async def on_mount(self) -> None:
        self.refresh_worker = self.set_interval(1 / 30, self.trace_tree.refresh_tree)
        self.execution_task = self.call_later(self.safe_execute)

    async def safe_execute(self) -> None:
        try:
            await self._execute_callback()
        except Exception as e:
            logger.error(f"Execution failed: {e}")
