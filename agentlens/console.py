from __future__ import annotations

import shutil
import subprocess
from logging import getLogger
from typing import Any, Callable, ClassVar

from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Tree

from agentlens.trace import Observation, Run

logger = getLogger("agentlens.console")


class RunTree(Tree):
    COMPONENT_CLASSES: ClassVar[set[str]] = {"run-tree"}

    def __init__(self, runs: list[Run], editor: str = "cursor"):
        super().__init__("Runs", id="run-tree")
        self.runs = runs
        self.editor = editor

    def on_mount(self) -> None:
        self.show_root = False
        self.guide_depth = 3
        self.root.expand()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        node = event.node
        if node.data and node.data.get("type") == "file":
            path = node.data["path"]
            editor_path = shutil.which(self.editor)
            if not editor_path:
                raise Exception(f"Could not find '{self.editor}' in PATH")
            try:
                subprocess.Popen([editor_path, path])
            except Exception as e:
                raise Exception(f"Failed to open file in {self.editor}: {e}")

    def render_trace(self, trace: Observation, parent_node: Any, run: Run) -> None:
        status = trace.get_status()
        if status == "failed":
            color = "red3"
        elif status == "running":
            color = "dodger_blue1"
        else:
            color = "green3"

        node = parent_node.add(f"[b {color}]{trace.get_status_icon()} {trace.name}[/]", expand=True)

        for log in trace.logs:
            node.add_leaf(f"{log.message}")

        for file in trace.files:
            file_path = run.dir / file.name
            abs_path = str(file_path.resolve())
            leaf_node = node.add_leaf(f"📄 {file.name}")
            leaf_node.data = {"type": "file", "path": abs_path}

        for child in trace.children:
            self.render_trace(child, node, run)

    def refresh_tree(self) -> None:
        node_states = self._get_node_states(self.root)
        self.root.remove_children()

        for i, run in enumerate(self.runs):
            run_node = self.root.add(f"Run {i}", expand=True)
            self.render_trace(run.observation, run_node, run)

        self._apply_node_states(self.root, node_states)
        self.refresh()

    def _get_node_states(self, node, path=()):
        """Recursively get the expanded state of nodes."""
        states = {}
        for child in node.children:
            key = path + (str(child.label),)
            states[key] = child.is_expanded
            states.update(self._get_node_states(child, key))
        return states

    def _apply_node_states(self, node, states, path=()):
        """Recursively apply the expanded state to nodes."""
        for child in node.children:
            key = path + (str(child.label),)
            if key in states:
                if states[key]:
                    child.expand()
                else:
                    child.collapse()
            self._apply_node_states(child, states, key)


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

    def __init__(
        self, runs: list[Run], execute_callback: Callable, editor: str = "cursor", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._runs = runs
        self._execute_callback = execute_callback
        self._editor = editor

    def compose(self) -> ComposeResult:
        self.trace_tree = RunTree(self._runs, editor=self._editor)
        with ScrollableContainer():
            yield self.trace_tree

    async def on_mount(self) -> None:
        self.refresh_worker = self.set_interval(1 / 2, self.trace_tree.refresh_tree)
        self.execution_task = self.call_later(self.safe_execute)

    async def safe_execute(self) -> None:
        try:
            await self._execute_callback()
        except Exception as e:
            logger.error(f"Execution failed: {e}")
