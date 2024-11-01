from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App
from textual.containers import ScrollableContainer
from textual.widgets import Tree

from agentlens.trace import Observation


class RunTree(Tree):
    COMPONENT_CLASSES: ClassVar[set[str]] = {"run-tree"}

    def __init__(self, trace: Observation):
        super().__init__("Trace Visualization", id="trace-tree")
        self.trace = trace

    def on_mount(self) -> None:
        self.show_root = False
        self.guide_depth = 3
        self.root.expand()

    def render_trace(self, trace: Observation, parent_node: Any) -> None:
        """Recursively render a trace and its children"""
        node = parent_node.add(f"{trace.get_status_icon()} {trace.name}", expand=True)
        node.add_leaf(f"⏰ Started at: {trace.start_time.strftime('%H:%M:%S')}")

        if trace.get_status() == "completed":
            node.add_leaf(f"✨ Completed in {trace.get_duration()}")
        elif trace.get_status() == "failed":
            node.add_leaf(f"❌ Failed: {trace.error}")

        for child in trace.children:
            self.render_trace(child, node)

    def refresh_tree(self) -> None:
        """Rebuild the entire tree from the current state of trace"""
        self.root.remove_children()
        self.render_trace(self.trace, self.root)
        self.refresh()


class RunConsole(App):
    CSS = """
    ScrollableContainer {
        width: 100%;
        height: 100%;
        dock: left;
        border: solid green;
    }

    TraceTree {
        width: 100%;
        height: 100%;
        background: #2c3e50;
        color: #ecf0f1;
        padding: 1;
    }
    """

    def __init__(self, trace: Observation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace = trace

    def compose(self):
        self.trace_tree = RunTree(self._trace)
        with ScrollableContainer():
            yield self.trace_tree

    def on_mount(self) -> None:
        # Start a worker to periodically refresh the tree
        self.refresh_worker = self.set_interval(1 / 30, self.trace_tree.refresh_tree)
