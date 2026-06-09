# -*- coding: utf-8 -*-
"""The graph trace notebook class, used to manage the graph trace state, 
providing hints and tool functions to the agent."""

from collections import OrderedDict
import re
from agentscope.message import Msg, TextBlock
from agentscope.module import StateModule
from agentscope.tool import ToolResponse
from smartcomment.runtime import ExecNetwork
from smartcomment.runtime.errors import ExecNetworkKeyError
from ._utils._agentscope import _execute_async_or_sync_func
from .models import GraphTraceNodeRecord, GraphTraceState
from ._utils._helpers import _paginate_text, _render_operation_subgraph
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Literal,
)


# Synthetic operation identifier and name for implicit root expansion.
DUMMY_OP = "DUMMY_OP"

DUMMY_OP_COMMENT = (
    "It is a synthetic operation used to expand from the implicit dummy "
    "graph trace node to every graph root. This operation is not stored in the execution graph."
)

# Synthetic graph trace node identifier used before a graph root is selected.
DUMMY_ROOT_NODE_ID = "__MEMTRACE_DUMMY_ROOT__@1"

# Synthetic timestamp used for the dummy root node.
DUMMY_TIMESTAMP = "0000-00-00 00:00:00.000"

# The category for the dummy root node.
DUMMY_ROOT_NODE_CATEGORY = "synthetic_dummy_root"

# The comment for the dummy root node.
DUMMY_ROOT_NODE_COMMENT = (
    "It is a synthetic root node used to expose all execution graph nodes with in-degree 0."
)

# Default maximum number of characters returned by exploration tools.
MAXIMUM_CHARACTER_LIMIT = 128_000


class DefaultGraphTraceToHint:
    """Generate graph trace hints from the current graph trace state."""

    hint_prefix: str = "<system-hint>"
    hint_suffix: str = "</system-hint>"

    no_graph_trace: str = (
        "There is an execution graph in the environment. " 
        "Call `initialize_execution_graph` first when the user's task requires "
        "the execution graph, including failure attribution, root-cause "
        "analysis, direct graph exploration, finding or inspecting a specific "
        "graph node, tracing how a value is produced, summarizing the graph, "
        "or answering questions that depend on graph structure or node history. "
        "This will load the corresponding execution graph from the environment, "
        "and tell you your initial position in the execution graph. " 
        "You can then explore this execution graph."
    )

    when_no_node_in_exploration: str = (
        "The current graph trace state:\n"
        "```\n"
        "{state}\n"
        "```\n"
        "No graph trace node is currently being explored. Your options include:\n"
        "- Explore the earliest variable node in the exploration frontier by calling "
        "`explore_graph_trace_node`. If you expect the returned operations to be very large, "
        "for example because this variable's value is large, you can set "
        "`include_variable_value=False` to inspect an operation preview without variable values.\n"
        "- If you think you can answer the user's task based on the current graph trace result, " 
        "call the function `finish_graph_exploration` to finish the graph exploration process."
    )

    when_a_node_in_exploration: str = (
        "The current graph trace state:\n "
        "```\n"
        "{state}\n"
        "```\n"
        "A graph trace node is currently marked as `exploring`. Your options "
        "include:\n"
        "- If the previous `explore_graph_trace_node` result is truncated, "
        "call `explore_graph_trace_node` again with the next `offset`. If the previous "
        "`explore_graph_trace_node` result is a preview without variable values, "
        "choose relevant variables from that preview and call `view_node_value` to inspect "
        "their values.\n"
        "- Call `update_to_explore_nodes` with action `add` or `delete` to "
        "maintain the future exploration frontier.\n"
        "- Call `finish_node_exploration` with the node-level outcome once "
        "you are done inspecting this node."
    )

    def __call__(self, state: GraphTraceState | None) -> str:
        """Generate a hint message from graph trace state.

        Args:
            state (`GraphTraceState | None`):
                The current graph trace state.

        Returns:
            `str`:
                The generated hint message.
        """
        if state is None:
            hint = self.no_graph_trace
        elif state.get_exploring_node() is None:
            hint = self.when_no_node_in_exploration.format(
                state=state.to_markdown(),
            )
        else:
            hint = self.when_a_node_in_exploration.format(
                state=state.to_markdown(),
            )

        return f"{self.hint_prefix}{hint}{self.hint_suffix}"


class GraphTraceNotebook(StateModule):
    """The graph trace state notebook used to manage the graph trace state, 
    provide tools and hints to the agent."""

    description: str = (
        "The execution-graph exploration tools. There is a predefined execution "
        "graph in the environment. Activate this tool when you need to perform "
        "failure attribution, root-cause analysis, graph exploration, node "
        "inspection, value provenance tracing, graph summarization, or any "
        "other task that depends on the execution graph. Once activated, "
        "you'll enter the graph trace mode, where you will be guided to "
        "complete the given query by initializing and exploring the execution graph, "
        "and hint messages wrapped by <system-hint></system-hint> will guide you " 
        "through the graph trace process. If you think you can answer the user's task " 
        "based on the current graph trace result, call the `finish_graph_exploration` "
        "function. If the user later no longer wants to perform the execution graph-related task, " 
        "such as asking for a task unrelated to the execution graph, confirm with the user and"
        "call the `finish_graph_exploration` function."
    )

    def __init__(
        self,
        graph: ExecNetwork,
        max_trace_nodes: int | None = None,
        graph_trace_to_hint: Callable[[GraphTraceState | None], str] | None = None,
        include_metadata: bool = False,
    ) -> None:
        """Initialize the graph trace notebook.

        Args:
            graph (`ExecNetwork`):
                The original smartcomment execution graph. It is kept immutable.
            max_trace_nodes (`int | None`, optional):
                The Maximum number of nodes to be explored.
            graph_trace_to_hint (`Callable[[GraphTraceState | None], str] | None`, optional):
                The function to generate hints based on the current graph trace state.
            include_metadata (`bool`, defaults to `False`):
                Whether rendered smartcomment subgraphs include metadata.
        """
        super().__init__()

        self._original_graph = graph
        self._working_graph = None
        self.max_trace_nodes = max_trace_nodes
        self.graph_trace_to_hint = graph_trace_to_hint or DefaultGraphTraceToHint()
        self.include_metadata = include_metadata
        self.current_state = None
        self._graph_trace_change_hooks = OrderedDict()

        self.register_state(
            "current_state",
            custom_to_json=lambda _: _.model_dump() if _ else None,
            custom_from_json=lambda _: GraphTraceState.model_validate(_) if _ else None,
        )

    def _validate_current_state(self) -> None:
        """Check if the current graph trace state is initialized.
        
        Raises:
            `ValueError`:
                If the current graph trace state is not initialized.
        """
        if self.current_state is None:
            raise ValueError(
               "The execution graph is not loaded. Please call "
                "`initialize_execution_graph` first.",
            )

    async def initialize_execution_graph(
        self,
        initial_full_node_ids: list[str] | None = None,
    ) -> ToolResponse:
        """Load and initialize the execution graph and the exploration frontier.

        Args:
            initial_full_node_ids (`list[str] | None`, optional):
                Optional full node identifiers used as the initial exploration frontier. 
                If omitted, the synthetic dummy root node is inserted instead, so the
                first exploration step will list the execution graph roots and let you choose 
                which root nodes to add.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        if self._working_graph is not None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: The execution graph is already loaded.",
                    ),
                ],
            )

        # Load the original execution graph into the working graph.
        self._working_graph = ExecNetwork.import_graph(
            self._original_graph.export_graph(),
        )
        self.current_state = GraphTraceState()

        if initial_full_node_ids:
            for full_node_id in initial_full_node_ids:
                record = self._record_from_graph(full_node_id)
                self.current_state.add_to_explore(record)
        else:
            self.current_state.add_to_explore(
                GraphTraceNodeRecord(
                    full_node_id=DUMMY_ROOT_NODE_ID,
                    created_at=DUMMY_TIMESTAMP,
                    category=DUMMY_ROOT_NODE_CATEGORY,
                    comment=DUMMY_ROOT_NODE_COMMENT,
                ),
            )

        await self._trigger_graph_trace_change_hooks()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        "The execution graph is loaded successfully. " 
                        "The initial exploration frontier is set with "
                        f"{len(self.current_state.to_explore_nodes)} variable node(s)."
                    ),
                ),
            ],
        )

    async def explore_graph_trace_node(
        self,
        offset: int = 1,
        limit: int = 128_000,
        include_variable_value: bool = True,
    ) -> ToolResponse:
        """Explore the current or earliest to-explore graph trace node by inspecting
        the operations where this variable is involved.

        Args:
            offset (`int`, defaults to `1`):
                One-based character offset to start reading from. It is ignored
                when `include_variable_value` is `False`.
            limit (`int`, defaults to `128000`):
                Maximum number of characters to return. It is ignored when
                `include_variable_value` is `False`.
            include_variable_value (`bool`, defaults to `True`):
                Whether to include stored variable values in the operation
                subgraph output. Set it to `False` to get an unpaginated preview
                of the operations where the variable is involved without showing
                variable values.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        self._validate_current_state()

        state = self.current_state
        node = state.get_exploring_node() or state.start_next_exploration()
        if node is None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "Error: There is no variable node to explore. "
                            "If you think you can answer the user's task based on the current graph trace result, "
                            "call the `finish_graph_exploration` function to finish the graph exploration process. " 
                            "If you find that you make a mistake and the exploration frontier contains no variable "
                            "nodes, you need to restart. You can call `view_exploration_history` and `view_node_value` " 
                            "to determine the initial exploration frontier for restarting. " 
                            "After confirming, you can call the `finish_graph_exploration` function to end the graph " 
                            "exploration process. Then call the `initialize_execution_graph` function to start over."
                        ),
                    ),
                ],
            )

        full_text = self._render_node_operations(
            node,
            include_variable_value=include_variable_value,
        )
        if not include_variable_value:
            await self._trigger_graph_trace_change_hooks()
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"```text\n{full_text}\n```",
                    ),
                    TextBlock(
                        type="text",
                        text=(
                            "`include_variable_value` is `False`, so this is an "
                            "unpaginated operation preview. Both `offset` and `limit` "
                            "are ignored. Use `view_node_value` on relevant "
                            "variable full node identifiers if you need their values."
                        ),
                    ),
                ],
            )

        effective_limit = min(limit, MAXIMUM_CHARACTER_LIMIT)
        window, status = _paginate_text(
            full_text, 
            offset=offset, 
            limit=effective_limit,
        )
        
        # This tool is called successfully.
        blocks = [] 
        if effective_limit < limit:
            blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        "This tool can display at most "
                        f"{MAXIMUM_CHARACTER_LIMIT} characters per call. "
                        f"`limit` has been automatically set to `{MAXIMUM_CHARACTER_LIMIT}`."
                    ),
                ),
            )
        blocks.extend(
            [
                TextBlock(
                    type="text",
                    text=f"```text\n{window}\n```",
                ), 
                TextBlock(
                    type="text",
                    text=status,
                ),
            ]
        )

        await self._trigger_graph_trace_change_hooks()
        return ToolResponse(content=blocks)

    async def update_to_explore_nodes(
        self,
        action: Literal["add", "delete"],
        full_node_id: str,
        discovered_by_op_id: str | None = None,
    ) -> ToolResponse:
        """Add or delete a node in the exploration frontier.

        Args:
            action (`Literal["add", "delete"]`):
                Whether to add or delete a frontier node.
            full_node_id (`str`):
                The graph trace variable full node identifier.
            discovered_by_op_id (`str | None`, optional):
                The operation identifier that leads you to discover this variable
                node. You can provide it when adding a node from the currently explored
                operation context. It is ignored when `action` is `delete`.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        self._validate_current_state()

        state = self.current_state
        if action not in ["add", "delete"]:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: The action `{action}` is invalid. " 
                            "It must be one of `add` or `delete`."
                        ),
                    ),
                ],
            )

        if action == "delete":
            removed = state.remove_to_explore(full_node_id)
            if removed is None:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=(
                                f"Error: Variable `{full_node_id}` " 
                                "is not in the exploration frontier."
                            ),
                        ),
                    ],
                )
            await self._trigger_graph_trace_change_hooks()
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"The variable `{full_node_id}` is deleted from the "
                            "exploration frontier successfully."
                        ),
                    ),
                ],
            )

        exploring_node = state.get_exploring_node()
        if exploring_node is not None and exploring_node.full_node_id == full_node_id:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: The variable `{full_node_id}` " 
                            "is currently being explored."
                        ),
                    ),
                ],
            )
        if state.get_to_explore(full_node_id) is not None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: The variable `{full_node_id}` " 
                            "is already in the exploration frontier."
                        ),
                    ),
                ],
            )
        if state.has_explored(full_node_id):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: The variable `{full_node_id}` " 
                            "has already been explored before. "
                            "You can call `view_exploration_history` to view the exploration history."
                        ),
                    ),
                ],
            )
        if (
            self.max_trace_nodes is not None
            and state.frontier_size - int(exploring_node is not None) >= self.max_trace_nodes
        ):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "The exploration frontier has reached its maximum size "
                            f"of {self.max_trace_nodes}. Delete an existing node before "
                            "adding another. Note that the node whose state is `exploring` " 
                            "is not counted."
                        ),
                    ),
                ],
            )

        record = self._record_from_graph(
            full_node_id,
            discovered_from=(
                exploring_node.full_node_id if exploring_node else None
            ),
            discovered_by_op_id=discovered_by_op_id,
        )

        state.add_to_explore(record)
        await self._trigger_graph_trace_change_hooks()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"The variable `{full_node_id}` is added "
                        "to the exploration frontier successfully."
                    ),
                ),
            ],
        )

    async def view_node_value(
        self,
        full_node_id: str,
        pattern: str | None = None,
        max_surrounding_chars: int = 120,
        offset: int = 1,
        limit: int = 128_000,
    ) -> ToolResponse:
        """View or search a graph variable value by character positions.

        Args:
            full_node_id (`str`):
                The variable's full node identifier.
            pattern (`str | None`, optional):
                Regular expression pattern to search. If omitted, the value is
                returned directly with character pagination.
            max_surrounding_chars (`int`, defaults to `120`):
                Number of characters shown before and after each match.
                It should be non-negative.
            offset (`int`, defaults to `1`):
                One-based character offset where reading or matching starts.
                It should be greater than or equal to 1.
            limit (`int`, defaults to `128000`):
                Maximum number of output characters to return.
                It should be greater than or equal to 1.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        if offset < 1:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: `offset` must be greater than or equal to 1.",
                    ),
                ],
            )
        if limit < 1:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: `limit` must be greater than or equal to 1.",
                    ),
                ],
            )
        if max_surrounding_chars < 0:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="Error: `max_surrounding_chars` must be non-negative.",
                    ),
                ],
            )

        self._validate_current_state()
        effective_limit = min(limit, MAXIMUM_CHARACTER_LIMIT)
        blocks = []
        if effective_limit < limit:
            blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        "This tool can display at most "
                        f"{MAXIMUM_CHARACTER_LIMIT} characters per call. "
                        f"`limit` has been automatically set to `{MAXIMUM_CHARACTER_LIMIT}`."
                    ),
                ),
            )

        try:
            value = self.working_graph.get_variable(full_node_id).raw_value
        except ExecNetworkKeyError:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"Error: The variable `{full_node_id}` " 
                            "is not found in the execution graph. "
                            "Please check the provided full node identifier."
                        ),
                    ),
                ],
            )

        if pattern is None:
            window, status = _paginate_text(
                value, 
                offset=offset, 
                limit=effective_limit,
            )
            blocks.extend(
                [
                    TextBlock(
                        type="text",
                        text=(
                            f"# Value for `{full_node_id}`\n"
                            f"```text\n{window}\n```"
                        ),
                    ),
                    TextBlock(
                        type="text",
                        text=status,
                    ),
                ],
            )
            return ToolResponse(
                content=blocks
            )

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"The regex pattern `{pattern}` is invalid. "
                            f"Below is the error message: {e}\n\n"
                            "Please check the provided regex pattern."
                        ),
                    ),
                ],
            )

        start_index = offset - 1
        if start_index >= len(value):
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"No content is available from the {offset}-th character. "
                            f"The full value has {len(value)} characters."
                        ),
                    ),
                ],
            )

        matches = []
        for match in regex.finditer(value, pos=start_index):
            match_start = match.start() + 1
            match_end = match.end()
            context_start = max(0, match.start() - max_surrounding_chars)
            context_end = min(len(value), match.end() + max_surrounding_chars)
            prefix = "..." if context_start > 0 else ""
            suffix = "..." if context_end < len(value) else ""
            snippet = prefix + value[context_start:context_end] + suffix
            matches.append(f"{match_start}:{match_end}: {snippet}")

        if not matches:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            f"No matches are found for the regex pattern `{pattern}` " 
                            f"from the {offset}-th character."
                        ),
                    ) 
                ],
            )

        output = "\n--\n".join(matches)
        window, status = _paginate_text(
            output, 
            offset=1, 
            limit=effective_limit,
        )
        blocks.extend(
            [
                TextBlock(
                    type="text",
                    text=(
                        f"# Matches for `{pattern}` in variable `{full_node_id}`\n"
                        f"```text\n{window}\n```"
                    ),
                ),
                TextBlock(
                    type="text",
                    text=status,
                ),
            ]
        )
        return ToolResponse(content=blocks)

    async def finish_node_exploration(self, outcome: str) -> ToolResponse:
        """Finish the currently exploring node.

        Args:
            outcome (`str`):
                The node exploration outcome.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        self._validate_current_state()

        state = self.current_state
        exploring_node = state.get_exploring_node()
        if exploring_node is None:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "Error: There is no variable node in exploration. "
                            "If you want to explore a variable node, call the function " 
                            "`explore_graph_trace_node` first."
                        ),
                    ),
                ],
            )

        history_record = state.finish_exploring_node(outcome)

        await self._trigger_graph_trace_change_hooks()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"The variable `{history_record.full_node_id}` is " 
                        "marked `explored` successfully."
                    ),
                ),
            ],
        )

    async def finish_graph_exploration(self, outcome: str) -> ToolResponse:
        """Finish the graph exploration process.

        Args:
            outcome (`str`):
                Final graph exploration outcome.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        # TODO: Future multi-operation decisive error sets may prune the
        # working graph before finalization by deleting downstream edges of
        # known faulty operations.
        self._validate_current_state()

        self.current_state.finish(outcome)
        metadata = self.current_state.model_dump()
        self.current_state = None
        self._working_graph = None
        
        await self._trigger_graph_trace_change_hooks()
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="The graph exploration process is finished successfully.",
                ),
            ],
            metadata=metadata,
        )

    async def view_exploration_history(self) -> ToolResponse:
        """View finished node exploration history.

        Returns:
            `ToolResponse`:
                The response of the tool call.
        """
        self._validate_current_state()

        state = self.current_state
        history = state.exploration_history
        if not history:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text="There is no exploration history.",
                    ),
                ],
            )

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="\n\n".join(
                        record.to_markdown() for record in history.values()
                    ),
                ),
            ],
        )

    def list_tools(
        self,
    ) -> list[Callable[..., Coroutine[Any, Any, ToolResponse]]]:
        """List all tool functions provided to the agent.

        Returns:
            `list[Callable[..., Coroutine[Any, Any, ToolResponse]]]`:
                A list of all tool functions provided by the notebook 
                to the agent.
        """
        return [
            self.initialize_execution_graph,
            self.explore_graph_trace_node,
            self.view_node_value,
            self.update_to_explore_nodes,
            self.finish_node_exploration,
            self.finish_graph_exploration,
            self.view_exploration_history,
        ]

    async def get_current_hint(self) -> Msg:
        """Get the hint message based on the current graph trace state.

        Returns:
            `Msg`:
                The hint message wrapped by <system-hint></system-hint>, or
                None if there is no relevant hint.
        """
        hint_content = self.graph_trace_to_hint(self.current_state)
        return Msg(
            "user", 
            hint_content, 
            "user",
        )

    def register_graph_trace_change_hook(
        self,
        hook_name: str,
        hook: Callable[["GraphTraceNotebook"], None]
        | Callable[["GraphTraceNotebook"], Awaitable[None]],
    ) -> None:
        """Register a hook triggered when the graph trace state changes.

        Args:
            hook_name (`str`):
                Unique hook name.
            hook (`Callable[[GraphTraceNotebook], None] | Callable[[GraphTraceNotebook], Awaitable[None]]`):
                Hook function.
        """
        if hook_name in self._graph_trace_change_hooks:
            raise ValueError(f"Hook named '{hook_name}' already exists.")
        self._graph_trace_change_hooks[hook_name] = hook

    def remove_graph_trace_change_hook(self, hook_name: str) -> None:
        """Remove a registered graph trace change hook.

        Args:
            hook_name (`str`):
                Hook name to remove.
        """
        if hook_name not in self._graph_trace_change_hooks:
            raise ValueError(f"Hook named '{hook_name}' does not exist.")
        self._graph_trace_change_hooks.pop(hook_name)

    @property
    def working_graph(self) -> ExecNetwork | None:
        """Return the current working execution graph.
        
        Returns:
            `ExecNetwork | None`:
                The current working execution graph, if initialized.
        """ 
        return self._working_graph

    def _record_from_graph(
        self,
        full_node_id: str,
        *,
        discovered_from: str | None = None,
        discovered_by_op_id: str | None = None,
    ) -> GraphTraceNodeRecord:
        """Create a graph trace node record from the working graph.

        Args:
            full_node_id (`str`):
                Full node identifier to load.
            discovered_from (`str | None`, optional):
                Source graph trace node.
            discovered_by_op_id (`str | None`, optional):
                Operation that exposes this node.

        Returns:
            `GraphTraceNodeRecord`:
                The created record.

        Raises:
            `ValueError`:
                If the variable is not found in the current execution graph.
        """
        try:
            variable = self.working_graph.get_variable(full_node_id)
        except ExecNetworkKeyError as e:
            raise ValueError(
                f"Variable (full node identifier: `{full_node_id}`) is not found " 
                "in the current execution graph. Please check the provided full node identifier."
            ) from e

        return GraphTraceNodeRecord(
            full_node_id=variable.full_node_id,
            created_at=variable.created_at,
            category=variable.category,
            comment=variable.comment,
            discovered_from=discovered_from,
            discovered_by_op_id=discovered_by_op_id,
        )

    def _render_node_operations(
        self,
        node: GraphTraceNodeRecord,
        *,
        include_variable_value: bool = True,
    ) -> str:
        """Render operation subgraphs involving a graph trace node.

        Args:
            node (`GraphTraceNodeRecord`):
                The node to render.
            include_variable_value (`bool`, defaults to `True`):
                Whether to include stored values for variable nodes.

        Returns:
            `str`:
                Rendered operation subgraphs.
        """
        if node.full_node_id == DUMMY_ROOT_NODE_ID:
            return self._render_dummy_root_expansion(
                include_variable_value=include_variable_value,
            )
        return self._render_variable_operations(
            node.full_node_id,
            include_variable_value=include_variable_value,
        )

    def _render_dummy_root_expansion(
        self,
        *,
        include_variable_value: bool = True,
    ) -> str:
        """Render synthetic dummy root expansion.

        Args:
            include_variable_value (`bool`, defaults to `True`):
                Whether to include stored values for root variable nodes.

        Returns:
            `str`:
                Text listing graph roots and instructions for adding them.
        """
        roots = sorted(
            self.working_graph.get_root_nodes(),
            key=lambda node: (node.created_at, node.full_node_id),
        )
        if roots:
            roots_markdown = "\n\n".join(
                root.to_xml(
                    include_metadata=self.include_metadata,
                    include_variable_value=include_variable_value,
                )
                for root in roots
            )
            instruction = (
                "Call the function `update_to_explore_nodes` "
                "with root variable identifiers when you think they should be explored next."
            ) 
        else: 
            roots_markdown = "There are no root graph trace nodes in the execution graph."
            instruction = (
                "The execution graph is not a directed acyclic graph or it is empty. " 
                "Please remind the user to check the execution graph."
            ) 

        return "\n".join(
            [
                f"# Operation `{DUMMY_OP}`",
                "",
                f"- Name: `{DUMMY_OP}`",
                f"- Comment: {DUMMY_OP_COMMENT}",
                "",
                "## Root Graph Trace Nodes",
                roots_markdown,
                "",
                instruction,
            ],
        )

    def _render_variable_operations(
        self,
        full_node_id: str,
        *,
        include_variable_value: bool = True,
    ) -> str:
        """Render all operation subgraphs involving a graph variable.

        Args:
            full_node_id (`str`):
                Variable to explore.
            include_variable_value (`bool`, defaults to `True`):
                Whether to include stored values for variable nodes.

        Returns:
            `str`:
                Operation subgraphs rendered for agent inspection.
        """
        try:
            variable = self.working_graph.get_variable(full_node_id)
        except ExecNetworkKeyError as e:
            raise ValueError(
                f"Variable (full node identifier: `{full_node_id}`) is not found " 
                "in the current execution graph. Please check the provided full node identifier."
            ) from e

        operations = self.working_graph.get_operations_by_variable(full_node_id)
        if not operations:
            return "\n".join(
                [
                    f"# Graph Trace Node `{variable.full_node_id}`",
                    "",
                    "No operations directly involve this variable.",
                ],
            )

        sections = [
            "\n".join(
                [
                    f"# Graph Trace Node `{variable.full_node_id}`",
                    "",
                    f"- Category: `{variable.category}`",
                    *(
                        [f"- Comment: {variable.comment}"] if variable.comment else []
                    ),
                    f"- Inserted into Execution Graph At: `{variable.created_at}`",
                    f"- Number of Operations Found: {len(operations)}",
                ],
            ),
        ]
        for idx, op in enumerate(operations, start=1):
            subgraph = self.working_graph.filter_by_operation(op.op_id)
            sections.append(
                "\n".join(
                    [
                        f"# Operation {idx} of {len(operations)}",
                        "",
                        _render_operation_subgraph(
                            subgraph,
                            op,
                            full_node_id,
                            render_format="xml",
                            include_metadata=self.include_metadata,
                            include_variable_value=include_variable_value,
                        ),
                    ],
                ),
            )
        return "\n\n---\n\n".join(sections)

    async def _trigger_graph_trace_change_hooks(self) -> None:
        """Trigger registered graph trace notebook change hooks."""
        for hook in self._graph_trace_change_hooks.values():
            await _execute_async_or_sync_func(hook, self)
