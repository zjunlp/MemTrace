# -*- coding: utf-8 -*-
"""The models used for failure attribution based on an execution graph."""

from collections import OrderedDict
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
)
from sortedcontainers import SortedDict
from ._utils._agentscope import _get_timestamp
from typing import Any, Literal


class GraphTraceNodeRecord(BaseModel):
    """It represents a variable in the execution graph."""

    full_node_id: str = Field(
        description=(
            "The full node identifier. "
            "It is the unique identifier of a variable in the execution graph."
        ),
    )
    created_at: str = Field(
        default_factory=_get_timestamp,
        description=(
            "The time at which this variable is inserted " 
            "into the execution graph."
        ),
    )
    state: Literal["to_explore", "exploring", "explored"] = Field(
        default="to_explore",
        description="The graph trace exploration state of this node.",
    )
    comment: str | None = Field(
        default=None,
        description="The variable comment or description.",
    )
    category: str = Field(
        default="variable",
        description="The variable category in the execution graph.",
    )
    added_at: str = Field(
        default_factory=_get_timestamp,
        description=(
            "The time this node is added to the graph trace state. "
            "This timestamp is different from the creation time of the variable."
        ),
    )
    discovered_from: str | None = Field(
        default=None,
        description="The trace node from which this one is discovered.",
    )
    discovered_by_op_id: str | None = Field(
        default=None,
        description="The operation that exposes this node.",
    )
    explored_at: str | None = Field(
        default=None,
        description="The time this variable is marked explored.",
    )
    outcome: str | None = Field(
        default=None,
        description="The outcome of this variable's exploration.",
    )

    @field_validator("full_node_id")
    @classmethod
    def _validate_full_node_id(cls, value: str) -> str:
        """Validate that a full node identifier ends with an integer version.

        Args:
            value (`str`):
                The full node identifier.

        Returns:
            `str`:
                The validated full node identifier.
        """
        if "@" not in value:
            raise ValueError("`full_node_id` must contain an '@' version separator.")
        version_text = value.rsplit("@", 1)[-1]
        if not version_text.isdigit():
            raise ValueError(
                "The substring after the last '@' in `full_node_id` must be an integer.",
            )
        return value

    @computed_field  # type: ignore[prop-decorator]
    @property
    def version(self) -> int:
        """Return the version parsed from the full node identifier.
        
        Returns:
            `int`:
                The version of the variable.
        """
        return int(self.full_node_id.rsplit("@", 1)[-1])

    def to_markdown(self) -> str:
        """Convert the graph trace node record to Markdown.

        Returns:
            `str`:
                A compact Markdown representation.
        """
        lines = [
            f"- `{self.full_node_id}`",
            f"\t- Version: `{self.version}`",
            f"\t- Inserted into Execution Graph At: `{self.created_at}`",
            f"\t- Added to Your To-Explore Frontier At: `{self.added_at}`",
            f"\t- Category: `{self.category}`",
            f"\t- State: `{self.state}`",
        ]
        if self.discovered_from:
            lines.append(f"\t- Discovered From: variable `{self.discovered_from}`")
        if self.discovered_by_op_id:
            lines.append(f"\t- Discovered By: operation `{self.discovered_by_op_id}`")
        if self.comment:
            lines.append(f"\t- Comment: {self.comment}")
        if self.explored_at:
            lines.append(f"\t- Finished Exploration At: `{self.explored_at}`")
        if self.outcome:
            lines.append(f"\t- Outcome: {self.outcome}")
        return "\n".join(lines)

    def to_explore_key(self) -> tuple[str, str]:
        """Return the priority key for the to-explore frontier.

        Returns:
            `tuple[str, str]`:
                A tuple containing the creation time and full node identifier.
        """
        return (self.created_at, self.full_node_id)

    def finish(self, outcome: str) -> None:
        """Finish this node exploration with the given outcome.

        Args:
            outcome (`str`):
                The outcome of this node's exploration.
        """
        self.state = "explored"
        self.explored_at = _get_timestamp()
        self.outcome = outcome

    def __lt__(self, other: Any) -> bool:
        """Compare records by exploration priority.

        Args:
            other (`Any`):
                The other graph trace node record.

        Returns:
            `bool`:
                Whether this record should be explored before the other one.
        """
        if not isinstance(other, GraphTraceNodeRecord):
            return NotImplemented
        return self.to_explore_key() < other.to_explore_key()


class GraphTraceState(BaseModel):
    """Store the data state of one graph trace exploration."""

    # It allows sorted containers as runtime fields with custom serialization.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    to_explore_nodes: SortedDict[tuple[str, str], GraphTraceNodeRecord] = Field(
        default_factory=SortedDict,
        description="Graph trace nodes waiting to be explored.",
    )
    exploration_history: OrderedDict[str, GraphTraceNodeRecord] = Field(
        default_factory=OrderedDict,
        description="Finished node exploration records.",
    )
    _to_explore_keys_by_id: dict[str, tuple[str, str]] = PrivateAttr(
        default_factory=dict,
    )
    finished_at: str | None = Field(
        default=None,
        description="The time the graph trace exploration is finished.",
    )
    outcome: str | None = Field(
        default=None,
        description="Final graph exploration outcome.",
    )

    @field_validator("to_explore_nodes", mode="before")
    @classmethod
    def _load_to_explore_nodes(
        cls,
        value: Any,
    ) -> SortedDict[tuple[str, str], GraphTraceNodeRecord]:
        """Load the to-explore frontier into a sorted runtime mapping.

        Args:
            value (`Any`):
                A mapping, list of records, or list of serialized records.

        Returns:
            `SortedDict[tuple[str, str], GraphTraceNodeRecord]`:
                Sorted frontier records keyed by priority.
        """
        if value is None:
            return SortedDict()

        # Rebuild even the existing input is a sorted container because deserialized state may wrap
        # plain dictionaries inside a sorted container rather than record objects.
        raw_records = value.values() if isinstance(value, dict) else value
        records = [
            record
            if isinstance(record, GraphTraceNodeRecord)
            else GraphTraceNodeRecord.model_validate(record)
            for record in raw_records
        ]
        return SortedDict((record.to_explore_key(), record) for record in records)

    @field_validator("exploration_history", mode="before")
    @classmethod
    def _load_exploration_history(
        cls,
        value: Any,
    ) -> OrderedDict[str, GraphTraceNodeRecord]:
        """Load explored nodes into an insertion-ordered mapping.

        Args:
            value (`Any`):
                A mapping, list of records, or list of serialized records.

        Returns:
            `OrderedDict[str, GraphTraceNodeRecord]`:
                Explored records keyed by full node identifier.
        """
        if value is None:
            return OrderedDict()

        raw_records = value.values() if isinstance(value, dict) else value
        records = [
            record
            if isinstance(record, GraphTraceNodeRecord)
            else GraphTraceNodeRecord.model_validate(record)
            for record in raw_records
        ]
        return OrderedDict((record.full_node_id, record) for record in records)

    @field_serializer("to_explore_nodes")
    def _dump_to_explore_nodes(
        self,
        value: SortedDict[tuple[str, str], GraphTraceNodeRecord],
    ) -> list[dict[str, Any]]:
        """Serialize the sorted frontier into JSON-native record lists.

        Args:
            value (`SortedDict[tuple[str, str], GraphTraceNodeRecord]`):
                Sorted frontier records.

        Returns:
            `list[dict[str, Any]]`:
                Serialized graph trace node records.
        """
        return [record.model_dump() for record in value.values()]

    def model_post_init(self, context: Any) -> None:
        """Rebuild private lookup indexes after Pydantic validation.

        Args:
            context (`Any`):
                Additional context passed by Pydantic.
        """
        self._to_explore_keys_by_id = {
            record.full_node_id: key
            for key, record in self.to_explore_nodes.items()
        }

    def get_exploring_node(self) -> GraphTraceNodeRecord | None:
        """Return the currently exploring node from the sorted mapping.

        Returns:
            `GraphTraceNodeRecord | None`:
                The first active node if it is in exploration.
        """
        if not self.to_explore_nodes:
            return None
        first = self.to_explore_nodes.peekitem(0)[1]
        if first.state == "exploring":
            return first
        return None

    def add_to_explore(self, record: GraphTraceNodeRecord) -> None:
        """Add a node to the sorted exploration frontier.

        Args:
            record (`GraphTraceNodeRecord`):
                The node record to add.
        """
        record.state = "to_explore"
        key = record.to_explore_key()
        self.to_explore_nodes[key] = record
        self._to_explore_keys_by_id[record.full_node_id] = key

    def start_next_exploration(self) -> GraphTraceNodeRecord | None:
        """Mark the earliest active node as exploring.

        If there is a node already being explored, return it.

        Returns:
            `GraphTraceNodeRecord | None`:
                The node now being explored, if any.
        """
        exploring_node = self.get_exploring_node()
        if exploring_node is not None:
            return exploring_node
        if not self.to_explore_nodes:
            return None
        node = self.to_explore_nodes.peekitem(0)[1]
        node.state = "exploring"
        return node

    def remove_to_explore(self, full_node_id: str) -> GraphTraceNodeRecord | None:
        """Remove a node from the to-explore frontier.

        Args:
            full_node_id (`str`):
                Full node identifier to remove.

        Returns:
            `GraphTraceNodeRecord | None`:
                The removed node, if it existed.
        """
        key = self._to_explore_keys_by_id.pop(full_node_id, None)
        if key is None:
            return None
        return self.to_explore_nodes.pop(key)

    def get_to_explore(self, full_node_id: str) -> GraphTraceNodeRecord | None:
        """Return a to-explore node by identifier.

        Args:
            full_node_id (`str`):
                Full node identifier to look up.

        Returns:
            `GraphTraceNodeRecord | None`:
                The matching record, if any.
        """
        key = self._to_explore_keys_by_id.get(full_node_id)
        if key is None:
            return None
        return self.to_explore_nodes[key]

    def has_explored(self, full_node_id: str) -> bool:
        """Check whether a node already appears in exploration history.

        Args:
            full_node_id (`str`):
                Full node identifier to check.

        Returns:
            `bool`:
                Whether the node has already been explored.
        """
        return full_node_id in self.exploration_history

    def finish_exploring_node(self, outcome: str) -> GraphTraceNodeRecord:
        """Move the exploring node into exploration history.

        Args:
            outcome (`str`):
                The exploration outcome for the node.

        Returns:
            `GraphTraceNodeRecord`:
                The history record that is appended.
        """
        node = self.get_exploring_node()
        if node is None:
            raise ValueError("There is no graph trace node in exploration.")

        key = self._to_explore_keys_by_id.pop(node.full_node_id)
        self.to_explore_nodes.pop(key)
        node.finish(outcome)
        self.exploration_history[node.full_node_id] = node
        return node
    
    @property
    def frontier_size(self) -> int:
        """Return the number of nodes in the exploration frontier."""
        return len(self.to_explore_nodes)

    def finish(self, outcome: str) -> None:
        """Finish the graph trace exploration with the given outcome.

        Args:
            outcome (`str`):
                The final outcome of the graph trace exploration.
        """
        self.finished_at = _get_timestamp()
        self.outcome = outcome

    def to_markdown(self) -> str:
        """Convert graph trace state to Markdown.

        Returns:
            `str`:
                A Markdown representation of the graph trace state.
        """
        to_explore_nodes = self.to_explore_nodes.values()
        exploring_node = self.get_exploring_node()
        to_explore_text = (
            "\n".join(record.to_markdown() for record in to_explore_nodes)
            if to_explore_nodes
            else "There are no variable nodes to be explored."
        )
        exploring_text = (
            f"Note that the variable node `{exploring_node.full_node_id}` is being explored."
            if exploring_node is not None
            else "There is no variable node in exploration."
        )

        lines = [
            "# Graph Trace State",
            "",
            "## To-Explore Nodes",
            to_explore_text,
            "",
            "## Exploring Node",
            exploring_text,
        ]
        if self.outcome:
            lines.extend(["", "## Outcome", self.outcome])
            lines.append(
                f"The execution graph exploration process is finished at `{self.finished_at}`."
            )
        return "\n".join(lines)
