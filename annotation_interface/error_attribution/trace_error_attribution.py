import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from smartcomment.runtime.network import ExecNetwork
from smartcomment.runtime.errors import ExecNetworkKeyError
from smartcomment.runtime.graph import (
    RuntimeGraph,
    RuntimeVariable,
    RuntimeEdge,
    RuntimeOp
)
from data_engine.qa_linkage import (
    LinkedEdgeRecord,
    LinkedNodeRecord,
    build_edge_indices,
    resolve_query_linkage,
    should_skip_query_node,
)

from .inference_utils.trace_operators import (
    TraceAnnotationChecker,
    TraceAnnotationCheckResult,
    TraceConstructionChecker,
    TraceConstructionCheckResult,
    TraceRetrievalChecker,
    TraceRetrievalCheckResult,
    TraceResponseChecker,
    TraceResponseCheckResult,
    get_trace_response_format,
)


@dataclass
class TraceCase:
    query_node: RuntimeVariable
    prediction_node: Optional[RuntimeVariable]
    golden_node: Optional[RuntimeVariable]
    answer_op: Optional[RuntimeOp]
    judge_op: Optional[RuntimeOp]
    search_op: Optional[RuntimeOp]


def collect_failed_cases(network: ExecNetwork) -> List[TraceCase]:
    """Collect all failed QA cases from one execution network.

    Args:
        network (`ExecNetwork`):
            Smartcomment execution network loaded from one graph JSON file.

    Returns:
        `list[TraceCase]`:
            Failed QA cases with linked query, prediction, golden answer, and
            stage-specific operations.
    """
    query_nodes = network.search_variables(class_name="query", category="message & query")
    if not query_nodes:
        return []

    all_edges = network.get_all_edges()
    runtime_node_by_id: Dict[str, RuntimeVariable] = {}
    normalized_node_by_id: Dict[str, LinkedNodeRecord] = {}
    normalized_edges: List[LinkedEdgeRecord] = []

    def _ensure_runtime_node(node_id: str) -> None:
        """Cache one runtime variable and its normalized view when resolvable.

        Args:
            node_id (`str`):
                Full node ID to resolve from the runtime network.

        Returns:
            `None`:
                This function mutates the local node caches in place.
        """
        if node_id in normalized_node_by_id:
            return
        node = network.get_variable(node_id)
        if node is None:
            return
        runtime_node_by_id[node_id] = node
        normalized_node_by_id[node_id] = LinkedNodeRecord(
            node_id=node.full_node_id,
            class_name=str(node.class_name or ""),
            category=str(node.category or ""),
            value=node.value,
            full_name=str(node.full_name or "") or None,
            metadata=node.metadata if isinstance(node.metadata, dict) else {},
        )

    for edge in all_edges:
        normalized_edges.append(
            LinkedEdgeRecord(
                source_id=edge.source_full_node_id,
                target_id=edge.target_full_node_id,
                op_id=edge.op_id,
            )
        )
        _ensure_runtime_node(edge.source_full_node_id)
        _ensure_runtime_node(edge.target_full_node_id)
    edges_by_source, edges_by_target = build_edge_indices(normalized_edges)

    cases: List[TraceCase] = []
    for query_node in query_nodes:
        _ensure_runtime_node(query_node.full_node_id)
        normalized_query = normalized_node_by_id.get(query_node.full_node_id)
        if normalized_query is None or should_skip_query_node(normalized_query):
            continue
        linkage = resolve_query_linkage(normalized_query, edges_by_source, edges_by_target, normalized_node_by_id)
        if linkage.judge_score_text is None or linkage.judge_score_text == "1.0":
            continue

        prediction_node = runtime_node_by_id.get(linkage.prediction_node_id or "")
        golden_node = runtime_node_by_id.get(linkage.golden_node_id or "")
        answer_op = network.get_operation(linkage.answer_op_id) if linkage.answer_op_id else None
        judge_op = network.get_operation(linkage.judge_op_id) if linkage.judge_op_id else None
        search_op = network.get_operation(linkage.search_op_id) if linkage.search_op_id else None

        cases.append(
            TraceCase(
                query_node=query_node,
                prediction_node=prediction_node,
                golden_node=golden_node,
                answer_op=answer_op,
                judge_op=judge_op,
                search_op=search_op
            )
        )

    return cases


def _resolve_source_evidence_full_names(query_node: RuntimeVariable) -> List[str]:
    """Return source-evidence full names recorded on one query node.

    Args:
        query_node (`RuntimeVariable`):
            Query variable selected for attribution.

    Returns:
        `list[str]`:
            Full names listed in the query metadata evidence field.
    """
    metadata = query_node.metadata or {}
    evidences = metadata.get("evidence")
    return [str(ev) for ev in evidences] if isinstance(evidences, list) else []


def resolve_latest_nodes_by_full_name(network: ExecNetwork, full_names: List[str]) -> List[RuntimeVariable]:
    """Resolve the latest node version for each full name.

    Args:
        network (`ExecNetwork`):
            Smartcomment execution network loaded from one graph JSON file.
        full_names (`list[str]`):
            Full names that should be resolved to latest node versions.

    Returns:
        `list[RuntimeVariable]`:
            Latest runtime variables that could be resolved successfully.
    """
    resolved_nodes: List[RuntimeVariable] = []
    for full_name in full_names:
        try:
            resolved_nodes.append(network.get_latest_variable(full_name))
        except ExecNetworkKeyError:
            continue
    return resolved_nodes
    

def _source_evidence_nodes_to_text(nodes: List[RuntimeVariable]) -> str:
    """Serialize source evidence nodes into prompt text.

    Args:
        nodes (`list[RuntimeVariable]`):
            Source evidence runtime variables resolved from query metadata.

    Returns:
        `str`:
            Prompt-ready text block describing the source evidence nodes.
    """
    if not nodes:
        return "[NO SOURCE EVIDENCE NODES FOUND]"
    chunks = []
    for idx, node in enumerate(nodes):
        chunks.append(
            f"### Source Evidence {idx + 1}\n"
            f"full_name: {node.full_name}\n"
            f"full_node_id: {node.full_node_id}\n"
            f"created_at: {node.created_at}\n"
            f"value: {node.value}\n"
            f"metadata: {json.dumps(node.metadata, ensure_ascii=False)}"
        )
    return "\n\n".join(chunks)


def _find_next_message_created_at(source_message_node: RuntimeVariable, session_graph: RuntimeGraph) -> Optional[str]:
    """Return the timestamp of the next message in the same session graph.

    Args:
        source_message_node (`RuntimeVariable`):
            Source message node whose next message boundary is required.
        session_graph (`RuntimeGraph`):
            Session-filtered runtime graph containing the source node.

    Returns:
        `str | None`:
            Timestamp of the next message in the session, or `None` when the
            source message is already the last one.
    """
    message_nodes = [
        n for n in session_graph.nodes
        if n.category == "message"
    ]
    message_nodes.sort(key=lambda n: n.created_at)
    for idx, node in enumerate(message_nodes):
        if node.full_node_id == source_message_node.full_node_id:
            if idx + 1 < len(message_nodes):
                return message_nodes[idx + 1].created_at
            return None
    return None

def build_construction_subgraph(network: ExecNetwork, session_graph: RuntimeGraph, source_evidence_node: RuntimeVariable) -> RuntimeGraph:
    """Build the construction-stage subgraph for one source evidence node.

    Args:
        network (`ExecNetwork`):
            Smartcomment execution network loaded from one graph JSON file.
        session_graph (`RuntimeGraph`):
            Session-filtered runtime graph that contains the source evidence node.
        source_evidence_node (`RuntimeVariable`):
            Source evidence runtime variable selected for construction analysis.

    Returns:
        `RuntimeGraph`:
            Unioned operation subgraph covering the construction window for the evidence.
    """
    next_message_created_at = _find_next_message_created_at(source_evidence_node, session_graph)

    candidate_edges_in_window = network.search_edges(
        session_ids=source_evidence_node.session_id,
        start_time=source_evidence_node.created_at,
        end_time=next_message_created_at,
    )
    candidate_op_ids = {
        edge.op_id
        for edge in candidate_edges_in_window
    }

    ops_in_window = network.search_operations(
        session_ids=source_evidence_node.session_id,
        start_time=None,
        end_time=None,
    )
    ops_in_window = [
        op for op in ops_in_window
        if op.op_id in candidate_op_ids
    ]

    construction_subgraph = None
    for op in ops_in_window:
        op_subgraph = network.filter_by_operation(op.op_id)
        construction_subgraph = (
            op_subgraph
            if construction_subgraph is None
            else (construction_subgraph | op_subgraph)
        )

    return construction_subgraph

def build_retrieval_subgraph(network: ExecNetwork, case: TraceCase) -> RuntimeGraph:
    """Build the retrieval-stage subgraph for one failed QA case.

    Args:
        network (`ExecNetwork`):
            Smartcomment execution network loaded from one graph JSON file.
        case (`TraceCase`):
            Failed QA case selected for attribution.

    Returns:
        `RuntimeGraph`:
            Retrieval-stage subgraph used by the attribution prompt.
    """
    if case.search_op is None:
        query_id = case.query_node.full_node_id
        query_session_id = case.query_node.session_id

        all_edges = network.get_all_edges()
        edges_by_source: Dict[str, List[RuntimeEdge]] = defaultdict(list)
        for edge in all_edges:
            if query_session_id and edge.session_id != query_session_id:
                continue
            edges_by_source[edge.source_full_node_id].append(edge)

        for source_id in edges_by_source:
            edges_by_source[source_id].sort(key=lambda e: e.created_at)

        def _is_memory_unit(node: Optional[RuntimeVariable]) -> bool:
            """Return whether one node should be treated as a memory entry hit.

            Args:
                node (`Optional[RuntimeVariable]`):
                    Runtime variable candidate reached during retrieval traversal.

            Returns:
                `bool`:
                    Whether the node should count as a retrieval memory unit.
            """
            if node is None:
                return False
            if node.category == "memory_entry":
                return True
            return "memory-unit" in node.full_name

        queue: List[str] = [query_id]
        visited_nodes: set[str] = {query_id}

        first_query_retrieval_edge_time: Optional[str] = None
        memory_unit_hit_times: List[str] = []

        while queue:
            current = queue.pop(0)
            for edge in edges_by_source.get(current, []):
                if (edge.source_full_node_id == query_id and first_query_retrieval_edge_time is None):
                    first_query_retrieval_edge_time = edge.created_at

                target = network.get_variable(edge.target_full_node_id)
                if _is_memory_unit(target):
                    memory_unit_hit_times.append(edge.created_at)
                    continue

                if edge.target_full_node_id not in visited_nodes:
                    visited_nodes.add(edge.target_full_node_id)
                    queue.append(edge.target_full_node_id)

        earliest_memory_unit_edge_time: Optional[str] = None
        if first_query_retrieval_edge_time is not None:
            valid_hit_times = [
                t for t in memory_unit_hit_times
                if t >= first_query_retrieval_edge_time
            ]
            if valid_hit_times:
                earliest_memory_unit_edge_time = min(valid_hit_times)

        if (
            first_query_retrieval_edge_time is not None
            and earliest_memory_unit_edge_time is not None
        ):
            # Select operations by first filtering edges in the time window,
            # then lifting to op_ids. This avoids dropping boundary ops whose
            # op.created_at is earlier than the first retrieval edge.
            candidate_edges_in_window = network.search_edges(
                session_ids=query_session_id,
                start_time=first_query_retrieval_edge_time,
                end_time=earliest_memory_unit_edge_time,
            )
            candidate_op_ids = {
                edge.op_id
                for edge in candidate_edges_in_window
            }

            if candidate_op_ids:
                ops_in_window = network.search_operations(
                    session_ids=query_session_id,
                    start_time=None,
                    end_time=None,
                )
                ops_in_window = [
                    op for op in ops_in_window
                    if op.op_id in candidate_op_ids
                ]

                retrieval_subgraph = None
                for op in ops_in_window:
                    op_subgraph = network.filter_by_operation(op.op_id)
                    retrieval_subgraph = (
                        op_subgraph
                        if retrieval_subgraph is None
                        else (retrieval_subgraph | op_subgraph)
                    )
                return retrieval_subgraph
    return network.filter_by_operation(case.search_op.op_id)

def build_response_subgraph(network: ExecNetwork, case: TraceCase) -> RuntimeGraph:
    """Build the response-stage subgraph for one failed QA case.

    Args:
        network (`ExecNetwork`):
            Smartcomment execution network loaded from one graph JSON file.
        case (`TraceCase`):
            Failed QA case selected for attribution.

    Returns:
        `RuntimeGraph`:
            Unioned answer and judge operation subgraph.
    """
    if case.answer_op is not None:
        answer_subgraph = network.filter_by_operation(case.answer_op.op_id)

    if case.judge_op is not None:
        judge_subgraph = network.filter_by_operation(case.judge_op.op_id)

    return answer_subgraph | judge_subgraph
class TraceErrorAttributionRunnerConfig(BaseModel):
    """Configuration for trace-based error attribution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_path: str = Field(
        ...,
        description="Path to graph_evaluation.json."
    )
    judge_model: str = Field(
        default="gpt-4o-mini",
        description="Model name used for LLM judging.",
    )
    batch_size: int = Field(
        default=4,
        description="Batch size for LLM inference.",
    )
    api_config_path: str | None = Field(
        default=None,
        description="Path to the API config file (keys and base URLs).",
    )
    interface_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Additional keyword arguments forwarded to the LLM interface. "
            "If provided, it takes precedence over ``api_config_path``."
        ),
    )
    include_metadata_in_graph_text: bool = Field(
        default=True,
        description="Whether to include metadata when serializing subgraphs.",
    )
    output_path: str | None = Field(
        default=None,
        description=(
            "Path to save the attribution results JSON file. "
            "Defaults to the graph path with an '_attribution' suffix."
        ),
    )


class TraceErrorAttributionRunner:
    """Trace-based error attribution runner built on smartcomment execution graph."""

    def __init__(self, config: TraceErrorAttributionRunnerConfig) -> None:
        """Initialize the attribution runner.

        Args:
            config (`TraceErrorAttributionRunnerConfig`):
                Runner configuration that defines graph paths, LLM settings,
                and output behavior.

        Returns:
            `None`:
                This method stores configuration and initializes graph caches.
        """
        self.config = config
        self._network: Optional[ExecNetwork] = None
        self._session_graph_cache: Dict[str, RuntimeGraph] = {}
        self._checkers: Dict[str, Any] | None = None


    def _resolve_interface_kwargs(self) -> dict[str, Any]:
        """Return the LLM interface keyword arguments.

        Args:
            None.

        Returns:
            `dict[str, Any]`:
                Interface keyword arguments for LLM checker construction.
        """
        if self.config.interface_kwargs is not None:
            return self.config.interface_kwargs
        if self.config.api_config_path is not None:
            with open(self.config.api_config_path, "r", encoding="utf-8") as f:
                api_config = json.load(f)
            return {
                "api_keys": api_config["api_keys"],
                "base_urls": api_config["base_urls"],
            }
        if os.environ.get("OPENAI_API_KEY") is not None:
            return {
                "api_keys": [os.environ["OPENAI_API_KEY"]],
                "base_urls": [os.environ.get("OPENAI_API_BASE")],
            }
        return {}


    def _resolve_output_path(self) -> str:
        """Derive the output file path for the attribution results.

        Args:
            None.

        Returns:
            `str`:
                Configured output path, or a default path derived from the graph path.
        """
        return self.config.output_path or (
            self.config.graph_path.rsplit(".", 1)[0] + "_attribution.json"
        )


    def _load_graph(self) -> ExecNetwork:
        """Load and cache the smartcomment execution network.

        Args:
            None.

        Returns:
            `ExecNetwork`:
                Execution network imported from `self.config.graph_path`.
        """
        if self._network is not None:
            return self._network

        with open(self.config.graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        self._network = ExecNetwork.import_graph(graph_data)
        return self._network
    

    def _get_session_graph(self, session_id: str) -> RuntimeGraph:
        """Return and cache one session-filtered runtime graph.

        Args:
            session_id (`str`):
                Session identifier to filter from the full execution network.

        Returns:
            `RuntimeGraph`:
                Cached session-filtered runtime graph.
        """
        if session_id not in self._session_graph_cache:
            self._session_graph_cache[session_id] = self._load_graph().filter_by_session(session_id)
        return self._session_graph_cache[session_id]


    def prepare_single_case(self, network: ExecNetwork, case: TraceCase) -> Dict[str, Any]:
        """Convert one failed trace case into LLM attribution inputs.

        Args:
            network (`ExecNetwork`):
                Smartcomment execution network loaded from the target graph.
            case (`TraceCase`):
                Failed QA case that should be transformed into attribution inputs.

        Returns:
            `Dict[str, Any]`:
                Serialized attribution inputs for annotation, construction,
                retrieval, and response checks.
        """
        query = case.query_node.value
        prediction = case.prediction_node.value if case.prediction_node else ""
        golden_answer = case.golden_node.value if case.golden_node else ""

        source_evidence_full_names = _resolve_source_evidence_full_names(case.query_node)
        source_evidence_nodes = resolve_latest_nodes_by_full_name(network, source_evidence_full_names)
        source_evidences_text = _source_evidence_nodes_to_text(source_evidence_nodes)

        retrieval_subgraph = build_retrieval_subgraph(network, case)
        response_subgraph = build_response_subgraph(network, case)

        construction_inputs = []
        for node in source_evidence_nodes:
            session_graph = self._get_session_graph(node.session_id)
            construction_subgraph = build_construction_subgraph(network, session_graph, node)
            construction_op_ids = sorted({op.op_id for op in construction_subgraph.ops})
            construction_inputs.append({
                "source_evidence_full_name": node.full_name,
                "construction_subgraph_text": construction_subgraph.to_xml(
                    include_metadata=self.config.include_metadata_in_graph_text
                ),
                "candidate_op_ids": "\n".join(construction_op_ids) if construction_op_ids else "[NO OP IDS]",
            })

        retrieval_op_ids = sorted({op.op_id for op in retrieval_subgraph.ops})
        response_op_ids = sorted({op.op_id for op in response_subgraph.ops})

        return {
            "query": query,
            "query_node": {
                "full_name": case.query_node.full_name,
                "full_node_id": case.query_node.full_node_id,
                "session_id": case.query_node.session_id,
            },
            "golden_answer": golden_answer,
            "prediction": prediction,
            "source_evidence_full_names": source_evidence_full_names,
            "source_evidences_text": source_evidences_text,
            "annotation_input": {
                "question": query,
                "golden_answer": golden_answer,
                "source_evidences_text": source_evidences_text,
            },
            "construction_inputs": construction_inputs,
            "retrieval_input": {
                "question": query,
                "golden_answer": golden_answer,
                "source_evidences_text": source_evidences_text,
                "retrieval_subgraph_text": retrieval_subgraph.to_xml(
                    include_metadata=self.config.include_metadata_in_graph_text
                ),
                "candidate_op_ids": "\n".join(retrieval_op_ids) if retrieval_op_ids else "[NO OP IDS]",
            },
            "response_input": {
                "question": query,
                "golden_answer": golden_answer,
                "prediction": prediction,
                "response_subgraph_text": response_subgraph.to_xml(
                    include_metadata=self.config.include_metadata_in_graph_text
                ),
                "candidate_op_ids": "\n".join(response_op_ids) if response_op_ids else "[NO OP IDS]",
            },
        }
    

    def build_trace_inputs_before_llm(self) -> List[Dict[str, Any]]:
        """Construct LLM input payloads for all failed QA cases.

        Args:
            None.

        Returns:
            `List[Dict[str, Any]]`:
                Prepared attribution payloads for every failed case in the graph.
        """
        network = self._load_graph()
        cases = collect_failed_cases(network)
        return [self.prepare_single_case(network, case) for case in cases]


    def _build_checkers(self) -> Dict[str, Any]:
        """Construct the LLM checker objects used by the attribution pipeline.

        Args:
            None.

        Returns:
            `dict[str, Any]`:
                Mapping from stage name to initialized checker instance.
        """
        if self._checkers is not None:
            return self._checkers

        interface_kwargs = self._resolve_interface_kwargs()

        annotation_checker = TraceAnnotationChecker(
            prompt_name="trace-annotation-check",
            model_name=self.config.judge_model,
            **interface_kwargs,
        )

        construction_checker = TraceConstructionChecker(
            prompt_name="trace-construction-check",
            model_name=self.config.judge_model,
            **interface_kwargs,
        )

        retrieval_checker = TraceRetrievalChecker(
            prompt_name="trace-retrieval-check",
            model_name=self.config.judge_model,
            **interface_kwargs,
        )

        response_checker = TraceResponseChecker(
            prompt_name="trace-response-check",
            model_name=self.config.judge_model,
            **interface_kwargs,
        )

        self._checkers = {
            "annotation": annotation_checker,
            "construction": construction_checker,
            "retrieval": retrieval_checker,
            "response": response_checker,
        }
        return self._checkers


    def run_single_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run full attribution over one prepared failed-case payload.

        Args:
            case (`dict[str, Any]`):
                Prepared attribution input returned by `prepare_single_case`.

        Returns:
            `dict[str, Any]`:
                Attribution result including error type, details, and original inputs.
        """
        checkers = self._build_checkers()
        query = case["query"]
        golden_answer = case["golden_answer"]
        prediction = case["prediction"]

        annotation = checkers["annotation"](
            [query],
            [golden_answer],
            [case["annotation_input"]["source_evidences_text"]],
            batch_size=1,
            aggregate=True,
            temperature=0.0,
            response_format=get_trace_response_format(TraceAnnotationCheckResult),
        )[0]
        if annotation["is_annotation_error"]:
            return {
                "query": query,
                "golden_answer": golden_answer,
                "prediction": prediction,
                "error_type": "annotation_error",
                "error_details": annotation,
                "llm_inputs": case,
            }

        construction_error = None
        if case["construction_inputs"]:
            question_list = [query] * len(case["construction_inputs"])
            golden_answer_list = [golden_answer] * len(case["construction_inputs"])
            source_evidence_list = [c["source_evidence_full_name"] for c in case["construction_inputs"]]
            construction_subgraph_list = [c["construction_subgraph_text"] for c in case["construction_inputs"]]
            candidate_op_ids_list = [c["candidate_op_ids"] for c in case["construction_inputs"]]

            construction_results = checkers["construction"](
                question_list,
                golden_answer_list,
                source_evidence_list,
                candidate_op_ids_list,
                construction_subgraph_list,
                batch_size=self.config.batch_size,
                aggregate=True,
                temperature=0.0,
                response_format=get_trace_response_format(TraceConstructionCheckResult),
            )

            for cons, cons_result in zip(case["construction_inputs"], construction_results):
                if cons_result["is_construction_error"]:
                    construction_error = {
                        **cons_result,
                        "source_evidence_full_name": cons["source_evidence_full_name"],
                    }
                    break

        if construction_error is not None:
            return {
                "query": query,
                "golden_answer": golden_answer,
                "prediction": prediction,
                "error_type": "memory_construction_error",
                "error_details": construction_error,
                "llm_inputs": case,
            }

        retrieval = checkers["retrieval"](
            [query],
            [golden_answer],
            [case["retrieval_input"]["source_evidences_text"]],
            [case["retrieval_input"]["retrieval_subgraph_text"]],
            [case["retrieval_input"]["candidate_op_ids"]],
            batch_size=1,
            aggregate=True,
            temperature=0.0,
            response_format=get_trace_response_format(TraceRetrievalCheckResult),
        )[0]
        if retrieval["is_retrieval_error"]:
            return {
                "query": query,
                "golden_answer": golden_answer,
                "prediction": prediction,
                "error_type": "retrieval_error",
                "error_details": retrieval,
                "llm_inputs": case,
            }

        response = checkers["response"](
            [query],
            [golden_answer],
            [prediction],
            [case["response_input"]["response_subgraph_text"]],
            [case["response_input"]["candidate_op_ids"]],
            batch_size=1,
            aggregate=True,
            temperature=0.0,
            response_format=get_trace_response_format(TraceResponseCheckResult),
        )[0]
        return {
            "query": query,
            "golden_answer": golden_answer,
            "prediction": prediction,
            "error_type": "response_error",
            "error_details": response,
            "llm_inputs": case,
        }
    
    def run(self) -> List[Dict[str, Any]]:
        """Run attribution over all failed QA cases and save the results.

        Args:
            None.

        Returns:
            `list[dict[str, Any]]`:
                Attribution results for every failed QA case in the graph.
        """
        prepared_cases = self.build_trace_inputs_before_llm()
        results = [self.run_single_case(case_dict) for case_dict in prepared_cases]

        output_path = self._resolve_output_path()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results
