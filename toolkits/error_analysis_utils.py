import asyncio
import json
import time
import os 
import warnings
from collections import defaultdict
from pathlib import Path
import tiktoken
from pydantic import BaseModel, Field
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from graphtrace import (
    ChatUsageTokenMonitor,
    StudioServer,
    agentscope_token_monitor,
)
from smartcomment.runtime import ExecNetwork
from smartcomment.runtime.errors import ExecNetworkKeyError 
from .bench_utils import FailedQueryCase
from ._base import AgentBaseConfig, AgentBaseRunner
from .memtrace_utils import ErrorAttributionPrediction


DEFAULT_TARGET_SYSTEM_OVERVIEW = (
    "The target system is a workflow composed of one or more operations whose "
    "internal behavior is captured by an execution graph. Each operation "
    "consumes some input variables and produces output variables. The system "
    "may produce incorrect final outputs for a variety of reasons, ranging "
    "from upstream input handling to downstream output generation."
)


SYSTEM_PROMPT_TEMPLATE = """You are an expert failure analyst tasked with iteratively producing a high-quality error analysis report for a target system.

## Target System Overview

{target_system_overview}
"""


REPORT_INSTRUCTION_TEMPLATE = """# Instruction

Your task is to iteratively improve an error analysis report based on a batch of failed cases.

Inputs you receive:
- The current version of the error analysis report.
- A batch of failed cases. Each failed case contains error type, attributed operation identifier, error reason, XML rendering of the operation graph around the attributed operation. In addition, each failed case is paired with the original user question, golden answer, model prediction, and corresponding source evidence.

Goal:
- Improve the current error analysis report so it better summarizes, categorizes, and analyzes the failure patterns reflected in the provided failed cases.
- The report should evolve iteratively across batches. Each batch may refine, expand, reorganize, or correct the existing report.

How to analyze failed cases:
- Use the error type and attributed reason to identify recurring failure patterns.
- Carefully read the operation graph and the failed case information, including the user question, golden answer, model prediction, and source evidence, to understand the underlying error type, failure mechanism, and attributed error reason.
- Pay attention to whether some existing error categories should be refined into more specific subtypes.

How to read the XML operation graph:
- The graph contains variables, edges, and operations.
- Variable comments describe what each variable represents, its intended role, and any local constraints on its value.
- Edge comments describe how information is transferred, transformed, filtered, or constrained between variables.
- Operation comments describe the operation's role, responsibility, and constraints in the workflow.
- Within one operation graph, variables with in-degree 0 relative to that graph are the operation's input variables.
- Variables with in-degree >= 1 relative to that graph are variables produced by the operation.
- Variables with out-degree 0 relative to that graph are the operation's final output variables.
- Use the operation name, category, comment, variable comments, edge comments, variable values, and produced variable roles to understand what the operation is supposed to do and how it fails.

Report writing requirements:
- The report should follow a high-level analytical writing style suitable for a research or technical report. Structure the report with an introductory overview, a systematic analysis of major error categories, and a concluding discussion of the broader findings and implications. Begin with a concise overview summarizing the purpose of the analysis and the major observed failure patterns. Group related failures into coherent error categories, and for each category, explain the defining characteristics of the errors, describe representative failure patterns, analyze the likely underlying causes, and discuss important recurring behaviors, trade-offs, or limitations revealed by the failed cases.
- If appropriate, refine broad error categories into more specific subcategories when meaningful distinctions emerge from the cases. Avoid redundancy and repetitive descriptions across categories.
- The report should be written as a sequence of coherent paragraphs rather than a heavily sectioned or bullet-pointed document. Use one paragraph for the introductory overview, one paragraph for each major error category, and one concluding paragraph summarizing the overall findings and implications.
- The generated report should read as a finalized standalone analysis document. Do not mention the iterative optimization process, batching procedure, or the fact that the report is being updated across multiple rounds of failed cases. Avoid phrases such as ''in this batch'', ''based on the current failed cases'', ''the 26-th failed cases'', or similar process-oriented descriptions.
- The report should have a level of detail and overall length similar to the provided example report. Focus on synthesizing representative failure patterns rather than exhaustively discussing every failed case. If subcategories are introduced within a major error category, describe them naturally within the same paragraph as the main category instead of creating deeply nested analyses or overly detailed breakdowns. 
- The analysis should remain concise, readable, and easy for readers to follow. If you find that the current report is becoming too long, condense it by abstracting and summarizing the common patterns across the failed cases. For example, you may summarize multiple related failures by stating that many cases reflect missing intermediate reasoning steps.


## Example Report 
We conduct a detailed analysis of errors made by GPT-4V to better understand its operational capabilities and limitations. This analysis aims not only to identify the model’s current shortcomings but also to provide insights for improving future model design and training. We examine a collection of failed prediction cases and analyze their root causes based on the model outputs, golden answers, and supporting evidence. The analysis reveals several recurring categories of failures, each reflecting different limitations in perception, knowledge, reasoning, and multimodal understanding.
Perceptual Errors: Perceptual errors form one of the largest categories of inaccuracies in GPT-4V. These errors can be divided into basic perceptual errors and domain-specific perceptual errors. Basic perceptual errors occur when the model generally understands the provided information but fails in elementary visual interpretation, such as incorrectly interpreting spatial layouts or sequential ordering. Domain-specific perceptual errors arise when the model lacks sufficient domain knowledge to correctly interpret visual elements within specialized contexts. In addition, the model often exhibits a bias toward textual information, prioritizing text-based interpretations over visual evidence. This tendency can lead the model to ignore or misinterpret important visual cues, highlighting the need for a more balanced multimodal understanding capability.
Lack of Knowledge: Another major source of failure is the lack of specialized or domain-specific knowledge. In many cases, the model can correctly recognize visual or textual elements but fails to interpret their meaning within the appropriate domain context. This limitation affects both perception and reasoning. For example, the model may identify relevant symbols or structures but fail to understand their technical significance in fields such as computer science or medicine. These failures suggest that improving the breadth and depth of domain-specific knowledge remains important for enhancing the general applicability and reliability of foundation models.
Reasoning Errors: Flawed reasoning is another significant contributor to incorrect predictions. Even when the model correctly interprets the input and recalls relevant knowledge, it may still fail to apply logical or mathematical reasoning effectively. Common failure patterns include missing intermediate reasoning steps, applying incorrect assumptions, or failing to maintain consistency across long reasoning chains. These observations indicate that improving reasoning robustness and multi-step inference capabilities remains a critical challenge.
Overall, the error analysis highlights several important limitations in current multimodal models. First, the interaction between language and vision can both improve interpretability and introduce hallucinations or incorrect biases. Second, grounding and referring to specific elements within visual inputs remain challenging. Third, complex reasoning involving long inference chains or detailed calculations continues to be a major source of failure. Together, these findings emphasize the need for continued improvements in perception, knowledge integration, reasoning, and multimodal grounding.

## Inputs

### Current Error Analysis Report

{current_error_analysis_report}

### Failed Cases

{failed_cases}
"""


EMPTY_REPORT_PLACEHOLDER = (
    "(There is no existing error analysis report yet. This is the first batch "
    "of failed cases, so you are producing the initial version of the report.)"
)


REPORT_FILE = "error_analysis_report.json"


class _ErrorAnalysisReport(BaseModel):
    """Structured output for one updated error analysis report."""

    rationale: str = Field(
        description=(
            "Your thinking process for refining the error analysis report "
            "based on the latest batch of failed cases."
        ),
    )
    report: str = Field(
        description=(
            "The full updated error analysis report after incorporating the "
            "latest batch of failed cases."
        ),
    )


def _render_op_subgraph_xml(graph: ExecNetwork, op_id: str) -> str:
    """Render the operation subgraph XML around an attributed operation.

    Args:
        graph (`ExecNetwork`):
            The execution graph containing the attributed operation.
        op_id (`str`):
            The attributed operation identifier.

    Returns:
        `str`:
            The XML rendering of the attributed operation subgraph. When the
            operation identifier cannot be resolved, an explanatory placeholder
            is returned instead so the report agent still receives a usable
            input.
    """
    subgraph = graph.filter_by_operation(op_id)
    if subgraph.is_empty:
        warnings.warn(
            f"The attributed operation identifier '{op_id}' could not be "
            "resolved to a subgraph in the provided execution graph.", 
            UserWarning,
        )
        return (
            f"(The attributed operation identifier '{op_id}' doesn't " 
            "exist in the provided execution graph. " 
            "This error attribution result is invalid.)"
        )
    return subgraph.to_xml(include_metadata=False)


def _resolve_source_evidence_texts(
    case: FailedQueryCase,
    graph: ExecNetwork,
) -> list[str]:
    """Resolve source-evidence text for one failed case from the graph.

    Args:
        case (`FailedQueryCase`):
            The failed query case.
        graph (`ExecNetwork`):
            The execution graph containing the source-evidence nodes.

    Returns:
        `list[str]`:
            One text per source-evidence node.
    """
    texts = []
    for full_node_id in case.source_evidence_full_node_ids:
        try:
            variable = graph.get_variable(full_node_id)
        except ExecNetworkKeyError:
            warnings.warn(
                f"Source evidence node '{full_node_id}' could not be resolved "
                "in the provided graph.",
                UserWarning,
            )
            continue
        texts.append(variable.value)
    return texts


def _format_source_evidence(texts: list[str]) -> str:
    """Format source-evidence text for one failed case.

    Args:
        texts (`list[str]`):
            Source-evidence text segments.

    Returns:
        `str`:
            Human-readable source-evidence text.
    """
    if not texts:
        return "(No source evidence is available for this case.)"
    chunks = []
    for index, text in enumerate(texts, start=1):
        chunks.append(f"Source Evidence {index}:\n{text}")
    return "\n\n".join(chunks)


def _format_failed_case(
    case_index: int,
    case: FailedQueryCase,
    prediction: ErrorAttributionPrediction,
    op_subgraph_xml: str,
    source_evidence_texts: list[str],
) -> str:
    """Format one failed case for the report-update prompt.

    Args:
        case_index (`int`):
            The 1-based index of the case within the current batch.
        case (`FailedQueryCase`):
            The failed query case.
        prediction (`ErrorAttributionPrediction`):
            The attributed error prediction for this failed case.
        op_subgraph_xml (`str`):
            XML rendering of the attributed operation subgraph.
        source_evidence_texts (`list[str]`):
            Source-evidence text for this failed case.

    Returns:
        `str`:
            Markdown-formatted failed-case text.
    """
    return (
        f"#### Failed Case {case_index}\n"
        f"Error type: {prediction.error_type}\n"
        f"Attributed operation id: {prediction.op_id}\n"
        f"Error reason: {prediction.reason}\n\n"
        f"Question:\n{case.query}\n\n"
        f"Golden answer:\n{case.golden_answer}\n\n"
        f"Model prediction:\n{case.prediction}\n\n"
        f"Source evidence:\n{_format_source_evidence(source_evidence_texts)}\n\n"
        "Erroneous operation's graph:\n"
        f"{op_subgraph_xml}"
    )


class ReportGenerationConfig(AgentBaseConfig):
    """The configuration for error analysis report generation."""

    batch_size: int = Field(
        default=4,
        description="Number of failed cases provided as input to a single LLM call.",
        ge=1,
    )
    max_op_xml_tokens: int = Field(
        default=300_000,
        description=(
            "Maximum number of tokens retained from one attributed operation "
            "subgraph XML. Oversized XML keeps its last tokens."
        ),
        ge=1,
    )


class ErrorAnalysisReportRunner(AgentBaseRunner):
    """Runner that iteratively generates an error analysis report."""

    def __init__(self, config: ReportGenerationConfig | None = None) -> None:
        """Initialize the error analysis report runner.

        Args:
            config (`ReportGenerationConfig | None`, optional):
                The error analysis report runner configuration. If not
                provided, default configuration is used.
        """
        super().__init__(config or ReportGenerationConfig())
        self._token_encoding = tiktoken.get_encoding("o200k_base")

    def _build_agent(
        self,
        sys_prompt: str,
        api_key: str,
        base_url: str,
    ) -> ReActAgent:
        """Build a plain structured-output report-update agent.

        Args:
            sys_prompt (`str`):
                System prompt for the agent.
            api_key (`str`):
                OpenAI-compatible API key.
            base_url (`str`):
                OpenAI-compatible base URL.

        Returns:
            `ReActAgent`:
                Configured report-update agent.
        """
        cfg = self.config
        model = OpenAIChatModel(
            model_name=cfg.model,
            api_key=api_key,
            stream=cfg.stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": cfg.temperature},
        )
        return ReActAgent(
            name="error_analysis_report",
            sys_prompt=sys_prompt,
            model=model,
            formatter=OpenAIChatFormatter(),
        )

    def _resolve_graph_inputs(
        self,
        n_cases: int,
        graph: ExecNetwork | list[ExecNetwork] | None,
        graph_paths: str | Path | list[str | Path] | None,
    ) -> tuple[list[ExecNetwork] | None, list[str] | None]:
        """Validate and normalize graph or graph-path inputs.

        Args:
            n_cases (`int`):
                Number of failed cases to render.
            graph (`ExecNetwork | list[ExecNetwork] | None`):
                One shared execution graph or one graph per case.
            graph_paths (`str | Path | list[str | Path] | None`):
                Path or list of paths to serialized execution graphs.

        Returns:
            `tuple[list[ExecNetwork] | None, list[str] | None]`:
                A tuple containing either eagerly-resolved graphs aligned with
                the failed cases or graph paths aligned with the failed cases.
        """
        if graph is None and graph_paths is None:
            raise ValueError(
                "At least one of `graph` or `graph_paths` must be provided."
            )
        if graph is not None and graph_paths is not None:
            raise ValueError(
                "Please provide either `graph` or `graph_paths`."
            )

        if graph is not None:
            if isinstance(graph, ExecNetwork):
                return [graph] * n_cases, None
            if not isinstance(graph, list):
                raise TypeError(
                    "`graph` must be an `ExecNetwork` or a list of "
                    "`ExecNetwork` instances."
                )
            if len(graph) != n_cases:
                raise ValueError(
                    f"`graph` (current size: {len(graph)}) must have the same "
                    f"size as `failed_cases` (current size: {n_cases})."
                )
            return list(graph), None

        if isinstance(graph_paths, (str, Path)):
            return None, [str(graph_paths)] * n_cases
        if not isinstance(graph_paths, list):
            raise TypeError(
                "`graph_paths` must be a string, a `Path`, or a list of those."
            )
        if len(graph_paths) != n_cases:
            raise ValueError(
                f"`graph_paths` (current size: {len(graph_paths)}) must have the "
                f"same size as `failed_cases` (current size: {n_cases})."
            )
        return None, [str(p) for p in graph_paths]

    def _render_cases(
        self,
        failed_cases: list[FailedQueryCase],
        error_predictions: list[ErrorAttributionPrediction],
        graphs_for_cases: list[ExecNetwork] | None,
        paths_for_cases: list[str] | None,
    ) -> list[str]:
        """Pre-render the per-case input text for the report-update prompt.

        Each case is rendered into a self-contained Markdown chunk that
        includes the attributed error metadata, the question, golden answer,
        model prediction, source-evidence text, and the XML of the attributed
        operation subgraph. When graph paths are provided, graphs are loaded
        lazily.

        Args:
            failed_cases (`list[FailedQueryCase]`):
                The failed query cases to render.
            error_predictions (`list[ErrorAttributionPrediction]`):
                The attributed error predictions.
            graphs_for_cases (`list[ExecNetwork] | None`):
                One shared execution graph or one graph per failed case.
            paths_for_cases (`list[str] | None`):
                Path or list of paths to serialized execution graphs. When
                used, graphs are loaded lazily. Note that either ``graph`` 
                or ``graph_paths`` must be provided.

        Returns:
            `list[str]`:
                One Markdown-formatted chunk per case, aligned with the inputs.
        """
        n = len(failed_cases)
        rendered = [""] * n

        if graphs_for_cases is not None:
            for index in range(n):
                graph = graphs_for_cases[index]
                op_xml = _render_op_subgraph_xml(
                    graph,
                    error_predictions[index].op_id,
                )
                op_xml_tokens = self._token_encoding.encode(op_xml)
                if len(op_xml_tokens) > self.config.max_op_xml_tokens:
                    op_xml = self._token_encoding.decode(
                        op_xml_tokens[-self.config.max_op_xml_tokens:],
                    )
                evidence = _resolve_source_evidence_texts(
                    failed_cases[index],
                    graph,
                )
                rendered[index] = _format_failed_case(
                    case_index=index + 1,
                    case=failed_cases[index],
                    prediction=error_predictions[index],
                    op_subgraph_xml=op_xml,
                    source_evidence_texts=evidence,
                )
            return rendered

        if paths_for_cases is None:
            raise ValueError(
                "Either `graphs_for_cases` or `paths_for_cases` must be provided."
            )

        path_to_indices = defaultdict(list)
        for index, path in enumerate(paths_for_cases):
            path_to_indices[path].append(index)

        for path, indices in path_to_indices.items():
            with open(path, "r", encoding="utf-8") as file:
                graph_data = json.load(file)
            graph = ExecNetwork.import_graph(graph_data)
            try:
                for index in indices:
                    op_xml = _render_op_subgraph_xml(
                        graph,
                        error_predictions[index].op_id,
                    )
                    op_xml_tokens = self._token_encoding.encode(op_xml)
                    if len(op_xml_tokens) > self.config.max_op_xml_tokens:
                        op_xml = self._token_encoding.decode(
                            op_xml_tokens[-self.config.max_op_xml_tokens:],
                        )
                    evidence = _resolve_source_evidence_texts(
                        failed_cases[index],
                        graph,
                    )
                    rendered[index] = _format_failed_case(
                        case_index=index + 1,
                        case=failed_cases[index],
                        prediction=error_predictions[index],
                        op_subgraph_xml=op_xml,
                        source_evidence_texts=evidence,
                    )
            finally:
                del graph
        return rendered

    @staticmethod
    def _summarize_total_cost(
        elapsed_times: list[float],
    ) -> dict[str, float | int]:
        """Summarize total token and wall-clock cost for the report run.

        Because each report-generation run produces a single final report,
        the cost is reported as totals over all batches rather than as
        per-batch averages.

        Args:
            elapsed_times (`list[float]`):
                Per-batch elapsed seconds.

        Returns:
            `dict[str, float | int]`:
                Total token and time cost for the run.
        """
        token_cost = ChatUsageTokenMonitor.to_dict()
        n_events = token_cost["n"]
        avg_input = token_cost["avg_input_tokens"]
        avg_output = token_cost["avg_output_tokens"]
        return {
            "input_tokens": n_events * avg_input,
            "output_tokens": n_events * avg_output,
            "minutes": sum(elapsed_times) / 60,
        }

    async def arun(
        self,
        failed_cases: list[FailedQueryCase],
        error_predictions: list[ErrorAttributionPrediction],
        *,
        target_system_overview: str | None = None,
        graph: ExecNetwork | list[ExecNetwork] | None = None,
        graph_paths: str | Path | list[str | Path] | None = None,
        save_folder: str | Path | None = None,
    ) -> tuple[str, dict[str, float | int]]:
        """Iteratively generate an error analysis report asynchronously.

        Args:
            failed_cases (`list[FailedQueryCase]`):
                Failed query cases used to generate the report.
            error_predictions (`list[ErrorAttributionPrediction]`):
                Attributed error predictions.
            target_system_overview (`str | None`, optional):
                Overview of the target system being diagnosed. The overview
                is embedded in the system prompt so the agent understands the
                method whose failures are being analyzed. When not provided,
                a generic system-agnostic overview is used.
            graph (`ExecNetwork | list[ExecNetwork] | None`, optional):
                One shared execution graph or one graph per failed case.
                Note that either ``graph`` or ``graph_paths`` must be provided.
            graph_paths (`str | Path | list[str | Path] | None`, optional):
                Path or list of paths to serialized execution graphs. When
                used, graphs are loaded lazily. Note that either ``graph`` 
                or ``graph_paths`` must be provided.
            save_folder (`str | Path | None`, optional):
                Folder where the final report is saved. If not provided, the
                report is only returned and not persisted.

        Returns:
            `tuple[str, dict[str, float | int]]`:
                The final error analysis report and the total cost statistics.
        """
        if not failed_cases:
            raise ValueError("No failed cases are provided.")
        if len(failed_cases) != len(error_predictions):
            raise ValueError(
                f"`failed_cases` (current size: {len(failed_cases)}) and " 
                f"`error_predictions` (current size: {len(error_predictions)}) " 
                "must have the same size."
            )

        n_cases = len(failed_cases)
        graphs_for_cases, paths_for_cases = self._resolve_graph_inputs(
            n_cases=n_cases,
            graph=graph,
            graph_paths=graph_paths,
        )
        rendered_cases = self._render_cases(
            failed_cases=failed_cases,
            error_predictions=error_predictions,
            graphs_for_cases=graphs_for_cases,
            paths_for_cases=paths_for_cases,
        )

        cfg = self.config
        batch_size = cfg.batch_size
        api_pool = self._api_pool 

        overview = target_system_overview
        if overview is None:
            overview = DEFAULT_TARGET_SYSTEM_OVERVIEW
        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(target_system_overview=overview)

        if save_folder is not None:
            save_folder = str(save_folder)
            os.makedirs(save_folder, exist_ok=True)

        studio = None
        if cfg.studio_url is not None:
            studio = StudioServer(
                url=cfg.studio_url,
                project=cfg.project,
            )
            studio.activate()

        report = None
        total_elapsed_time = 0.0
        batch_starts = list(range(0, n_cases, batch_size))

        ChatUsageTokenMonitor.reset()
        try:
            with agentscope_token_monitor():
                for batch_index, start in enumerate(batch_starts):
                    end = min(start + batch_size, n_cases)
                    batch_text = "\n\n".join(rendered_cases[start:end])
                    prompt = REPORT_INSTRUCTION_TEMPLATE.format(
                        current_error_analysis_report=(
                            EMPTY_REPORT_PLACEHOLDER if report is None else report 
                        ),
                        failed_cases=batch_text,
                    )

                    if batch_index == len(batch_starts) - 1:
                        prompt += "\n\nNOTE: This is the final batch of failed cases."

                    api_key, base_url = api_pool.credential_for(batch_index)
                    agent = self._build_agent(
                        sys_prompt=sys_prompt,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    started = time.perf_counter()
                    reply = await agent(
                        Msg("user", prompt, "user"),
                        structured_model=_ErrorAnalysisReport,
                    )
                    elapsed = time.perf_counter() - started
                    total_elapsed_time += elapsed

                    response = _ErrorAnalysisReport.model_validate(reply.metadata)
                    report = response.report
        finally:
            if studio is not None:
                studio.deactivate()

        token_cost = ChatUsageTokenMonitor.to_dict() 
        n_events = token_cost["n"]
        avg_input = token_cost["avg_input_tokens"]
        avg_output = token_cost["avg_output_tokens"]
        costs = {
            "input_tokens": n_events * avg_input,
            "output_tokens": n_events * avg_output,
            "minutes": total_elapsed_time / 60,
        }
        ChatUsageTokenMonitor.reset()

        if save_folder is not None:
            final_path = os.path.join(save_folder, REPORT_FILE)
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"report": report},
                    f,
                    indent=4,
                    ensure_ascii=False,
                )

        return report, costs

    def run(
        self,
        failed_cases: list[FailedQueryCase],
        error_predictions: list[ErrorAttributionPrediction],
        *,
        target_system_overview: str | None = None,
        graph: ExecNetwork | list[ExecNetwork] | None = None,
        graph_paths: str | Path | list[str | Path] | None = None,
        save_folder: str | Path | None = None,
    ) -> tuple[str, dict[str, float | int]]:
        """Iteratively generate an error analysis report.

        Args:
            failed_cases (`list[FailedQueryCase]`):
                Failed query cases used to generate the report.
            error_predictions (`list[ErrorAttributionPrediction]`):
                Attributed error predictions.
            target_system_overview (`str | None`, optional):
                Overview of the target system being diagnosed.
            graph (`ExecNetwork | list[ExecNetwork] | None`, optional):
                One shared execution graph or one graph per failed case.
            graph_paths (`str | Path | list[str | Path] | None`, optional):
                Path or list of paths to serialized execution graphs. When
                used, graphs are loaded lazily. Note that either ``graph`` 
                or ``graph_paths`` must be provided.
            save_folder (`str | Path | None`, optional):
                Folder where the final report is saved. If not provided, the
                report is only returned and not persisted.

        Returns:
            `tuple[str, dict[str, float | int]]`:
                The final error analysis report and the total cost statistics.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    failed_cases=failed_cases,
                    error_predictions=error_predictions,
                    target_system_overview=target_system_overview,
                    graph=graph,
                    graph_paths=graph_paths,
                    save_folder=save_folder,
                )
            )

        raise RuntimeError(
            "`ErrorAnalysisReportRunner(...).run(...)` cannot be called from "
            "a running event loop. "
            "Use `await ErrorAnalysisReportRunner(...).arun(...)` instead."
        )
