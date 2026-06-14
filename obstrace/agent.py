from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase, OpenAIChatFormatter
from agentscope.memory import LongTermMemoryBase, MemoryBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase, OpenAIChatModel
from agentscope.plan import PlanNotebook
from agentscope.rag import KnowledgeBase
from agentscope.token import OpenAITokenCounter
from agentscope.tool import Toolkit
from agentscope.tts import TTSModelBase
from graphtrace._utils._agentscope import _MemoryMark

try:
    from .trace_notebook import CCTraceNotebook
except ImportError:  # pragma: no cover - supports running this folder as scripts.
    from trace_notebook import CCTraceNotebook


DEFAULT_MAX_ITERS = 200
DEFAULT_CONTEXT_WINDOW = 272_000


class AttributionPrediction(BaseModel):
    """Structured output for one flat operation-log attribution."""

    error_type: Literal[
        "ExtractionError",
        "UpdateError",
        "DeletionError",
        "RetrievalError",
        "ResponseError",
    ] = Field(description="The predicted memory-system error type.")
    op_id: str = Field(description="The operation id of the earliest decisive fault.")
    reason: str = Field(description="A concise operation-log-grounded attribution reason.")


class ObsTraceAttributionAgent(ReActAgent):
    """AgentScope ReAct agent equipped with flat operation-log tools."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        toolkit: Toolkit | None = None,
        memory: MemoryBase | None = None,
        long_term_memory: LongTermMemoryBase | None = None,
        long_term_memory_mode: Literal[
            "agent_control",
            "static_control",
            "both",
        ] = "both",
        enable_meta_tool: bool = False,
        parallel_tool_calls: bool = False,
        knowledge: KnowledgeBase | list[KnowledgeBase] | None = None,
        enable_rewrite_query: bool = True,
        plan_notebook: PlanNotebook | None = None,
        cc_trace_notebook: CCTraceNotebook | None = None,
        print_hint_msg: bool = False,
        max_iters: int = DEFAULT_MAX_ITERS,
        tts_model: TTSModelBase | None = None,
        compression_config: ReActAgent.CompressionConfig | None = None,
        task_template: str = "",
        attribution_instructions: str = "",
    ) -> None:
        """Initialize the flat operation-log attribution agent.

        Args:
            name (`str`):
                Agent name used by AgentScope.
            sys_prompt (`str`):
                System prompt defining the flatlog-only attribution task.
            model (`ChatModelBase`):
                AgentScope-compatible chat model.
            formatter (`FormatterBase`):
                Formatter that converts AgentScope messages and tools to the
                model provider format.
            toolkit (`Toolkit | None`, optional):
                Existing AgentScope toolkit. If omitted, the parent
                `ReActAgent` creates one.
            memory (`MemoryBase | None`, optional):
                Short-term agent memory. If omitted, AgentScope creates the
                default memory.
            long_term_memory (`LongTermMemoryBase | None`, optional):
                Optional AgentScope long-term memory.
            long_term_memory_mode (`Literal["agent_control", "static_control", "both"]`, defaults to `"both"`):
                AgentScope long-term memory control mode.
            enable_meta_tool (`bool`, defaults to `False`):
                Whether AgentScope should expose the meta tool for dynamic tool
                management.
            parallel_tool_calls (`bool`, defaults to `False`):
                Whether AgentScope may execute multiple model-requested tool
                calls in parallel.
            knowledge (`KnowledgeBase | list[KnowledgeBase] | None`, optional):
                Optional AgentScope knowledge base objects. This agent does not
                require them.
            enable_rewrite_query (`bool`, defaults to `True`):
                Whether AgentScope may rewrite queries before knowledge
                retrieval when knowledge is configured.
            plan_notebook (`PlanNotebook | None`, optional):
                Optional AgentScope planning notebook.
            cc_trace_notebook (`CCTraceNotebook | None`, optional):
                Flatlog notebook that provides trace inspection tools.
            print_hint_msg (`bool`, defaults to `False`):
                Whether to print notebook hint messages before reasoning.
            max_iters (`int`, defaults to `DEFAULT_MAX_ITERS`):
                Maximum AgentScope ReAct reasoning/tool iterations per reply.
            tts_model (`TTSModelBase | None`, optional):
                Optional text-to-speech model passed through to AgentScope.
            compression_config (`ReActAgent.CompressionConfig | None`, optional):
                AgentScope context compression configuration.
        """
        super().__init__(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            long_term_memory=long_term_memory,
            long_term_memory_mode=long_term_memory_mode,
            enable_meta_tool=enable_meta_tool,
            parallel_tool_calls=parallel_tool_calls,
            knowledge=knowledge,
            enable_rewrite_query=enable_rewrite_query,
            plan_notebook=plan_notebook,
            print_hint_msg=print_hint_msg,
            max_iters=max_iters,
            tts_model=tts_model,
            compression_config=compression_config,
        )
        self.cc_trace_notebook = cc_trace_notebook
        self.task_template = task_template
        self.attribution_instructions = attribution_instructions
        self.last_elapsed_seconds = 0.0
        if cc_trace_notebook:
            if enable_meta_tool:
                self.toolkit.create_tool_group(
                    "cc_trace_related",
                    description=cc_trace_notebook.description,
                )
                for tool in cc_trace_notebook.list_tools():
                    self.toolkit.register_tool_function(
                        tool,
                        group_name="cc_trace_related",
                    )
            else:
                for tool in cc_trace_notebook.list_tools():
                    self.toolkit.register_tool_function(tool)

    def build_task_prompt(self, case: dict[str, Any]) -> str:
        """Build the case prompt.

        Args:
            case (`dict[str, Any]`):
                Case dictionary containing the question, golden answer, and
                model answer under review.

        Returns:
            `str`:
                Prompt text for one attribution case.
        """
        if not self.task_template:
            raise RuntimeError("ObsTraceAttributionAgent requires a task template.")
        return self.task_template.format(
            instructions=self.attribution_instructions,
            question=case.get("question", ""),
            golden_answer=case.get("golden_answer", ""),
            prediction=case.get("prediction", ""),
        )

    def solve_case(self, case: dict[str, Any]) -> dict[str, Any]:
        """Synchronously solve one attribution case.

        Args:
            case (`dict[str, Any]`):
                Attribution case produced by `graph_loader`.

        Returns:
            `dict[str, Any]`:
                Agent prediction, usage, and timing.
        """
        async def run_and_close() -> dict[str, Any]:
            try:
                return await self.solve_case_async(case)
            finally:
                await self.aclose()

        return asyncio.run(run_and_close())

    async def solve_case_async(
        self,
        case: dict[str, Any],
    ) -> dict[str, Any]:
        """Asynchronously solve one attribution case with validation retries.

        Args:
            case (`dict[str, Any]`):
                Attribution case produced by `graph_loader`.

        Returns:
            `dict[str, Any]`:
                Agent prediction, usage, and timing.
        """
        if self.cc_trace_notebook is None:
            raise RuntimeError("ObsTraceAttributionAgent requires a cc_trace_notebook.")
        reset_usage = getattr(self.model, "reset_usage", None)
        if reset_usage is not None:
            reset_usage()
        started_at = perf_counter()
        prompt = self.build_task_prompt(case)
        try:
            reply = await self.reply(
                Msg("user", prompt, "user"),
                structured_model=AttributionPrediction,
            )
            prediction = AttributionPrediction.model_validate(reply.metadata or {})
            final = self.cc_trace_notebook.finish_attribution(
                prediction.error_type,
                prediction.op_id,
                prediction.reason,
            )
            if "error" in final:
                final["warning"] = final.pop("error")
        except (TypeError, ValueError, ValidationError) as exc:
            final = {
                "error_type": "",
                "op_id": "",
                "reason": "",
                "warning": f"Malformed structured attribution: {type(exc).__name__}: {exc}",
            }

        get_usage_snapshot = getattr(self.model, "get_usage_snapshot", None)
        usage = get_usage_snapshot() if get_usage_snapshot is not None else None
        self.last_elapsed_seconds = perf_counter() - started_at
        output = {
            "question": case.get("question", ""),
            "golden_answer": case.get("golden_answer", ""),
            "prediction": case.get("prediction", ""),
            "error_type": final.get("error_type", "") if final else "",
            "op_id": final.get("op_id", "") if final else "",
            "reason": final.get("reason", "") if final else "",
            "warning": final.get("warning", "") if final else "",
            "trace_stats": {
                "operation_count": len(self.cc_trace_notebook.index.blocks),
            },
            "usage": {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
                "num_calls": getattr(usage, "num_calls", 0),
            },
            "elapsed_seconds": self.last_elapsed_seconds,
        }
        return output

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Inject the current flatlog notebook hint before each reasoning step.

        Args:
            tool_choice (`Literal["auto", "none", "required"] | None`, optional):
                AgentScope tool-choice policy.

        Returns:
            `Msg`:
                The assistant reasoning message returned by AgentScope.
        """
        if self.cc_trace_notebook:
            hint_msg = await self.cc_trace_notebook.get_current_hint()
            if self.print_hint_msg and hint_msg:
                await self.print(hint_msg)
            await self.memory.add(hint_msg, marks=_MemoryMark.HINT)
        return await super()._reasoning(tool_choice)

    def close(self) -> None:
        """Close the underlying model client from synchronous code."""
        close = getattr(self.model, "close", None)
        if close is not None:
            close()

    async def aclose(self) -> None:
        """Close the underlying model client from asynchronous code."""
        close = getattr(self.model, "aclose", None)
        if close is not None:
            await close()


def build_obstrace_agent(
    flattened_trace_text: str,
    *,
    system_prompt: str,
    task_template: str,
    attribution_instructions: str,
    initial_ranked_ops: list[dict[str, Any]] | None = None,
    api_config_path: str | Path,
    endpoint_index: int | None = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 1.0,
    max_iters: int = DEFAULT_MAX_ITERS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    stream: bool = False,
    enable_compression: bool = True,
) -> ObsTraceAttributionAgent:
    """Build an AgentScope attribution agent over a flattened operation log.

    Args:
        flattened_trace_text (`str`):
            Edge-free flattened operation trace for the current case.
        initial_ranked_ops (`list[dict[str, Any]] | None`, optional):
            Precomputed ranked starting operations injected into notebook
            state before reasoning begins.
        api_config_path (`str | Path`):
            Path to API endpoint configuration.
        endpoint_index (`int | None`, optional):
            If provided, use only the indexed endpoint from `api_config.json`
            so one worker can be pinned to one API key/base-url pair.
        model (`str`, defaults to `"gpt-4.1-mini"`):
            Model name passed to the OpenAI-compatible endpoint.
        temperature (`float`, defaults to `1.0`):
            Sampling temperature for model calls.
        max_iters (`int`, defaults to `DEFAULT_MAX_ITERS`):
            Maximum AgentScope ReAct iterations per reply.
        context_window (`int`, defaults to `DEFAULT_CONTEXT_WINDOW`):
            Token threshold used to trigger AgentScope compression.
        stream (`bool`, defaults to `False`):
            Whether to request streaming model responses.
        enable_compression (`bool`, defaults to `True`):
            Whether to enable AgentScope context compression.
    Returns:
        `ObsTraceAttributionAgent`:
            A ready-to-run flat operation-log attribution agent.
    """
    """
    加载 AgentScope 组件
    读取 API endpoint 配置
    根据 endpoint_index 选择 endpoint
    构建可轮换、可统计 usage 的 LLM model
    构建 OpenAI formatter
    可选地构建上下文压缩配置
    创建并返回带 trace 工具的 attribution agent
    """
    with Path(api_config_path).open("r", encoding="utf-8") as file:
        config = json.load(file)
    api_keys = config.get("api_keys", [])
    base_urls = config.get("base_urls", [])
    if not isinstance(api_keys, list) or not isinstance(base_urls, list):
        raise ValueError("api_keys and base_urls must both be lists.")
    if len(api_keys) != len(base_urls):
        raise ValueError("api_keys and base_urls must have the same length.")
    endpoints = [
        (str(api_key), str(base_url))
        for api_key, base_url in zip(api_keys, base_urls)
        if str(api_key).strip()
    ]
    if not endpoints:
        raise ValueError(f"No valid API endpoints found in {api_config_path}")
    if endpoint_index is not None:
        if endpoint_index < 0 or endpoint_index >= len(endpoints):
            raise ValueError(
                f"`endpoint_index` out of range: {endpoint_index}; "
                f"available endpoints={len(endpoints)}"
            )
        endpoint = endpoints[endpoint_index]
    else:
        endpoint = endpoints[0]
    api_key, base_url = endpoint
    llm_model = OpenAIChatModel(
        model_name=model,
        api_key=api_key,
        stream=stream,
        client_kwargs={"base_url": base_url},
        generate_kwargs={"temperature": temperature},
    )
    formatter = OpenAIChatFormatter(token_counter=OpenAITokenCounter(model))
    compression_config = None
    if enable_compression:
        compression_config = ObsTraceAttributionAgent.CompressionConfig(
            enable=True,
            agent_token_counter=OpenAITokenCounter(model),
            trigger_threshold=context_window,
        )
    return ObsTraceAttributionAgent(
        name="obstrace",
        sys_prompt=system_prompt,
        model=llm_model,
        formatter=formatter,
        cc_trace_notebook=CCTraceNotebook(
            flattened_trace_text,
            initial_ranked_ops=initial_ranked_ops,
        ),
        max_iters=max_iters,
        compression_config=compression_config,
        task_template=task_template,
        attribution_instructions=attribution_instructions,
    )
