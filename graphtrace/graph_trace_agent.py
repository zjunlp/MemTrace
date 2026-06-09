# -*- coding: utf-8 -*-
# TODO: Current evaluation data labels each error case with a singleton decisive 
# error set. Future work should equip the graph trace notebook with additional
# tools for cases where the target decisive error set contains multiple
# operations. For example, if the agent find an earilest decisive error, it can 
# prune the execution graph to improve the efficiency of the search.
"""An agent specialized for automatic failure attribution based on an execution graph."""

from agentscope.agent import ReActAgent
from agentscope.formatter import FormatterBase
from agentscope.memory import LongTermMemoryBase, MemoryBase
from agentscope.message import Msg
from agentscope.model import ChatModelBase
from agentscope.plan import PlanNotebook
from agentscope.rag import KnowledgeBase
from agentscope.tool import Toolkit
from agentscope.tts import TTSModelBase
from agentscope import logger
from ._utils._agentscope import _MemoryMark
from .graph_trace_notebook import GraphTraceNotebook
from typing import Literal


class GraphTraceAgent(ReActAgent):
    """An agent specialized for automatic failure attribution based on an execution graph.

    This agent extends the base functionality with an additional capability for
    automatic failure attribution by leveraging structured execution graphs,
    enabling systematic root-cause analysis in complex workflows.
    """

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
        graph_trace_notebook: GraphTraceNotebook | None = None,
        print_hint_msg: bool = False,
        max_iters: int = 10,
        tts_model: TTSModelBase | None = None,
        max_context_limit: int = 600_000,
        compression_config: ReActAgent.CompressionConfig | None = None,
    ) -> None:
        """Initialize the graph trace agent. 

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`str`):
                The system prompt of the agent.
            model (`ChatModelBase`):
                The chat model used by the agent.
            formatter (`FormatterBase`):
                The formatter used to format the messages into the required
                format of the model API provider.
            toolkit (`Toolkit | None`, optional):
                A `Toolkit` object that contains the tool functions. If not
                provided, a default empty `Toolkit` will be created.
            memory (`MemoryBase | None`, optional):
                The memory used to store the dialogue history. If not provided,
                a default `InMemoryMemory` will be created, which stores
                messages in a list in memory.
            long_term_memory (`LongTermMemoryBase | None`, optional):
                The optional long-term memory, which will provide two tool
                functions: `retrieve_from_memory` and `record_to_memory`, and
                will attach the retrieved information to the system prompt
                before each reply.
            enable_meta_tool (`bool`, defaults to `False`):
                If `True`, a meta tool function `reset_equipped_tools` will be
                added to the toolkit, which allows the agent to manage its
                equipped tools dynamically.
            long_term_memory_mode (`Literal['agent_control', 'static_control',\
              'both']`, defaults to `both`):
                The mode of the long-term memory. If `agent_control`, two
                tool functions `retrieve_from_memory` and `record_to_memory`
                will be registered in the toolkit to allow the agent to
                manage the long-term memory. If `static_control`, retrieving
                and recording will happen in the beginning and end of
                each reply respectively.
            parallel_tool_calls (`bool`, defaults to `False`):
                When LLM generates multiple tool calls, whether to execute
                them in parallel.
            knowledge (`KnowledgeBase | list[KnowledgeBase] | None`, optional):
                The knowledge object(s) used by the agent to retrieve
                relevant documents at the beginning of each reply.
            enable_rewrite_query (`bool`, defaults to `True`):
                Whether ask the agent to rewrite the user input query before
                retrieving from the knowledge base(s), e.g. rewrite "Who am I"
                to "{user's name}" to get more relevant documents. Only works
                when the knowledge base(s) is provided.
            plan_notebook (`PlanNotebook | None`, optional):
                The plan notebook instance. It allows the agent to finish the
                complex task by decomposing it into a sequence of subtasks.
            graph_trace_notebook (`GraphTraceNotebook | None`, optional):
                The graph trace notebook instance. It allows the agent to find
                the decisive error set by exploring the execution graph.
            print_hint_msg (`bool`, defaults to `False`):
                Whether to print the hint messages, including the reasoning
                hints from the notebooks, the retrieved information from the 
                long-term memory and knowledge base(s).
            max_iters (`int`, defaults to `10`):
                The maximum number of iterations of the reasoning-acting loops.
            tts_model (`TTSModelBase | None` optional):
                The TTS model used by the agent.
            max_context_limit (`int`, defaults to `600_000`):
                The maximum context length allowed when preparing messages for
                memory compression.
            compression_config (`CompressionConfig | None`, optional):
                The compression configuration. If provided, the auto
                compression will be activated.
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

        self.max_context_limit = max_context_limit
        self.graph_trace_notebook = None
        if graph_trace_notebook:
            self.graph_trace_notebook = graph_trace_notebook
            if enable_meta_tool:
                self.toolkit.create_tool_group(
                    "graph_trace_related",
                    description=self.graph_trace_notebook.description,
                )
                for tool in graph_trace_notebook.list_tools():
                    self.toolkit.register_tool_function(
                        tool,
                        group_name="graph_trace_related",
                    )
            else:
                for tool in graph_trace_notebook.list_tools():
                    self.toolkit.register_tool_function(tool)

    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """If a graph trace notebook is provided, it performs reasoning 
        process with the graph trace notebook hint.

        Args:
            tool_choice (`Literal["auto", "none", "required"] | None`, optional):
                Tool-choice policy passed to AgentScope.

        Returns:
            `Msg`:
                The assistant reasoning message generated by AgentScope.
        """
        if self.graph_trace_notebook:
            # Insert the reasoning hint from the graph trace notebook. 
            hint_msg = await self.graph_trace_notebook.get_current_hint()
            if self.print_hint_msg and hint_msg:
                await self.print(hint_msg)
            await self.memory.add(hint_msg, marks=_MemoryMark.HINT)

        return await super()._reasoning(tool_choice)


    async def _compress_memory_if_needed(self) -> None:
        """Compress the memory content if needed."""
        if (
            self.compression_config is None
            or not self.compression_config.enable
        ):
            return

        # Obtain the messages that have not been compressed yet
        to_compressed_msgs = await self.memory.get_memory(
            exclude_mark=_MemoryMark.COMPRESSED,
        )

        # Keep recent messages uncompressed while respecting tool use and result
        # groups. If the kept suffix itself would exceed the context limit,
        # include the oldest overflowing complete group in compression.
        n_keep = 0
        num_accumulated = 0
        accumulated_tool_call_ids = set()
        for i in range(len(to_compressed_msgs) - 1, -1, -1):
            msg = to_compressed_msgs[i]
            for block in msg.get_content_blocks("tool_result"):
                accumulated_tool_call_ids.add(block["id"])

            for block in msg.get_content_blocks("tool_use"):
                if block["id"] in accumulated_tool_call_ids:
                    accumulated_tool_call_ids.remove(block["id"])

            # Handle tool use/result pairs as an indivisible message group.
            if len(accumulated_tool_call_ids) == 0:
                candidate_prompt = await self.formatter.format(
                    [
                        Msg("system", self.sys_prompt, "system"),
                        *to_compressed_msgs[i:],
                    ],
                )
                candidate_tokens = (
                    await self.compression_config.agent_token_counter.count(
                        candidate_prompt,
                    )
                )
                if candidate_tokens > self.max_context_limit:
                    to_compressed_msgs = to_compressed_msgs[:i + num_accumulated + 1]
                    break

                n_keep += 1
                num_accumulated = 0
            else:
                num_accumulated += 1

            # Break if reach the number of messages to keep
            if n_keep >= self.compression_config.keep_recent:
                # Remove the messages that should be kept uncompressed
                to_compressed_msgs = to_compressed_msgs[:i]
                break

        # Skip compression if no messages to compress
        if not to_compressed_msgs:
            return
        original_to_compressed_msgs = to_compressed_msgs

        # Calculate the token
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *to_compressed_msgs,
            ],
        )
        n_tokens = await self.compression_config.agent_token_counter.count(
            prompt,
        )

        if n_tokens > self.compression_config.trigger_threshold:
            original_n_tokens = n_tokens
            logger.info(
                "Memory compression is triggered (%d > "
                "threshold %d) for agent %s.",
                n_tokens,
                self.compression_config.trigger_threshold,
                self.name,
            )

            if n_tokens > self.max_context_limit:
                selected_start_idx = None
                accumulated_tool_call_ids = set()
                for i in range(len(to_compressed_msgs) - 1, -1, -1):
                    msg = to_compressed_msgs[i]
                    for block in msg.get_content_blocks("tool_result"):
                        accumulated_tool_call_ids.add(block["id"])

                    for block in msg.get_content_blocks("tool_use"):
                        if block["id"] in accumulated_tool_call_ids:
                            accumulated_tool_call_ids.remove(block["id"])

                    # Only cut at complete tool use/result group boundaries.
                    # OpenAI requires each tool response to have a preceding
                    # tool call in the same request.
                    if len(accumulated_tool_call_ids) != 0:
                        continue

                    candidate_prompt = await self.formatter.format(
                        [
                            Msg("system", self.sys_prompt, "system"),
                            *to_compressed_msgs[i:],
                        ],
                    )
                    candidate_tokens = (
                        await self.compression_config.agent_token_counter.count(
                            candidate_prompt,
                        )
                    )
                    if candidate_tokens < self.max_context_limit:
                        selected_start_idx = i
                        n_tokens = candidate_tokens
                    else:
                        break

                if selected_start_idx is None:
                    logger.warning(
                        "Skipped memory compression for agent %s because even "
                        "the latest message to be compressed exceeds the maximum context limit %d. "
                        "Marking all %d target messages as compressed.",
                        self.name,
                        self.max_context_limit,
                        len(original_to_compressed_msgs),
                    )
                    await self.memory.update_messages_mark(
                        msg_ids=[_.id for _ in original_to_compressed_msgs],
                        new_mark=_MemoryMark.COMPRESSED,
                    )
                    return

                logger.warning(
                    "Memory compression for agent %s is limited to the latest "
                    "%d messages because %d tokens exceed the maximum context "
                    "limit %d.",
                    self.name,
                    len(to_compressed_msgs) - selected_start_idx,
                    original_n_tokens,
                    self.max_context_limit,
                )
                to_compressed_msgs = to_compressed_msgs[selected_start_idx:]

            # The formatter used for compression
            compression_formatter = (
                self.compression_config.compression_formatter or self.formatter
            )

            # Prepare the prompt used to compress the memories
            compression_prompt = await compression_formatter.format(
                [
                    Msg("system", self.sys_prompt, "system"),
                    *to_compressed_msgs,
                    Msg(
                        "user",
                        self.compression_config.compression_prompt,
                        "user",
                    ),
                ],
            )

            # TODO: What if the compressed messages include multimodal blocks?
            # Use the specified compression model if provided
            compression_model = (
                self.compression_config.compression_model or self.model
            )
            res = await compression_model(
                compression_prompt,
                structured_model=(self.compression_config.summary_schema),
            )

            # Obtain the structured output from the model response
            last_chunk = None
            if compression_model.stream:
                async for chunk in res:
                    last_chunk = chunk
            else:
                last_chunk = res

            # Format the compressed memory summary
            if last_chunk.metadata:
                # Update the compressed summary in the memory storage
                await self.memory.update_compressed_summary(
                    self.compression_config.summary_template.format(
                        **last_chunk.metadata,
                    ),
                )

                # Mark the compressed messages in the memory storage
                await self.memory.update_messages_mark(
                    msg_ids=[_.id for _ in original_to_compressed_msgs],
                    new_mark=_MemoryMark.COMPRESSED,
                )

                logger.info(
                    "Finished compressing %d messages and marked %d messages "
                    "as compressed in agent %s.",
                    len(to_compressed_msgs),
                    len(original_to_compressed_msgs),
                    self.name,
                )

            else:
                logger.warning(
                    "Failed to obtain compression summary from the model "
                    "structured output in agent %s.",
                    self.name,
                )

