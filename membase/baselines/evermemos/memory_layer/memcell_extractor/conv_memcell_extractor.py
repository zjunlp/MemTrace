"""
Simple Boundary Detection Base Class for EverMemOS

This module provides a simple and extensible base class for detecting
boundaries in various types of content (conversations, emails, notes, etc.).
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
from functools import partial
import uuid
import json, re
import asyncio
from smartcomment import (
    comment_link,
    comment_op,
    comment_op_scope,
    comment_variable,
    current_context,
    is_tracing_enabled, 
)
from core.di.utils import get_bean, get_bean_by_type
from core.component.llm.tokenizer.tokenizer_factory import TokenizerFactory
from common_utils.datetime_utils import from_iso_format as dt_from_iso_format
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import RawDataType

from memory_layer.prompts import get_prompt_by
from memory_layer.memcell_extractor.base_memcell_extractor import (
    MemCellExtractor,
    RawData,
    MemCell,
    StatusResult,
    MemCellExtractRequest,
)
from core.observation.logger import get_logger
from agentic_layer.metrics.memorize_metrics import (
    record_boundary_detection,
    record_memcell_extracted,
    get_space_id_for_metrics,
)
import time

logger = get_logger(__name__)


@dataclass
class BoundaryDetectionResult:
    """Boundary detection result."""

    should_end: bool
    should_wait: bool
    reasoning: str
    confidence: float
    topic_summary: Optional[str] = None


@dataclass
class ConversationMemCellExtractRequest(MemCellExtractRequest):
    pass


class ConvMemCellExtractor(MemCellExtractor):
    """
    Conversation MemCell Extractor - Responsible only for boundary detection and creating basic MemCell

    Responsibilities:
    1. Boundary detection (determine whether current MemCell should end)
    2. Create basic MemCell (including basic fields such as original_data, summary, timestamp, etc.)

    Not included:
    - Episode extraction (handled by EpisodeMemoryExtractor)
    - Foresight extraction (handled by ForesightExtractor)
    - EventLog extraction (handled by EventLogExtractor)
    - Embedding computation (handled by MemoryManager)

    Language support:
    - Controlled by MEMORY_LANGUAGE env var: 'zh' (Chinese) or 'en' (English), default 'en'
    """

    # Default limits for force splitting
    DEFAULT_HARD_TOKEN_LIMIT = 8192
    DEFAULT_HARD_MESSAGE_LIMIT = 50

    @classmethod
    def _get_tokenizer(cls):
        """Get the shared tokenizer from tokenizer factory (with caching)."""
        tokenizer_factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)
        return tokenizer_factory.get_tokenizer_from_tiktoken("o200k_base")

    def __init__(
        self,
        llm_provider=LLMProvider,
        boundary_detection_prompt: Optional[str] = None,
        use_eval_prompts: bool = False,
        hard_token_limit: Optional[int] = None,
        hard_message_limit: Optional[int] = None,
    ):
        super().__init__(RawDataType.CONVERSATION, llm_provider)
        self.llm_provider = llm_provider

        # Force split limits
        self.hard_token_limit = hard_token_limit or self.DEFAULT_HARD_TOKEN_LIMIT
        self.hard_message_limit = hard_message_limit or self.DEFAULT_HARD_MESSAGE_LIMIT

        # Use custom prompt or get default via PromptManager
        self.conv_boundary_detection_prompt = (
            boundary_detection_prompt or get_prompt_by("CONV_BOUNDARY_DETECTION_PROMPT")
        )

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens in message list using tiktoken.

        Includes speaker_name in token count since it's included when passed to LLM.

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        tokenizer = self._get_tokenizer()
        total = 0
        for msg in messages:
            if isinstance(msg, dict):
                speaker = msg.get('speaker_name', '')
                content = msg.get('content', '')
                # Format matches what's sent to LLM: "speaker: content"
                text = f"{speaker}: {content}" if speaker else content
            else:
                text = str(msg)
            total += len(tokenizer.encode(text))
        return total

    def _extract_participant_ids(
        self, chat_raw_data_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract all participant IDs from chat_raw_data_list

        Retrieve from the content dictionary of each element:
        1. speaker_id (speaker ID)
        2. All _id in referList (@mentioned user IDs)

        Args:
            chat_raw_data_list: List of raw chat data

        Returns:
            List[str]: List of deduplicated participant IDs
        """
        participant_ids = set()

        for raw_data in chat_raw_data_list:

            # Extract speaker_id
            if 'speaker_id' in raw_data and raw_data['speaker_id']:
                participant_ids.add(raw_data['speaker_id'])

            # Extract all IDs from referList
            if 'referList' in raw_data and raw_data['referList']:
                for refer_item in raw_data['referList']:
                    # refer_item may be a dictionary format containing _id field
                    if isinstance(refer_item, dict):
                        # Handle MongoDB ObjectId format _id
                        if '_id' in refer_item:
                            refer_id = refer_item['_id']
                            # If it's an ObjectId object, convert to string
                            if hasattr(refer_id, '__str__'):
                                participant_ids.add(str(refer_id))
                            else:
                                participant_ids.add(refer_id)
                        # Also check regular id field
                        elif 'id' in refer_item:
                            participant_ids.add(refer_item['id'])
                    # If refer_item is directly an ID string
                    elif isinstance(refer_item, str):
                        participant_ids.add(refer_item)

        return list(participant_ids)

    def _format_conversation_dicts(
        self, messages: list[dict[str, str]], include_timestamps: bool = False
    ) -> str:
        """Format conversation from message dictionaries into plain text."""
        lines = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            speaker_name = msg.get("speaker_name", "")
            timestamp = msg.get("timestamp", "")

            if content:
                if include_timestamps and timestamp:
                    try:
                        # Handle different types of timestamp
                        if isinstance(timestamp, datetime):
                            # If it's a datetime object, format directly
                            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            lines.append(f"[{time_str}] {speaker_name}: {content}")
                        elif isinstance(timestamp, str):
                            # If it's a string, parse and then format
                            dt = datetime.fromisoformat(
                                timestamp.replace("Z", "+00:00")
                            )
                            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                            lines.append(f"[{time_str}] {speaker_name}: {content}")
                        else:
                            # Other types, do not include timestamp
                            lines.append(f"{speaker_name}: {content}")
                    except (ValueError, AttributeError, TypeError):
                        # Fallback if timestamp parsing fails
                        lines.append(f"{speaker_name}: {content}")
                else:
                    lines.append(f"{speaker_name}: {content}")
            else:
                logger.debug(
                    f"[ConversationEpisodeBuilder] Warning: message {i} has no content"
                )
        return "\n".join(lines)

    def _calculate_time_gap(
        self,
        conversation_history: list[dict[str, str]],
        new_messages: list[dict[str, str]],
    ):
        if not conversation_history or not new_messages:
            return "No time gap information available"

        try:
            # Get the last message from history and first new message
            last_history_msg = conversation_history[-1]
            first_new_msg = new_messages[0]

            last_timestamp_str = last_history_msg.get("timestamp", "")
            first_timestamp_str = first_new_msg.get("timestamp", "")

            if not last_timestamp_str or not first_timestamp_str:
                return "No timestamp information available"

            # Parse timestamps - handle different types of timestamp
            try:
                if isinstance(last_timestamp_str, datetime):
                    last_time = last_timestamp_str
                elif isinstance(last_timestamp_str, str):
                    last_time = datetime.fromisoformat(
                        last_timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    return "Invalid timestamp format for last message"

                if isinstance(first_timestamp_str, datetime):
                    first_time = first_timestamp_str
                elif isinstance(first_timestamp_str, str):
                    first_time = datetime.fromisoformat(
                        first_timestamp_str.replace("Z", "+00:00")
                    )
                else:
                    return "Invalid timestamp format for first message"
            except (ValueError, TypeError):
                return "Failed to parse timestamps"

            # Calculate time difference
            time_diff = first_time - last_time
            total_seconds = time_diff.total_seconds()

            if total_seconds < 0:
                return "Time gap: Messages appear to be out of order"
            elif total_seconds < 60:  # Less than 1 minute
                return f"Time gap: {int(total_seconds)} seconds (immediate response)"
            elif total_seconds < 3600:  # Less than 1 hour
                minutes = int(total_seconds // 60)
                return f"Time gap: {minutes} minutes (recent conversation)"
            elif total_seconds < 86400:  # Less than 1 day
                hours = int(total_seconds // 3600)
                return f"Time gap: {hours} hours (same day, but significant pause)"
            else:  # More than 1 day
                days = int(total_seconds // 86400)
                return f"Time gap: {days} days (long gap, likely new conversation)"

        except (ValueError, KeyError, AttributeError) as e:
            return f"Time gap calculation error: {str(e)}"

    async def _detect_boundary(
        self,
        conversation_history: list[dict[str, str]],
        new_messages: list[dict[str, str]],
    ) -> BoundaryDetectionResult:
        if not conversation_history:
            return BoundaryDetectionResult(
                should_end=False,
                should_wait=False,
                reasoning="First messages in conversation",
                confidence=1.0,
                topic_summary="",
            )
        history_text = self._format_conversation_dicts(
            conversation_history, include_timestamps=True
        )
        new_text = self._format_conversation_dicts(
            new_messages, include_timestamps=True
        )
        time_gap_info = self._calculate_time_gap(conversation_history, new_messages)

        runtime_time_gap_info = comment_variable(
            time_gap_info,
            to_runtime=True,
            category="time_gap",
            class_name="time_gap",
            comment=(
                "A human-readable summary of the temporal gap between the "
                "last history message and the first newly introduced message. "
                "It acts as an auxiliary timeline signal for boundary detection, "
                "for example indicating whether the new message is an immediate "
                "continuation or arrives after a much longer pause."
            ),
        )

        runtime_history_messages = runtime_new_messages = None 
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_history_messages = tracing_ctx.get_variable("history_messages")
            runtime_new_messages = tracing_ctx.get_variable("new_messages")
        comment_op(
            inputs=[runtime_history_messages, runtime_new_messages],
            outputs=[runtime_time_gap_info],
            comment=(
                "Based on the history messages and the newly introduced messages, "
                "the memory system computes a human-readable temporal-gap summary "
                "between the previous message and the new message. This summary is "
                "later provided to the boundary-detection model as an extra signal "
                "for deciding whether a new memory unit should start."
            ),
        )

        logger.debug(
            f"[ConversationEpisodeBuilder] Detect boundary – history tokens: {len(history_text)} new tokens: {len(new_text)} time gap: {time_gap_info}"
        )

        prompt = self.conv_boundary_detection_prompt.format(
            conversation_history=history_text,
            new_messages=new_text,
            time_gap_info=time_gap_info,
        )


        runtime_prompt_template = comment_variable(
            self.conv_boundary_detection_prompt,
            to_runtime=True,
            id_strategy=lambda _: "boundary-detection-prompt-template",
            category="prompt",
            comment=(
                "A prompt template that asks the large language model to judge whether "
                "the newly added message or messages indicate that the current scenario "
                "should end and a new scenario should begin."
            ),
            metadata={
                "op_type": "boundary_detection",
            },
        )
        runtime_prompt = comment_variable(
            prompt,
            to_runtime=True,
            class_name="boundary_detection_prompt",
            category="prompt",
            comment=(
                "A concrete boundary-detection prompt built from the history messages, "
                "the newly introduced messages, the temporal-gap summary, and the "
                "boundary-detection prompt template."
            ),
        )
        comment_op(
            inputs=[
                runtime_history_messages, 
                runtime_new_messages, 
                runtime_time_gap_info, 
                runtime_prompt_template,
            ],
            outputs=[runtime_prompt],
            comment=(
                "The memory system assembles a boundary-detection prompt for the "
                "large language model using the history messages, the newly introduced "
                "messages, the temporal-gap summary, and the prompt template."
            ),
        )


        errors = [] 
        for i in range(5):
            try:
                resp = await self.llm_provider.generate(prompt)

                runtime_raw_response = comment_variable(
                    resp,
                    to_runtime=True,
                    class_name="raw_boundary_detection_response",
                    category="llm_response",
                    comment=(
                        "A raw large-language-model response for the memory system's " 
                        "boundary detection."
                    ),
                )
                comment_link(
                    source=runtime_prompt,
                    target=runtime_raw_response,
                    comment=(
                        "The large language model judges whether the newly added message or "
                        "messages indicate that the current scenario should end and a new "
                        "scenario should begin based on the assembled boundary-detection prompt."
                    ),
                )

                logger.debug(
                    f"[ConversationEpisodeBuilder] Boundary response length: {len(resp)} chars"
                )

                # Parse JSON response from LLM boundary detection
                json_match = re.search(r"\{[^{}]*\}", resp, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    result = BoundaryDetectionResult(
                        should_end=data.get("should_end", False),
                        should_wait=data.get("should_wait", True),
                        reasoning=data.get("reasoning", "No reason provided"),
                        confidence=data.get("confidence", 1.0),
                        topic_summary=data.get("topic_summary", ""),
                    )
                    # Record success metrics
                    detection_result = 'should_end' if result.should_end else 'should_wait'
                    record_boundary_detection(
                        space_id=get_space_id_for_metrics(),
                        raw_data_type=self.raw_data_type.value,
                        result=detection_result,
                        trigger_type='llm',
                    )


                    runtime_boundary_detection_result = comment_variable(
                        result,
                        variable_name="boundary_detection_result",
                        to_runtime=True,
                        class_name="boundary_detection_result",
                        category="llm_response",
                        comment=(
                            "The structured boundary-detection result parsed from the "
                            "large language model's response."
                        ),
                    )
                    comment_link(
                        source=runtime_raw_response,
                        target=runtime_boundary_detection_result,
                        comment=(
                            "The memory system parses the raw boundary-detection response "
                            "into a structured result."
                        ),
                        edge_metadata={
                            "source_code": (
                                "json_match = re.search(r\"\\\\{[^{}]*\\\\}\", resp, re.DOTALL)\n"
                                "data = json.loads(json_match.group())\n"
                                "result = BoundaryDetectionResult(\n"
                                "    should_end=data.get(\"should_end\", False),\n"
                                "    should_wait=data.get(\"should_wait\", True),\n"
                                "    reasoning=data.get(\"reasoning\", \"No reason provided\"),\n"
                                "    confidence=data.get(\"confidence\", 1.0),\n"
                                "    topic_summary=data.get(\"topic_summary\", \"\"),\n"
                                ")"
                            ),
                        },
                    )
                    return result
                else:
                    # JSON parsing failed, retry
                    errors.append(
                        f"Attempt {i + 1} fails because the boundary-detection response "
                        "does not contain a JSON object matching the expected schema."
                    )
                    logger.warning(
                        f"[ConversationEpisodeBuilder] Failed to parse JSON from LLM response (attempt {i+1}/5), response: {resp[:200]}..."
                    )
                    continue
            except Exception as e:
                errors.append(
                    f"Attempt {i + 1} fails because boundary detection raises "
                    f"{e.__class__.__name__}: {e}."
                )
                logger.warning(
                    f"[ConversationEpisodeBuilder] Boundary detection error (attempt {i+1}/5): {e}"
                )
                continue

        # All retries exhausted, return default result
        logger.error(
            f"[ConversationEpisodeBuilder] All 5 retries exhausted for boundary detection, returning default (should_end=False)"
        )

        runtime_fallback_result = comment_variable(
            BoundaryDetectionResult(
                should_end=False,
                should_wait=True,
                reasoning="All retries exhausted - failed to parse LLM response",
                confidence=0.0,
                topic_summary="",
            ),
            variable_name="boundary_detection_result",
            to_runtime=True,
            class_name="boundary_detection_result",
            category="llm_response",
            comment=(
                "The conservative fallback boundary-detection result used after "
                "all retry attempts fail."
            ),
        )
        
        comment_link(
            source=runtime_prompt,
            target=runtime_fallback_result,
            comment=(
                "After five failed attempts, the memory system falls back to a "
                "conservative boundary-detection result that keeps waiting for "
                "more messages."
            ),
            edge_metadata={
                "error": (
                    "The memory system attempts boundary detection five times "
                    "and fails to obtain a valid structured result each time.\n\n"
                    + "\n\n".join(errors)
                ), 
            },
        )

        return BoundaryDetectionResult(
            should_end=False,
            should_wait=True,
            reasoning="All retries exhausted - failed to parse LLM response",
            confidence=0.0,
            topic_summary="",
        )

    async def extract_memcell(
        self, request: ConversationMemCellExtractRequest
    ) -> tuple[Optional[MemCell], Optional[StatusResult]]:
        """
        Extract basic MemCell (only contains raw data and basic fields)

        The returned MemCell only includes:
        - event_id: event ID
        - user_id_list: list of user IDs
        - original_data: raw message data
        - timestamp: timestamp
        - summary: summary
        - group_id: group ID
        - participants: participant list
        - type: data type

        Not included (to be filled by other extractors later):
        - episode: filled by EpisodeMemoryExtractor
        - foresights: filled by ForesightExtractor
        - event_log: filled by EventLogExtractor
        - extend['embedding']: filled by MemoryManager
        """
        # Get the current message buffer
        runtime_message_buffer = None 
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_message_buffer = tracing_ctx.get_variable("message_buffer")

        # EverMemOS performs a check on its current message buffer to filter out unsupported messages.
        # However, in this memory evaluation framework, such cases do not occur, so this step is not tracked.
        history_message_dict_list = []
        for raw_data in request.history_raw_data_list:
            processed_data = self._data_process(raw_data)
            if processed_data is not None:  # Filter out unsupported message types
                history_message_dict_list.append(processed_data)

        # Check if the last new_raw_data is None
        if (
            request.new_raw_data_list
            and self._data_process(request.new_raw_data_list[-1]) is None
        ):
            logger.warning(
                f"[ConvMemCellExtractor] The last new_raw_data is None, skipping processing"
            )
            status_control_result = StatusResult(should_wait=True)
            return (None, status_control_result)

        new_message_dict_list = []
        for new_raw_data in request.new_raw_data_list:
            processed_data = self._data_process(new_raw_data)
            if processed_data is not None:  # Filter out unsupported message types
                new_message_dict_list.append(processed_data)

        # Check if there are valid messages to process
        if not new_message_dict_list:
            logger.warning(
                f"[ConvMemCellExtractor] No valid new messages to process (possibly all filtered out)"
            )
            status_control_result = StatusResult(should_wait=True)
            return (None, status_control_result)

        # === Force split check (token limit or message limit) ===
        # Calculate tokens for history + new messages combined
        accumulated_tokens = self._count_tokens(history_message_dict_list)
        new_tokens = self._count_tokens(new_message_dict_list)
        total_tokens = accumulated_tokens + new_tokens
        total_messages = len(history_message_dict_list) + len(new_message_dict_list)

        # Check if force split is needed (before calling LLM)
        needs_force_split = (
            total_tokens >= self.hard_token_limit
            or total_messages >= self.hard_message_limit
        )


        if needs_force_split and len(history_message_dict_list) >= 2:
            # Force split: create MemCell from history, new message starts next accumulation
            trigger_type = 'token_limit' if total_tokens >= self.hard_token_limit else 'message_limit'

            logger.debug(
                f"[ConvMemCellExtractor] Force split triggered: "
                f"tokens={total_tokens}/{self.hard_token_limit}, "
                f"messages={total_messages}/{self.hard_message_limit}"
            )

            # Parse timestamp from last history message
            ts_value = history_message_dict_list[-1].get("timestamp")
            timestamp = dt_from_iso_format(ts_value)
            participants = self._extract_participant_ids(history_message_dict_list)

            memcell = MemCell(
                user_id_list=request.user_id_list,
                original_data=history_message_dict_list,
                timestamp=timestamp,
                summary="",  # Empty summary for force split, will be filled by episode extractor
                group_id=request.group_id,
                participants=participants,
                type=self.raw_data_type,
            )


            runtime_memcell = comment_variable(
                {
                    "original_data": memcell.original_data,
                    "timestamp": memcell.timestamp.isoformat(),
                },
                variable_name="memcell",
                to_runtime=True,
                class_name="temporary_variable",
                category="temporary_variable",
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                comment=(
                    "This is a temporary variable. " 
                    "It contains a timestamp (`timestamp`) and the raw data (`original_data`). "
                    "It represents that the messages within the original data will be grouped together " 
                    "and later form a memory unit. "
                    "The timestamp corresponds to the last message's timestamp in the original data."
                ),
            )


            op_comment = (
                f"The memory system forcibly groups the first {len(history_message_dict_list)} " 
                "messages in the message buffer together, "
                f"because after introducing the {len(history_message_dict_list) + 1}-th message, "
                f"the total token count ({total_tokens}) reaches or exceeds " 
                f"the maximum token limit ({self.hard_token_limit})."
            ) if trigger_type == 'token_limit' else (
                f"The memory system forcibly groups the first {len(history_message_dict_list)} " 
                "messages in the message buffer together, "
                f"because after introducing the {len(history_message_dict_list) + 1}-th message, "
                f"the total number of messages ({total_messages}) reaches or exceeds " 
                f"the maximum message limit ({self.hard_message_limit})."
            )
            comment_op(
                inputs=[runtime_message_buffer],
                outputs=[runtime_memcell],
                op_name="evermemos.boundary_detection",
                category="boundary_detection", 
                comment=op_comment,
                metadata={
                    "trigger_type": trigger_type,
                    "total_tokens": total_tokens,
                    "total_messages": total_messages,
                    "hard_token_limit": self.hard_token_limit,
                    "hard_message_limit": self.hard_message_limit,
                },
            )

            # Record force split metrics
            record_boundary_detection(
                space_id=get_space_id_for_metrics(),
                raw_data_type=self.raw_data_type.value,
                result='force_split',
                trigger_type=trigger_type,
            )
            record_memcell_extracted(
                space_id=get_space_id_for_metrics(),
                raw_data_type=self.raw_data_type.value,
                trigger_type=trigger_type,
            )

            logger.debug(
                f"✅ Force split MemCell created: event_id={memcell.event_id}, "
                f"messages={len(history_message_dict_list)}, tokens={accumulated_tokens}"
            )

            return (memcell, StatusResult(should_wait=False))

        elif needs_force_split:
            # Needs split but not enough messages (single long message case)
            # Don't split, just log warning and continue normal flow
            logger.debug(
                f"[ConvMemCellExtractor] Exceeds limits but only {len(history_message_dict_list)} history messages, "
                f"not splitting single message. tokens={total_tokens}, messages={total_messages}"
            )

            no_split_marker = comment_variable(
                "NO_SPLIT",
                to_runtime=True,
                comment=(
                    "A marker indicating that the memory system explicitly decides not to "
                    "split the current message buffer into a new memory unit in this "
                    "step."
                ),
                class_name="no_split_marker",
                category="marker",
            ) 

            comment_op(
                inputs=[runtime_message_buffer],
                outputs=[no_split_marker],
                op_name="evermemos.boundary_detection",
                category="boundary_detection", 
                comment=(
                    "Although the total token usage or the number of messages in the current message buffer "
                    "exceeds the user-specified threshold, removing the last message leaves only a single message. "
                    "The memory system does not form a memory unit around a single message, so the memory extraction "  
                    "step is not performed."
                ),
                metadata={
                    "total_tokens": total_tokens,
                    "total_messages": total_messages,
                    "hard_token_limit": self.hard_token_limit,
                    "hard_message_limit": self.hard_message_limit, 
                }
            )

        # === Normal LLM-based boundary detection ===
        with comment_op_scope(
            op_name="evermemos.boundary_detection",
            category="boundary_detection", 
            comment=(
                "Use a large language model to determine whether " 
                "the newly introduced message indicates that the episode " 
                "described by the previous messages has ended." 
            ),
            metadata={
                "smart_mask_flag": request.smart_mask_flag,
                "llm_model": self.llm_provider.provider.model,
            }
        ):
            runtime_new_messages = comment_variable(
                new_message_dict_list,
                variable_name="new_messages",
                to_runtime=True,
                class_name="new_messages",
                category="new_messages",
                comment=(
                    "The newly introduced message or messages at the tail of the current "
                    "message buffer. The memory system uses them as the potential boundary trigger "
                    "where the large language model judges whether these new messages indicate that "
                    "the episode described by the earlier buffered messages has already ended."
                )
            )
            comment_link(
                source=runtime_message_buffer,
                target=runtime_new_messages,
                category="split_message_buffer", 
                comment=(
                    "The memory system derives these new messages from the current "
                    "message buffer by taking the newly appended tail portion. "
                    "These tail messages are treated as the potential boundary trigger "
                    "that may signal the previous episode has ended."
                ),
            )
            if request.smart_mask_flag:
                runtime_history_messages = comment_variable(
                    history_message_dict_list[:-1],
                    variable_name="history_messages",
                    to_runtime=True,
                    class_name="history_messages",
                    category="history_messages",
                    comment=(
                        "These are the history messages that are actually shown to the "
                        "boundary-detection model when smart mask is enabled. "
                        "Specifically, the memory system excludes the most recent historical message "
                        "from this variable and holds it out as an overlap candidate. "
                        "If a boundary is later detected, that held-out message will still be "
                        "included in the just-created memory unit and also retained in the buffer "
                        "for the next memory unit, so the adjacent memory unit shares a small overlap "
                        "instead of splitting too sharply."
                    )
                ) 
                comment_link(
                    source=runtime_message_buffer,
                    target=runtime_history_messages, 
                    category="split_message_buffer", 
                    comment=(
                        "When smart mask is enabled, the memory system derives the "
                        "history messages used for boundary detection from the current "
                        "message buffer by dropping the most recent historical message. "
                        "That held-out message is reserved as an overlap candidate, so "
                        "if a boundary is detected it can still belong to both the "
                        "ending memory unit and the next buffer."
                    ),
                )
                boundary_detection_result = await self._detect_boundary(
                    conversation_history=history_message_dict_list[:-1],
                    new_messages=new_message_dict_list,
                )
            else:
                runtime_history_messages = comment_variable(
                    history_message_dict_list,
                    variable_name="history_messages",
                    to_runtime=True,
                    class_name="history_messages",
                    category="history_messages",
                    comment=(
                        "These are the history messages that are shown to the "
                        "boundary-detection model when smart mask is disabled. "
                        "In this case the memory system does not hold out any overlap "
                        "candidate, so all currently accumulated historical messages "
                        "are provided directly to the model."
                    )
                )
                comment_link(
                    source=runtime_message_buffer,
                    target=runtime_history_messages,
                    category="split_message_buffer",
                    comment=(
                        "When smart mask is disabled, the memory system derives the "
                        "history messages used for boundary detection from the current "
                        "message buffer without dropping any historical message. "
                        "All accumulated historical messages are passed directly to "
                        "the boundary-detection model."
                    ),
                )
                boundary_detection_result = await self._detect_boundary(
                    conversation_history=history_message_dict_list,
                    new_messages=new_message_dict_list,
                )
            should_end = boundary_detection_result.should_end
            should_wait = boundary_detection_result.should_wait
            reason = boundary_detection_result.reasoning

            status_control_result = StatusResult(should_wait=should_wait)

            runtime_boundary_detection_result = None 
            if tracing_ctx is not None and is_tracing_enabled():
                runtime_boundary_detection_result = tracing_ctx.get_variable(
                    "boundary_detection_result"
                )

            if should_end:
                # Parse timestamp
                ts_value = history_message_dict_list[-1].get("timestamp")
                timestamp = dt_from_iso_format(ts_value)
                participants = self._extract_participant_ids(history_message_dict_list)

                # Generate summary (prioritize topic summary from boundary detection)
                fallback_text = ""
                if new_message_dict_list:
                    last_msg = new_message_dict_list[-1]
                    if isinstance(last_msg, dict):
                        fallback_text = last_msg.get("content") or ""
                    elif isinstance(last_msg, str):
                        fallback_text = last_msg
                summary_text = boundary_detection_result.topic_summary or (
                    fallback_text.strip()[:200] if fallback_text else "Conversation segment"
                )

                # Create basic MemCell (without episode, foresight, event_log, embedding)
                memcell = MemCell(
                    user_id_list=request.user_id_list,
                    original_data=history_message_dict_list,
                    timestamp=timestamp,
                    summary=summary_text,
                    group_id=request.group_id,
                    participants=participants,
                    type=self.raw_data_type,
                )


                runtime_memcell = comment_variable(
                    {
                        "original_data": memcell.original_data,
                        "timestamp": memcell.timestamp.isoformat(),
                        "summary": memcell.summary,
                    },
                    variable_name="memcell",
                    to_runtime=True,
                    class_name="temporary_variable",
                    category="temporary_variable",
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    comment=(
                        "This is a temporary variable representing the newly formed "
                        "memory-unit candidate after the memory system decides that "
                        "the current scenario should end. It stores the grouped raw "
                        "messages (`original_data`), the timestamp of the last history "
                        "message (`timestamp`), and the initial summary text (`summary`) "
                        "that will be attached to the new memory unit."
                    ),
                )
                comment_op(
                    inputs=[runtime_message_buffer, runtime_boundary_detection_result],
                    outputs=[runtime_memcell],
                    comment=(
                        "Based on the current message buffer and the structured "
                        "boundary-detection result, the memory system groups the "
                        "history messages into a new temporary memory-unit candidate. "
                        "Its summary comes from the model-provided topic summary when "
                        "available, and otherwise falls back to a short summary "
                        "derived from the newest message."
                    ),
                    reuse_op=True,
                )

                # Record MemCell extraction metric
                record_memcell_extracted(
                    space_id=get_space_id_for_metrics(),
                    raw_data_type=self.raw_data_type.value,
                    trigger_type='llm',
                )

                logger.debug(
                    f"✅ Successfully created basic MemCell: event_id={memcell.event_id}, "
                    f"participants={len(participants)}, messages={len(history_message_dict_list)}"
                )

                # Delete history messages, new messages and boundary detection result 
                # from the tracing context as we don't need them anymore. 
                if tracing_ctx is not None and is_tracing_enabled():
                    for name in ["history_messages", "new_messages", "boundary_detection_result"]:
                        tracing_ctx.remove_variable(name)

                return (memcell, status_control_result)
            elif should_wait:
                logger.debug(f"⏳ Waiting for more messages: {reason}")

                no_split_marker = comment_variable(
                    "NO_SPLIT",
                    to_runtime=True,
                    comment=(
                        "A marker indicating that the memory system explicitly decides not to "
                        "split the current message buffer into a new memory unit in this "
                        "step."
                    ),
                    class_name="no_split_marker",
                    category="marker",
                ) 
                comment_link(
                    source=runtime_boundary_detection_result,
                    target=no_split_marker,
                    comment=(
                        "Based on the structured boundary-detection result, the memory "
                        "system decides not to split the current message buffer into a "
                        "new memory unit yet and instead keeps waiting for more messages. "
                        f"The model's reason is:\n {reason}"
                    ),
                )

        # Delete history messages, new messages and boundary detection result 
        # from the tracing context as we don't need them anymore. 
        if tracing_ctx is not None and is_tracing_enabled():
            for name in ["history_messages", "new_messages", "boundary_detection_result"]:
                tracing_ctx.remove_variable(name)

        return (None, status_control_result)

    def _data_process(self, raw_data: RawData) -> Dict[str, Any]:
        """Process raw data, including message type filtering and preprocessing"""
        content = (
            raw_data.content.copy()
            if isinstance(raw_data.content, dict)
            else raw_data.content
        )

        # Get message type
        msg_type = content.get('msgType') if isinstance(content, dict) else None

        # Define supported message types and corresponding placeholders
        SUPPORTED_MSG_TYPES = {
            1: None,  # TEXT - keep original text
            2: "[Image]",  # PICTURE
            3: "[Video]",  # VIDEO
            4: "[Audio]",  # AUDIO
            5: "[File]",  # FILE - keep original text (text and file in same message)
            6: "[File]",  # FILES
        }

        if isinstance(content, dict) and msg_type is not None:
            # Check if it's a supported message type
            if msg_type not in SUPPORTED_MSG_TYPES:
                # Unsupported message type, skip directly (returning None will be handled at upper level)
                logger.warning(
                    f"[ConvMemCellExtractor] Skipping unsupported message type: {msg_type}"
                )
                return None

            # Preprocess non-text messages
            placeholder = SUPPORTED_MSG_TYPES[msg_type]
            if placeholder is not None:
                # Replace message content with placeholder
                content = content.copy()
                content['content'] = placeholder
                logger.debug(
                    f"[ConvMemCellExtractor] Message type {msg_type} converted to placeholder: {placeholder}"
                )

        return content
