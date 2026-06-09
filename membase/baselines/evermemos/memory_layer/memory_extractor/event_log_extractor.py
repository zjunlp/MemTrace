"""
Event Log Extractor for EverMemOS

This module extracts atomic event logs from episode memories for optimized retrieval.
Each event log contains a time and a list of atomic facts extracted from the episode.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from functools import partial
import json
import re
import inspect
from smartcomment import (
    comment_link, 
    comment_op, 
    comment_variable, 
    current_context,
    is_tracing_enabled, 
)

from memory_layer.prompts import get_prompt_by
from memory_layer.llm.llm_provider import LLMProvider
from common_utils.datetime_utils import get_now_with_timezone, from_iso_format
from api_specs.memory_types import EventLog, MemoryType, MemCell

from core.observation.logger import get_logger

logger = get_logger(__name__)


class EventLogExtractor:
    """
    Extractor for converting episode memories into structured event logs.

    The event log format is optimized for retrieval:
    - Time field provides temporal context
    - Atomic facts are independent, searchable units
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        event_log_prompt: Optional[str] = None,
        embedding_provider: Literal["deepinfra", "vllm"] = "vllm",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dims: int = 384,
    ):
        """
        Initialize the event log extractor.

        Args:
            llm_provider: LLM provider for generating event logs
            event_log_prompt: Optional custom event log prompt
            embedding_provider: Embedding provider type.
            embedding_model: Embedding model name.
            embedding_api_key: API key for embedding service.
            embedding_base_url: API base URL for embedding service.
            embedding_dims: Embedding vector dimension.
        """
        self.llm_provider = llm_provider

        # Use custom prompt or get default via PromptManager
        self.event_log_prompt = event_log_prompt or get_prompt_by("EVENT_LOG_PROMPT")

        # Import vectorize service. 
        from agentic_layer.vectorize_service import (
            HybridVectorizeService, 
            HybridVectorizeConfig, 
        )
        # Create `HybridVectorizeConfig` from input parameters.
        # We don't use fallback here.
        vectorize_config = HybridVectorizeConfig(
            primary_provider=embedding_provider,
            primary_api_key=embedding_api_key or "",
            primary_base_url=embedding_base_url or "",
            model=embedding_model or "",
            dimensions=embedding_dims,
            enable_fallback=False, 
        )
        
        self._vectorize_service = HybridVectorizeService(config=vectorize_config)

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        Parse timestamp into datetime object
        Supports multiple formats: numeric timestamp, ISO string, datetime object, etc.

        Args:
            timestamp: Timestamp, can be in multiple formats

        Returns:
            datetime: Parsed datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                if timestamp.isdigit():
                    return datetime.fromtimestamp(int(timestamp))
                else:
                    # Try parsing ISO format
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                logger.error(f"Failed to parse timestamp: {timestamp}")
                return get_now_with_timezone()
        else:
            logger.error(f"Unknown timestamp format: {timestamp}")
            return get_now_with_timezone()

    def _format_timestamp(self, dt: datetime) -> str:
        """
        Format datetime into required string format for event logs
        Format: "March 10, 2024(Sunday) at 2:00 PM"

        Args:
            dt: datetime object

        Returns:
            str: Formatted time string
        """
        weekday = dt.strftime("%A")  # Monday, Tuesday, etc.
        month_day_year = dt.strftime("%B %d, %Y")  # March 10, 2024
        time_of_day = dt.strftime("%I:%M %p")  # 2:00 PM
        return f"{month_day_year}({weekday}) at {time_of_day}"

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response returned by LLM
        Supports multiple formats: plain JSON, JSON code block, etc.

        Args:
            response: Raw response from LLM

        Returns:
            Dict: Parsed JSON object

        Raises:
            ValueError: If response cannot be parsed
        """
        # 1. Try extracting JSON from code block
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end > start:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # 2. Try extracting from any code block
        if '```' in response:
            start = response.find('```') + 3
            # Skip language identifier (if any)
            if response[start : start + 10].strip().split()[0].isalpha():
                start = response.find('\n', start) + 1
            end = response.find('```', start)
            if end > start:
                json_str = response[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # 3. Try extracting JSON object containing event_log
        json_match = re.search(
            r'\{[^{}]*"event_log"[^{}]*\{[^{}]*"time"[^{}]*"atomic_fact"[^{}]*\}[^{}]*\}',
            response,
            re.DOTALL,
        )
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 4. Try parsing entire response directly
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # 5. If all fail, raise exception
        logger.error(f"Unable to parse LLM response: {response[:200]}...")
        raise ValueError(f"Unable to parse LLM response into valid JSON format")

    async def _extract_event_log(
        self,
        input_text: str,
        timestamp: Any,
        user_id: str = "",
        ori_event_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
    ) -> Optional[EventLog]:
        """
        Extract event log from episode memory

        Args:
            episode_text: Text content of episode memory
            timestamp: Timestamp of episode (can be in multiple formats)
            user_id: User ID for the event log
            ori_event_id_list: Original event ID list
            group_id: Group ID

        Returns:
            EventLog: Extracted event log, return None if extraction fails
        """

        # 1. Parse and format timestamp
        dt = self._parse_timestamp(timestamp)
        time_str = self._format_timestamp(dt)

        # 2. Build prompt (using instance variable self.event_log_prompt)
        prompt = self.event_log_prompt.replace("{{INPUT_TEXT}}", input_text)
        prompt = prompt.replace("{{TIME}}", time_str)

        runtime_prompt = runtime_memcell_result = None 
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_memcell_result = tracing_ctx.get_variable("memcell_result")

            candidates = tracing_ctx.query_variables(name_pattern="event_extraction_prompt")
            assert len(candidates) <= 1, (
                "Multiple event extraction prompts are found in the tracing context."
            )

            if candidates:
                runtime_prompt = candidates[0]
            else:
                runtime_prompt = comment_variable(
                    prompt,
                    variable_name="event_extraction_prompt",
                    to_runtime=True,
                    class_name="event_extraction_prompt",
                    category="prompt",
                    comment=(
                        "A concrete event-log-extraction prompt assembled from the "
                        "refined memory-unit and the event-log prompt template."
                    ),
                )

            comment_op(
                inputs=[
                    runtime_memcell_result,
                    (
                        self.event_log_prompt,
                        {
                            "id_strategy": lambda _: "event-log-prompt-template",
                            "category": "prompt",
                            "comment": (
                                "A prompt template that asks the large language model "
                                "to extract a retrieval-oriented event log from the "
                                "current refined memory-unit."
                            ),
                            "metadata": {
                                "op_type": "event_log_extraction",
                            },
                        }
                    ),
                ],
                outputs=[runtime_prompt],
                comment=(
                    "The memory system assembles the concrete event-log-extraction "
                    "prompt from the refined memory-unit and the event-log prompt "
                    "template."
                ),
                reuse_op=True,
            )

        # 3. Call LLM to generate event log
        response = await self.llm_provider.generate(prompt)

        # 4. Parse LLM response
        data = self._parse_llm_response(response)

        # 5. Validate response format
        if "event_log" not in data:
            raise ValueError(f"Missing 'event_log' field in LLM response")

        event_log_data = data["event_log"]

        # Validate required fields: time and atomic_fact must exist
        if "time" not in event_log_data or not event_log_data["time"]:
            raise ValueError("Missing time field in event_log")
        if "atomic_fact" not in event_log_data:
            raise ValueError("Missing atomic_fact field in event_log")

        # Validate atomic_fact is a list
        if not isinstance(event_log_data["atomic_fact"], list):
            raise ValueError(
                f"atomic_fact is not a list: {type(event_log_data['atomic_fact'])}"
            )

        # Validate atomic_fact is not empty
        if len(event_log_data["atomic_fact"]) == 0:
            raise ValueError("atomic_fact list is empty")

        # 6. Batch generate embedding for all atomic_fact (performance optimization)
        vectorize_service = self._vectorize_service

        # Batch compute embeddings (using get_embeddings, accepts List[str])
        fact_embeddings_batch = await vectorize_service.get_embeddings(
            event_log_data["atomic_fact"]
        )

        # Convert to list format
        fact_embeddings = [
            emb.tolist() if hasattr(emb, 'tolist') else emb
            for emb in fact_embeddings_batch
        ]

        # 7. Create EventLog object with Memory base class fields
        event_log = EventLog(
            memory_type=MemoryType.EVENT_LOG,
            user_id=user_id,
            timestamp=dt,
            ori_event_id_list=ori_event_id_list or [],
            group_id=group_id,
            time=event_log_data["time"],
            atomic_fact=event_log_data["atomic_fact"],
            fact_embeddings=fact_embeddings,
        )


        event_log_snapshot = {
            "atomic_fact": event_log.atomic_fact,
        } 
        runtime_response = comment_variable(
            response,
            to_runtime=True,
            class_name="raw_event_log_extraction_response",
            category="llm_response",
            comment=(
                "A raw event-log-extraction response from the large language model."
            ),
        )
        runtime_event_log_snapshot = comment_variable(
            event_log_snapshot,
            variable_name="event_log",
            to_runtime=True,
            class_name="event_log",
            category="llm_response",
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            comment=(
                "A structured event-log parsed from the large language "
                "model's response."
            ),
        )
        comment_link(
            source=runtime_prompt,
            target=runtime_response,
            comment=(
                "The large language model generates a retrieval-oriented event "
                "log based on the concrete event-log-extraction prompt."
            ),
        )
        comment_link(
            source=runtime_response,
            target=runtime_event_log_snapshot,
            comment=(
                "The memory system parses the raw event-log-extraction response "
                "into a structured event-log."
            ),
            edge_metadata={
                "source_code": inspect.getsource(self._parse_llm_response),
            },
        )

        logger.debug(
            f"✅ Successfully extracted event log, containing {len(event_log.atomic_fact)} atomic facts (embeddings generated)"
        )
        return event_log

    async def extract_event_log(
        self,
        memcell: MemCell,
        timestamp: Any,
        user_id: str = "",
        ori_event_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
    ) -> Optional[EventLog]:
        """
        Extract event log
        """
        input_text = ""
        for data in memcell.original_data:
            speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
            content = data['content']
            msg_ts = data.get('timestamp')
            ts_str = from_iso_format(msg_ts)
            input_text += f"[{ts_str}] {speaker}: {content}\n"

        # Episode Mode
        # if memcell.episode:
        #    input_text = memcell.episode
        #    timestamp = memcell.timestamp

        errors = [] 

        for retry in range(5):
            try:
                return await self._extract_event_log(
                    input_text,
                    timestamp,
                    user_id=user_id,
                    ori_event_id_list=ori_event_id_list,
                    group_id=group_id,
                )
            except Exception as e:
                logger.warning(f"Retrying to extract event log {retry+1}/5: {e}")
                errors.append(
                    f"Attempt {retry + 1} fails because event log extraction raises " 
                    f"{e.__class__.__name__}: {e}."
                )
                if retry == 4:
                    logger.error(f"Failed to extract event log after 5 retries")

                    runtime_prompt = None
                    tracing_ctx = current_context()
                    if tracing_ctx is not None and is_tracing_enabled():
                        runtime_prompt = tracing_ctx.get_variable("event_extraction_prompt")

                    failed_marker = comment_variable(
                        "FAILED",
                        to_runtime=True,
                        category="marker",
                        class_name="failed_marker",
                        comment=(
                            "A marker indicating that the process fails."
                        ),
                    )
                    if runtime_prompt is not None:
                        comment_link(
                            source=runtime_prompt,
                            target=failed_marker,
                            comment=(
                                "The memory system fails to extract an event-log "
                                "after five attempts."
                            ),
                            edge_metadata={
                                "error": (
                                    "The memory system attempts event-log extraction five "
                                    "times and fails to obtain a valid structured event-log "
                                    "each time.\n\n"
                                    + "\n\n".join(errors)
                                ),
                            },
                        )

                    raise Exception(f"Failed to extract event log: {e}")
                continue


def format_event_log_for_bm25(event_log: EventLog) -> str:
    """
    Format event log for BM25 retrieval
    Use only atomic_fact field, concatenate all atomic facts into a single string

    Args:
        event_log: EventLog object

    Returns:
        str: Text for BM25 retrieval
    """
    if not event_log or not event_log.atomic_fact:
        return ""

    # Directly concatenate all atomic facts, separated by spaces
    return " ".join(event_log.atomic_fact)


def format_event_log_for_rerank(event_log: EventLog) -> str:
    """
    Format event log for rerank
    Use "time" + "：" + "atomic_fact" concatenation

    Args:
        event_log: EventLog object

    Returns:
        str: Text for rerank
    """
    if not event_log:
        return ""

    # Concatenate time and atomic facts
    time_part = event_log.time or ""
    facts_part = " ".join(event_log.atomic_fact) if event_log.atomic_fact else ""

    if time_part and facts_part:
        return f"{time_part}：{facts_part}"
    elif time_part:
        return time_part
    elif facts_part:
        return facts_part
    else:
        return ""
