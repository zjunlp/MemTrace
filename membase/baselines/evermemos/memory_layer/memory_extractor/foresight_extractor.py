"""
Foresight Extractor - Based on associative prediction method
Generate predictions of potential impacts on user's future life and decisions from MemCell
"""

import json
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime, timedelta
import inspect
from smartcomment import (
    current_context, 
    is_tracing_enabled, 
    comment_link,
    comment_op, 
    comment_variable,
)

from memory_layer.prompts import get_prompt_by
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.base_memory_extractor import (
    MemoryExtractor,
    MemoryExtractRequest,
)
from api_specs.memory_types import MemoryType, MemCell, Foresight, BaseMemory
from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone

logger = get_logger(__name__)


class ForesightExtractor(MemoryExtractor):
    """
    Foresight Extractor - Based on associative prediction method

    Supports conversation mode:
    - Generate associations based on raw conversation transcript text (assistant scene).

    New strategy implementation:
    1. Based on content, large model associates 10 potential impacts on user's subsequent life and decisions
    2. Each association considers its possible duration
    3. Focus on personal-level impacts for the user

    Main methods:
    - generate_foresights_for_conversation(): Generate foresights from raw conversation text
    """

    def __init__(
        self, 
        llm_provider: LLMProvider,
        embedding_provider: Literal["deepinfra", "vllm"] = "vllm",
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_dims: int = 384,
    ):
        """
        Initialize foresight extractor

        Args:
            llm_provider: LLM provider
            embedding_provider: Embedding provider type.
            embedding_model: Embedding model name.
            embedding_api_key: API key for embedding service.
            embedding_base_url: API base URL for embedding service.
            embedding_dims: Embedding vector dimension.
        """
        super().__init__(MemoryType.FORESIGHT)
        self.llm_provider = llm_provider

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

        logger.info("Foresight extractor initialized (associative prediction mode)")

    async def extract_memory(
        self, request: MemoryExtractRequest
    ) -> Optional[BaseMemory]:
        """
        Implement abstract base class required extract_memory method

        Note: ForesightExtractor should not directly use extract_memory method
        Use generate_foresights_for_conversation instead

        Args:
            request: Memory extraction request

        Returns:
            None - This method should not be called
        """
        raise NotImplementedError(
            "ForesightExtractor should not directly use extract_memory method."
            "Please use generate_foresights_for_conversation method."
        )

    async def generate_foresights_for_conversation(
        self,
        conversation_text: str,
        timestamp: datetime,
        user_id: str,
        user_name: Optional[str] = None,
        group_id: Optional[str] = None,
        ori_event_id_list: Optional[List[str]] = None,
    ) -> List[Foresight]:
        """
        Generate foresight association predictions from raw conversation text.

        Args:
            conversation_text: Raw conversation transcript text
            timestamp: Conversation timestamp (used as base time)
            user_id: Target user id
            user_name: Optional user display name
            group_id: Optional group id
            ori_event_id_list: Optional original event id list

        Returns:
            List of foresight items (up to 10 items), including time information
        """
        # Collect errors. 
        errors = [] 

        runtime_memcell_result = None 
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            # Get the runtime variable of the latest memory-unit.
            runtime_memcell_result = tracing_ctx.get_variable("memcell_result")

        runtime_prompt = None
        runtime_response = None
        runtime_prompt_template = None

        # Maximum 5 retries
        for retry in range(5):
            try:
                if retry == 0:
                    logger.info(
                        f"🎯 Generating foresight associations for conversation: user_id={user_id}"
                    )
                else:
                    logger.info(
                        f"🎯 Generating foresight associations for conversation: user_id={user_id}, retry {retry}/5"
                    )

                # Build prompt (static prompt template via PromptManager)
                prompt_template = get_prompt_by("FORESIGHT_GENERATION_PROMPT")
                prompt = prompt_template.format(
                    USER_ID=user_id,
                    USER_NAME=user_name,
                    CONVERSATION_TEXT=conversation_text,
                )

                if runtime_prompt_template is None:
                    runtime_prompt_template = comment_variable(
                        prompt_template,
                        to_runtime=True,
                        id_strategy=lambda _: "foresight-generation-prompt-template",
                        category="prompt",
                        comment=(
                            "A prompt template that asks the large language model to "
                            "generate foresight memories from the current refined "
                            "memory-unit."
                        ),
                        metadata={
                            "op_type": "foresight_generation",
                        },
                    )

                if runtime_prompt is None:
                    runtime_prompt = comment_variable(
                        prompt,
                        to_runtime=True,
                        class_name="foresight_generation_prompt",
                        category="prompt",
                        comment=(
                            "A concrete foresight-generation prompt assembled from "
                            "the refined memory-unit and the foresight prompt template."
                        ),
                    )

                comment_op(
                    inputs=[
                        runtime_memcell_result,
                        runtime_prompt_template,
                    ],
                    outputs=[runtime_prompt],
                    comment=(
                        "The memory system assembles the concrete foresight-generation "
                        "prompt from the refined memory-unit and the foresight prompt "
                        "template."
                    ),
                    reuse_op=True,
                )

                # Call LLM to generate associations
                logger.debug(
                    f"📝 Starting LLM call to generate foresight associations, prompt length: {len(prompt)}"
                )
                response = await self.llm_provider.generate(
                    prompt=prompt, temperature=0.3
                )
                logger.debug(
                    f"✅ LLM call completed, response length: {len(response) if response else 0}"
                )

                # Parse JSON response
                start_time = self._extract_start_time_from_timestamp(timestamp)
                foresights = await self._parse_foresights_response(
                    response,
                    start_time=start_time,
                    user_id=user_id,
                    timestamp=timestamp,
                    ori_event_id_list=ori_event_id_list or [],
                    group_id=group_id,
                )

                # Validate at least 1 item is returned
                if len(foresights) == 0:
                    raise ValueError("LLM returned empty foresight list")

                # Ensure at most 10 items are returned
                if len(foresights) > 10:
                    foresights = foresights[:10]
                elif len(foresights) < 4:
                    logger.warning(
                        f"Generated foresight associations less than 4, actual count: {len(foresights)}"
                    )


                runtime_response = comment_variable(
                    response,
                    to_runtime=True,
                    class_name="raw_foresight_generation_response",
                    category="llm_response",
                    comment=(
                        "A raw foresight-generation response from the large language "
                        "model."
                    ),
                )
                comment_link(
                    source=runtime_prompt,
                    target=runtime_response,
                    comment=(
                        "The large language model generates foresight memories based "
                        "on the concrete foresight-generation prompt."
                    ),
                )

                logger.info(
                    f"✅ Successfully generated {len(foresights)} foresight associations"
                )
                for i, memory in enumerate(foresights[:3], 1):
                    logger.info(f"  Association {i}: {memory.foresight}")

                foresight_snapshots = [
                    {
                        "foresight": item.foresight,
                        "evidence": item.evidence,
                        "start_time": item.start_time,
                        "end_time": item.end_time,
                        "duration_days": item.duration_days,
                    }
                    for item in foresights
                ]
                runtime_foresights = comment_variable(
                    foresight_snapshots,
                    variable_name="foresights",
                    to_runtime=True,
                    class_name="foresight_list",
                    category="llm_response",
                    encoding_fn=lambda value: json.dumps(
                        value,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    comment=(
                        "A structured list of foresight memories parsed from the "
                        "large language model's response."
                    ),
                )
                comment_link(
                    source=runtime_response,
                    target=runtime_foresights,
                    comment=(
                        "The memory system parses the raw foresight-generation "
                        "response into a structured list of foresight memories."
                    ),
                    edge_metadata={
                        "source_code": (
                            inspect.getsource(self._parse_foresights_response)
                            + "\n\n"
                            + inspect.getsource(self._clean_date_string)
                        )
                    }, 
                )

                return foresights

            except Exception as e:
                logger.warning(f"Foresight generation retry {retry+1}/5: {e}")
                errors.append(
                    f"Attempt {retry + 1} fails because foresight generation raises "
                    f"{e.__class__.__name__}: {e}."
                )
                if retry == 4:
                    logger.error(f"Foresight generation failed after 5 retries")

                    failed_marker = comment_variable(
                        "FAILED",
                        to_runtime=True,
                        category="marker",
                        class_name="failed_marker",
                        comment=(
                            "A marker indicating that the process fails."
                        ),
                    )
                    comment_link(
                        source=runtime_prompt,
                        target=failed_marker,
                        comment=(
                            "The memory system fails to generate foresight memories "
                            "after five attempts."
                        ),
                        edge_metadata={
                            "error": (
                                "The memory system attempts foresight generation five "
                                "times and fails to obtain a valid structured foresight "
                                "list each time.\n\n"
                                + "\n\n".join(errors)
                            ),
                        },
                    )

                    return []
                continue

        # All retries exhausted, return default result
        logger.error(
            f"[ForesightExtractor] All 5 retries exhausted for foresight generation, returning default (foresights=[])"
        )
        return []

    @staticmethod
    def _clean_date_string(date_str: Optional[str]) -> Optional[str]:
        """Clean date string, remove invalid characters and validate date validity

        Args:
            date_str: Original date string

        Returns:
            Cleaned date string, return None if invalid
        """
        if not date_str or not isinstance(date_str, str):
            return None

        import re

        # Keep only digits and hyphens, remove other characters (e.g., Chinese, spaces, etc.)
        cleaned = re.sub(r'[^\d\-]', '', date_str)

        # Validate format is YYYY-MM-DD
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', cleaned):
            logger.warning(
                f"Invalid time format, does not match YYYY-MM-DD: original='{date_str}', cleaned='{cleaned}'"
            )
            return None

        # Validate date values are valid (month 1-12, day 1-31, etc.)
        try:
            year, month, day = map(int, cleaned.split('-'))
            # Use datetime to validate date validity
            datetime(year, month, day)
            return cleaned
        except ValueError as e:
            logger.warning(f"Invalid date value: '{cleaned}', error: {e}")
            return None

    async def _parse_foresights_response(
        self,
        response: str,
        start_time: Optional[str] = None,
        user_id: str = "",
        timestamp: Optional[datetime] = None,
        ori_event_id_list: Optional[List[str]] = None,
        group_id: Optional[str] = None,
    ) -> List[Foresight]:
        """
        Parse LLM's JSON response to extract foresight association list

        Args:
            response: LLM response text
            start_time: Start time, format YYYY-MM-DD
            user_id: User ID for the foresight
            timestamp: Timestamp for the foresight
            ori_event_id_list: Original event ID list
            group_id: Group ID

        Returns:
            List of foresight association items
        """
        try:
            # First try to extract JSON from code block
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_str = response[start:end].strip()
                    data = json.loads(json_str)
                else:
                    data = json.loads(response)
            else:
                # Try to parse entire response as JSON array
                data = json.loads(response)

            # Ensure data is a list
            if isinstance(data, list):
                foresights = []

                # First collect all data to be processed
                items_to_process = []
                for item in data:
                    content = item.get('content', '')
                    evidence = item.get('evidence', '')

                    # Use passed start_time or LLM-provided time
                    item_start_time = item.get('start_time', start_time)
                    item_end_time = item.get('end_time')
                    item_duration_days = item.get('duration_days')

                    # Clean time format (prevent LLM outputting incorrect format)
                    item_start_time = self._clean_date_string(item_start_time)
                    item_end_time = self._clean_date_string(item_end_time)

                    # Smart time calculation: prioritize LLM-provided time information
                    if item_start_time:
                        # If LLM provides duration_days but no end_time, calculate end_time
                        if item_duration_days and not item_end_time:
                            item_end_time = self._calculate_end_time_from_duration(
                                item_start_time, item_duration_days
                            )
                        # If LLM provides end_time but no duration_days, calculate duration_days
                        elif item_end_time and not item_duration_days:
                            item_duration_days = self._calculate_duration_days(
                                item_start_time, item_end_time
                            )
                        # If LLM provides neither, keep as None (no additional extraction)

                    items_to_process.append(
                        {
                            'foresight': content,
                            'evidence': evidence,
                            'start_time': item_start_time,
                            'end_time': item_end_time,
                            'duration_days': item_duration_days,
                        }
                    )

                # Batch compute embeddings for all content (performance optimization)
                vs = self._vectorize_service
                contents = [item['foresight'] for item in items_to_process]
                vectors_batch = await vs.get_embeddings(
                    contents
                )  # Use get_embeddings (List[str])

                # Create Foresight objects
                for i, item_data in enumerate(items_to_process):
                    # Handle embedding: could be numpy array or already list
                    vector = vectors_batch[i]
                    if hasattr(vector, 'tolist'):
                        vector = vector.tolist()
                    elif not isinstance(vector, list):
                        vector = list(vector)

                    memory_item = Foresight(
                        memory_type=MemoryType.FORESIGHT,
                        user_id=user_id,
                        timestamp=timestamp or get_now_with_timezone(),
                        ori_event_id_list=ori_event_id_list or [],
                        group_id=group_id,
                        foresight=item_data['foresight'],
                        evidence=item_data['evidence'],
                        start_time=item_data['start_time'],
                        end_time=item_data['end_time'],
                        duration_days=item_data['duration_days'],
                        vector=vector,
                        vector_model=vs.get_model_name(),
                    )
                    foresights.append(memory_item)

                return foresights
            else:
                logger.error(f"Response is not in JSON array format: {data}")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Response content: {response[:200]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing foresight response: {e}")
            return []

    def _extract_start_time_from_timestamp(self, timestamp: datetime) -> str:
        """
        Extract start time from MemCell's timestamp field

        Args:
            timestamp: MemCell timestamp

        Returns:
            Start time string in YYYY-MM-DD format
        """
        return timestamp.strftime('%Y-%m-%d')

    def _calculate_end_time_from_duration(
        self, start_time: str, duration_days: int
    ) -> Optional[str]:
        """
        Calculate end time based on start time and duration

        Args:
            start_time: Start time in YYYY-MM-DD format
            duration_days: Duration in days

        Returns:
            End time string in YYYY-MM-DD format, return None if calculation fails
        """
        try:
            if not start_time or duration_days is None:
                return None

            start_date = datetime.strptime(start_time, '%Y-%m-%d')
            end_date = start_date + timedelta(days=duration_days)

            return end_date.strftime('%Y-%m-%d')

        except Exception as e:
            logger.error(f"Error calculating end time from duration: {e}")
            return None

    def _calculate_duration_days(self, start_time: str, end_time: str) -> Optional[int]:
        """
        Calculate duration (in days)

        Args:
            start_time: Start time in YYYY-MM-DD format
            end_time: End time in YYYY-MM-DD format

        Returns:
            Duration in days, return None if calculation fails
        """
        try:
            if not start_time or not end_time:
                return None

            start_date = datetime.strptime(start_time, '%Y-%m-%d')
            end_date = datetime.strptime(end_time, '%Y-%m-%d')

            duration = end_date - start_date
            return duration.days

        except Exception as e:
            logger.error(f"Error calculating duration: {e}")
            return None
