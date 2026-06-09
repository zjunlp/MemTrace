import asyncio
import pickle
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
import json
from smartcomment import (
    current_context, 
    is_tracing_enabled, 
    comment_variable,
    comment_op,
    comment_mutation, 
    comment_op_scope, 
) 

from api_specs.memory_types import MemCell, RawDataType
from memory_layer.memcell_extractor.base_memcell_extractor import RawData
from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    ConversationMemCellExtractRequest,
)
from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractRequest
from memory_layer.memory_extractor.episode_memory_extractor import EpisodeMemoryExtractor
from memory_layer.memory_extractor.event_log_extractor import EventLogExtractor
from memory_layer.memory_extractor.foresight_extractor import ForesightExtractor
from memory_layer.memory_extractor.profile_memory.types import ProfileMemory
from memory_layer.cluster_manager import ClusterManager, ClusterManagerConfig, ClusterState
from memory_layer.profile_manager import ProfileManager, ProfileManagerConfig
from memory_layer.llm.llm_provider import LLMProvider
from core.oxm.mongo.mongo_utils import generate_object_id_str
from common_utils.datetime_utils import from_iso_format

from online_memory.config import OnlineMemoryManagerConfig
from online_memory.index_manager import InMemoryIndexManager
from online_memory.retriever import OnlineRetriever

from typing import (
    Any, 
    Dict, 
    List, 
    Optional,
    Tuple,
)


class OnlineMemoryManager:
    """
    Online Memory Manager for incremental message processing.
    
    This manager processes messages one by one, automatically detecting
    conversation boundaries and extracting memories incrementally.
    
    Features:
    - Single message processing 
    - Automatic boundary detection
    - MemCell extraction on boundary detection
    - Incremental clustering
    - Profile updates for user roles only
    """
    
    def __init__(self, config: OnlineMemoryManagerConfig) -> None:
        """Initialize the Online Memory Manager."""
        self.config = config
        
        # Message history buffer (for boundary detection)
        self._message_buffer: List[Dict[str, Any]] = []
        
        # Extracted MemCells (using existing MemCell type)
        self._memcells: List[MemCell] = []
        # Existing user profiles 
        self._profiles: Dict[str, ProfileMemory] = {}
        
        # Statistics
        self._stats = {
            "total_messages": 0,
            "boundaries_detected": 0,
            "memcells_extracted": 0,
            "profiles_updated": 0,
        }
        
        # Initialize LLM provider
        llm_cfg = self.config.llm
        self._llm_provider = LLMProvider(
            provider_type=llm_cfg.provider,
            model=llm_cfg.model,
            api_key=llm_cfg.api_key,
            base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )
        
        # Initialize MemCell extractor
        self._memcell_extractor = ConvMemCellExtractor(
            llm_provider=self._llm_provider,
            hard_token_limit=self.config.boundary.hard_token_limit,
            hard_message_limit=self.config.boundary.hard_message_limit,
        )
        
        # Initialize Episode extractor
        self._episode_extractor = EpisodeMemoryExtractor(
            llm_provider=self._llm_provider,
            embedding_provider=self.config.embedding.provider,
            embedding_model=self.config.embedding.model,
            embedding_api_key=self.config.embedding.api_key,
            embedding_base_url=self.config.embedding.base_url,
            embedding_dims=self.config.embedding.embedding_dims,
        )
        
        # Initialize Event Log extractor (if enabled)
        self._event_log_extractor: Optional[EventLogExtractor] = None
        if self.config.extraction.enable_event_log:
            self._event_log_extractor = EventLogExtractor(
                llm_provider=self._llm_provider,
                embedding_provider=self.config.embedding.provider,
                embedding_model=self.config.embedding.model,
                embedding_api_key=self.config.embedding.api_key,
                embedding_base_url=self.config.embedding.base_url,
                embedding_dims=self.config.embedding.embedding_dims,
            )
        
        # Initialize Foresight extractor (if enabled)
        self._foresight_extractor: Optional[ForesightExtractor] = None
        if self.config.extraction.enable_foresight:
            self._foresight_extractor = ForesightExtractor(
                llm_provider=self._llm_provider,
                embedding_provider=self.config.embedding.provider,
                embedding_model=self.config.embedding.model,
                embedding_api_key=self.config.embedding.api_key,
                embedding_base_url=self.config.embedding.base_url,
                embedding_dims=self.config.embedding.embedding_dims,
            )
        
        # Initialize Cluster manager (if enabled)
        self._cluster_manager: Optional[ClusterManager] = None
        self._cluster_state: Optional[ClusterState] = None
        if self.config.clustering.enabled:
            cluster_cfg = ClusterManagerConfig(
                similarity_threshold=self.config.clustering.similarity_threshold,
                max_time_gap_days=self.config.clustering.max_time_gap_days,
            )
            self._cluster_manager = ClusterManager(
                config=cluster_cfg,
                embedding_provider=self.config.embedding.provider,
                embedding_model=self.config.embedding.model,
                embedding_api_key=self.config.embedding.api_key,
                embedding_base_url=self.config.embedding.base_url,
                embedding_dims=self.config.embedding.embedding_dims,
            )
            self._cluster_state = ClusterState()
        
        # Initialize Profile manager (if enabled)
        self._profile_manager: Optional[ProfileManager] = None
        if self.config.profile.enabled:
            profile_cfg = ProfileManagerConfig(
                scenario=self.config.profile.scenario,
                min_confidence=self.config.profile.min_confidence,
                batch_size=self.config.profile.batch_size,
            )
            self._profile_manager = ProfileManager(
                llm_provider=self._llm_provider,
                config=profile_cfg,
                group_id=self.config.group_id,
                group_name=self.config.group_id,
            )
        
        # Initialize index manager
        self._index_manager = InMemoryIndexManager(embedding_config=self.config.embedding)
        
        # Initialize retriever
        self._retriever = OnlineRetriever(
            index_manager=self._index_manager,
            config=self.config.retrieval,
            llm_provider=self._llm_provider,
        )
    
    @property
    def llm_model(self) -> LLMProvider:
        """Get the LLM model."""
        return self._llm_provider 
    
    async def add_message_async(
        self,
        message: Dict[str, str],
        timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[MemCell]:
        """
        Add a single message and process it incrementally.
        
        This method:
        1. Adds the message to the buffer
        2. Checks for boundary detection
        3. If boundary detected, extracts MemCell and updates indexes
        4. Updates profiles for user roles
        
        Parameters
        ----------
        message : Dict[str, str]
            Message dictionary with 'role', 'content', and optionally 'name'.
        timestamp : datetime, optional
            Message timestamp in ISO 8601 format. Uses current time if not provided.
        **kwargs : Any
            Additional metadata.
        
        Returns
        -------
        Optional[MemCell]
            The extracted MemCell if a boundary was detected, None otherwise.
        """
        # Prepare timestamp
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        
        # Prepare message dict
        role = message.get("role", "user")
        speaker_id = message.get("speaker_id") or message.get("name") or role
        speaker_name = message.get("name") or message.get("speaker_name") or speaker_id
        
        msg_dict = {
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "content": message.get("content", ""),
            "timestamp": timestamp.isoformat(),
            "role": role,
            **kwargs,
        }
        
        self._stats["total_messages"] += 1
        
        # Detect boundary and extract MemCell if needed
        memcell = await self._process_message(msg_dict)
        
        return memcell
    
    def add_message(
        self,
        message: Dict[str, str],
        timestamp: Optional[datetime] = None,
        **kwargs,
    ) -> Optional[MemCell]:
        """Synchronous wrapper for add_message_async."""
        return asyncio.run(
            self.add_message_async(message, timestamp, **kwargs)
        )
    
    async def _process_message(self, msg_dict: Dict[str, Any]) -> Optional[MemCell]:
        """Process a single message: detect boundary and extract MemCell if needed."""
        # Get the raw message and make a link between it and the message buffer. 
        runtime_raw_message = None 
        tracing_ctx = current_context() 
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_raw_message = tracing_ctx.get_variable(name="raw_input")

        # The operation made on the message buffer should be an in-place mutation.
        # Note that the memory system maintains only a single message buffer,
        # so our identity strategy directly names this variable "message-buffer".
        with comment_mutation(
            target=self._message_buffer, 
            inputs=[runtime_raw_message],  
            category="message_buffer",
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            comment=(
                "A message buffer that stores messages " 
                "not yet processed by the memory system. " 
                "Its data structure is a list."
            ), 
            id_strategy=lambda _: "message-buffer",
            mutation_name="memory_system.message_buffer_update",
            mutation_category="update", 
            mutation_comment=(
                "Add a new input message into the current message buffer."
            ), 
        ) as mutation_scope:
            # Add to buffer
            self._message_buffer.append(msg_dict)
        
        # We can access the new version of the message buffer after the mutation scope exits.
        # We register it in the tracing context so we can reuse it in the boundary detection stage.
        if tracing_ctx is not None and is_tracing_enabled():
            tracing_ctx.register_variable(
                "message_buffer", 
                mutation_scope.result,
                overwrite=True,
            )
        
        # Need at least 2 messages for boundary detection
        if len(self._message_buffer) < 2:
            return None
        
        # Prepare RawData list
        history_raw_data = [
            RawData(content=m, data_id=str(uuid.uuid4()))
            for m in self._message_buffer[:-1]
        ]
        new_raw_data = [
            RawData(content=self._message_buffer[-1], data_id=str(uuid.uuid4()))
        ]
        
        # Get all speaker IDs
        speakers = set()
        for m in self._message_buffer:
            if "speaker_id" in m:
                speakers.add(m["speaker_id"])
        
        # Determine smart mask flag
        smart_mask_flag = (
            self.config.boundary.use_smart_mask 
            and len(self._message_buffer) > self.config.boundary.smart_mask_threshold
        )
        
        # Create extraction request
        request = ConversationMemCellExtractRequest(
            history_raw_data_list=history_raw_data,
            new_raw_data_list=new_raw_data,
            user_id_list=list(speakers),
            smart_mask_flag=smart_mask_flag,
        )
        
        # Detect boundary
        result = await self._memcell_extractor.extract_memcell(request)
        memcell_result = result[0]
        
        if memcell_result is None:
            # No boundary detected, continue accumulating
            return None
        
        # Boundary detected!
        self._stats["boundaries_detected"] += 1


        # We can get the runtime variable of the temporary memory cell.
        runtime_temporary_memcell, runtime_memcell_result_snapshot = None, None 
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_temporary_memcell = tracing_ctx.get_variable("memcell")
        
        if memcell_result.event_id is None:
            memcell_result.event_id = generate_object_id_str()

            # The event identifier represents the unique identifier of the memory cell.
            memcell_result_snapshot = {
                "original_data": memcell_result.original_data,
                "timestamp": memcell_result.timestamp.isoformat(),
                "event_id": memcell_result.event_id,
                "episode": None,
                "subject": None,
                "summary": None,
                "foresights": None,
                "event_log": None,
            }

            # Thanks to runtime variables being able to store decoding functions,
            # we can easily obtain the original snapshot rather than its string representation.
            if runtime_temporary_memcell is not None and "summary" in runtime_temporary_memcell.value:
                memcell_result_snapshot["summary"] = memcell_result.summary 
            
            runtime_memcell_result_snapshot = comment_variable(
                memcell_result_snapshot,
                variable_name="memcell_result",
                to_runtime=True,
                category="memory_entry",
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                id_strategy="evermemos-dict",
                comment=(
                    "An initial memory-unit created from the temporary memory unit "
                    "candidate after the memory system assigns it a unique event identifier."
                ),
            )
            
            comment_op(
                inputs=[runtime_temporary_memcell],
                outputs=[runtime_memcell_result_snapshot],
                op_name="memory_system.finalize_memory_unit",
                category="initialization",
                comment=(
                    "The memory system finalizes the temporary memory unit candidate into "
                    "a concrete memory unit by assigning a unique event identifier. "
                    "It is an initial memory unit that will be used in the "
                    "subsequent extraction and indexing stages."
                ),
            )

        if tracing_ctx is not None and is_tracing_enabled():
            tracing_ctx.remove_variable("memcell")
        
        # Extract Episode
        episode_request = MemoryExtractRequest(
            memcell=memcell_result,
            user_id=None,  # Group episode
            participants=list(speakers),
            group_id=self.config.group_id,
        )


        # Initialize the extraction operation context.
        foresight_enabled = self._foresight_extractor is not None
        event_log_enabled = self._event_log_extractor is not None
        extraction_comment = (
            "Run the memory-extraction stage on the initial memory-unit. "
            "This stage first extracts an episode-level narrative memory."
        )
        if foresight_enabled:
            extraction_comment += (
                " The foresight extraction is enabled, so the stage also derives "
                "foresight memories from the extracted episode memory."
            )
        if event_log_enabled:
            extraction_comment += (
                " The event-log extraction is enabled, so the stage also derives a "
                "retrieval-oriented event log from the same initial memory-unit."
            )
        with comment_op_scope(
            op_name="memory_system.extract_memory",
            category="extraction",
            comment=extraction_comment,
            metadata={
                "llm_model": self._llm_provider.provider.model,
                "foresight_enabled": foresight_enabled,
                "event_log_enabled": event_log_enabled,
            },
        ):
            episode_memory = await self._episode_extractor.extract_memory(episode_request)
        
            if episode_memory and episode_memory.episode:
                # Get the current snapshot of the initial memory-unit.
                runtime_episode_memory = None 
                memcell_result_snapshot = {} 
                if tracing_ctx is not None and is_tracing_enabled():
                    runtime_episode_memory = tracing_ctx.get_variable("episode_memory")
                    memcell_result_snapshot = runtime_memcell_result_snapshot.value
                    
                with comment_mutation(
                    target=memcell_result_snapshot,
                    inputs=[runtime_episode_memory],
                    comment=(
                        "This is a refined memory-unit after "
                        "the memory system enriches it with the extracted episode-level "
                        "narrative memory." 
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    category="memory_entry",
                    decoding_fn=json.loads,
                    id_strategy="evermemos-dict",
                    mutation_comment=(
                        "The memory system refines the initial memory-unit by writing "
                        "the extracted episode-level narrative memory back into it as "
                        "the episode text, the main subject, and a short derived summary."
                    ), 
                    reuse_op=True,
                ) as memcell_mutation_scope:
                    memcell_result.episode = episode_memory.episode
                    memcell_result.subject = episode_memory.subject if episode_memory.subject else ""
                    memcell_result.summary = episode_memory.episode[:200] + "..."

                    memcell_result_snapshot["episode"] = memcell_result.episode
                    memcell_result_snapshot["subject"] = memcell_result.subject 
                    memcell_result_snapshot["summary"] = memcell_result.summary

                runtime_memcell_result_snapshot = memcell_mutation_scope.result
                # Overwrite the runtime variable of the memory-unit.
                # A new version of memory-unit is registered in the tracing context.
                if tracing_ctx is not None and is_tracing_enabled():
                    tracing_ctx.register_variable(
                        "memcell_result", 
                        runtime_memcell_result_snapshot,
                        overwrite=True,
                    )

                if runtime_episode_memory is not None:
                    tracing_ctx.remove_variable("episode_memory")

                # 2. Extract Foresight
                if self._foresight_extractor:
                    # Note: The following code snippet is not included in the official implementation.
                    input_text = ""
                    for data in memcell_result.original_data:
                        speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                        content = data['content']
                        msg_ts = data.get('timestamp')
                        ts_str = from_iso_format(msg_ts)
                        input_text += f"[{ts_str}] {speaker}: {content}\n"

                    foresight_memories = (
                        await self._foresight_extractor.generate_foresights_for_conversation(
                            input_text, 
                            episode_memory.timestamp,
                            episode_memory.user_id, 
                        )
                    )
                    if foresight_memories:
                        memcell_result.foresights = foresight_memories

                        # Simulate an in-place mutation of the memory-unit.
                        memcell_result_snapshot = {} 
                        runtime_foresights = None 
                        if tracing_ctx is not None and is_tracing_enabled():
                            runtime_foresights = tracing_ctx.get_variable("foresights")
                            memcell_result_snapshot = runtime_memcell_result_snapshot.value

                        with comment_mutation(
                            target=memcell_result_snapshot,
                            inputs=[runtime_foresights],
                            comment=(
                                "This is a further refined memory-unit after the memory "
                                "system enriches it with the structured foresight memories "
                                "derived from the memory unit's original data."
                            ),
                            encoding_fn=partial(
                                json.dumps,
                                ensure_ascii=False,
                                indent=4,
                                sort_keys=True,
                            ),
                            category="memory_entry",
                            decoding_fn=json.loads,
                            id_strategy="evermemos-dict",
                            mutation_comment=(
                                "The memory system further refines the current memory-unit "
                                "by writing the structured foresight memories derived from "
                                "the memory unit's original data back into the memory-unit."
                            ), 
                            reuse_op=True,
                        ) as memcell_mutation_scope: 
                            memcell_result_snapshot["foresights"] = runtime_foresights.value

                        runtime_memcell_result_snapshot = memcell_mutation_scope.result
                        if tracing_ctx is not None and is_tracing_enabled():
                            tracing_ctx.register_variable(
                                "memcell_result", 
                                runtime_memcell_result_snapshot,
                                overwrite=True,
                            )

                        if runtime_foresights is not None:
                            tracing_ctx.remove_variable("foresights")

            else:
                # Episode extraction failed - raise exception, don't hide errors
                raise ValueError(
                    f"❌ Episode extraction failed! conv_id={self.config.group_id}, memcell_id={memcell_result.event_id}"
                )
            
            # Extract Event Log (if enabled)
            if self._event_log_extractor and episode_memory and episode_memory.episode:
                event_log = await self._event_log_extractor.extract_event_log(
                    memcell=memcell_result,
                    timestamp=memcell_result.timestamp,
                )
                if event_log:
                    memcell_result.event_log = event_log

                    # Simulate an in-place mutation of the memory-unit.
                    memcell_result_snapshot = {} 
                    runtime_event_log = None 
                    if tracing_ctx is not None and is_tracing_enabled():
                        runtime_event_log = tracing_ctx.get_variable("event_log")
                        memcell_result_snapshot = runtime_memcell_result_snapshot.value

                    with comment_mutation(
                        target=memcell_result_snapshot,
                        inputs=[runtime_event_log],
                        comment=(
                            "This is a further refined memory-unit after the memory "
                            "system enriches it with the structured retrieval-oriented "
                            "event-log extracted from the memory unit's original data."
                        ),
                        encoding_fn=partial(
                            json.dumps,
                            ensure_ascii=False,
                            indent=4,
                            sort_keys=True,
                        ),
                        category="memory_entry",
                        decoding_fn=json.loads,
                        id_strategy="evermemos-dict",
                        mutation_comment=(
                            "The memory system further refines the current memory-unit "
                            "by writing the structured retrieval-oriented event-log "
                            "back into the memory-unit."
                        ), 
                        reuse_op=True,
                    ) as memcell_mutation_scope: 
                        if runtime_event_log is not None:
                            memcell_result_snapshot["event_log"] = runtime_event_log.value

                    runtime_memcell_result_snapshot = memcell_mutation_scope.result
                    if tracing_ctx is not None and is_tracing_enabled():
                        tracing_ctx.register_variable(
                            "memcell_result", 
                            runtime_memcell_result_snapshot,
                            overwrite=True,
                        )

                    if runtime_event_log is not None:
                        tracing_ctx.remove_variable("event_log")
        
            # Clustering (if enabled)
            # For this version, we don't need to trace the clustering operation 
            # as it doesn't affect the memory system's performance.
            if self._cluster_manager and self._cluster_state:
                # Incremental clustering: Update the current cluster state  
                _, self._cluster_state = await self._cluster_manager.cluster_memcell(
                    memcell_result.to_dict(),
                    self._cluster_state,
                )
            
            # Add to index
            await self._index_manager.add_memcell(memcell_result)

            if tracing_ctx is not None and is_tracing_enabled():
                memcell_result_snapshot = runtime_memcell_result_snapshot.value
                with comment_mutation(
                    target=memcell_result_snapshot,
                    comment=(
                        "This is the final memory-unit prepared for downstream memory "
                        "use and storage."
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    category="memory_entry",
                    decoding_fn=json.loads,
                    id_strategy="evermemos-dict",
                    mutation_comment=(
                        "After memory extraction is complete, the memory system "
                        "conceptually removes the raw original data from the current "
                        "memory-unit, forms the final memory unit for downstream use, "
                        "and places that final memory unit into the memory store."
                    ),
                    reuse_op=True,
                ):
                    memcell_result_snapshot.pop("original_data")
        
            # Store MemCell
            self._memcells.append(memcell_result)
            self._stats["memcells_extracted"] += 1
        
        # Update profiles for user roles
        # For this version , we don't need to trace the profile update operation 
        # as it doesn't affect the memory system's performance.
        await self._update_profiles_for_users(memcell_result)
        

        mutation_comment = (
            "After the current memory unit is extracted, the memory system resets "
            "the message buffer by keeping the last two messages because smart mask "
            "is enabled."
            if smart_mask_flag
            else
            "After the current memory unit is extracted, the memory system resets "
            "the message buffer by keeping only the newest message."
        )
        # The original implementation updates the message buffer by reassigning
        # `self._message_buffer` instead of mutating that list strictly in place.
        # Therefore, this mutation is traced on the parent container `self`,
        # while the encoder projects only the embedded message-buffer field.
        with comment_mutation(
            target=self,
            id_strategy=lambda _: "message-buffer",
            encoding_fn=lambda memory_manager: json.dumps(
                memory_manager._message_buffer,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            mutation_name="memory_system.message_buffer_update",
            mutation_category="update",
            mutation_comment=mutation_comment,
            mutation_metadata={
                "smart_mask_flag": smart_mask_flag,
            },
        ):
            # Reset buffer with smart mask
            # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage1_memcells_extraction.py#L219
            if smart_mask_flag:
                self._message_buffer = [self._message_buffer[-2], self._message_buffer[-1]]
            else:
                self._message_buffer = [self._message_buffer[-1]]
            
        # Remove the tracing variables. 
        if tracing_ctx is not None and is_tracing_enabled():
            tracing_ctx.remove_variable("message_buffer")
            tracing_ctx.remove_variable("memcell_result")
        
        return memcell_result
    
    async def _update_profiles_for_users(self, memcell: MemCell) -> None:
        """Update profiles for users in the MemCell."""
        if not self._profile_manager:
            return
        
        # Only speakers who are users need to be updated
        user_ids_to_update = []
        
        for msg in memcell.original_data:
            role = msg.get("role", "user")
            if role == "user":
                speaker_id = msg.get("speaker_id")
                if speaker_id and speaker_id not in user_ids_to_update:
                    user_ids_to_update.append(speaker_id)
        
        if not user_ids_to_update:
            return
        
        # Get old profiles from existing profiles
        old_profiles = [
            self._profiles[uid] 
            for uid in user_ids_to_update 
            if uid in self._profiles
        ]
        
        # Extract new profiles
        new_profiles = await self._profile_manager.extract_profiles(
            memcells=[memcell],
            old_profiles=old_profiles if old_profiles else None,
            user_id_list=user_ids_to_update,
        )
        
        # Update stored profiles 
        for profile in new_profiles:
            if isinstance(profile, ProfileMemory):
                user_id = profile.user_id
                if user_id:
                    self._profiles[user_id] = profile
                    self._stats["profiles_updated"] += 1
    
    async def retrieve_async(
        self,
        query: str,
        k: int = 10,
        **kwargs,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant memories for a query.
        
        Parameters
        ----------
        query : str
            The search query.
        k : int, default 10
            Maximum number of results to return.
        **kwargs : Any
            Additional retrieval parameters.
        
        Returns
        -------
        List[Tuple[Dict[str, Any], float]]
            List of (document, score) tuples from the retriever.
        """
        return await self._retriever.retrieve(query, k=k, **kwargs)
    
    def retrieve(self, query: str, k: int = 10, **kwargs) -> List[Tuple[Dict[str, Any], float]]:
        """Synchronous wrapper for retrieve_async."""
        async def _run():
            try:
                return await self.retrieve_async(query, k, **kwargs)
            finally:
                # Reset the async HTTP client to avoid "Event loop is closed" errors
                # when the next `asyncio.run()` creates a new event loop.
                await self._index_manager._vectorize_service.close()
                if self._retriever._reranker is not None:
                    await self._retriever._reranker.close()
        return asyncio.run(_run())
    
    async def flush_async(self) -> Optional[MemCell]:
        """
        Flush remaining messages in the buffer as a final MemCell.
        
        Call this when ending a conversation to ensure all messages are processed.
        """
        if len(self._message_buffer) < 1:
            return None
        
        # Note: This message buffer may not have been incorporated into the execution graph.
        # For example, if the conversation is very short, EverMemOS may terminate before
        # it starts segmenting the dialogue, leaving the buffer unprocessed.
        runtime_message_buffer = comment_variable(
            self._message_buffer,
            to_runtime=True,
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            id_strategy=lambda _: "message-buffer",
            category="message_buffer",
            comment=(
                "A message buffer that stores messages " 
                "not yet processed by the memory system. " 
                "Its data structure is a list."
            ), 
        )
        
        # Get speakers
        speakers = set()
        for m in self._message_buffer:
            if "speaker_id" in m:
                speakers.add(m["speaker_id"])
        
        # Get timestamp
        timestamp = datetime.now()
        if self._message_buffer:
            ts_str = self._message_buffer[-1].get("timestamp")
            if ts_str:
                timestamp = from_iso_format(ts_str)
        
        # Create final MemCell using existing type
        memcell = MemCell(
            user_id_list=list(speakers),
            original_data=self._message_buffer.copy(),
            timestamp=timestamp,
            summary="Final conversation segment",
            event_id=generate_object_id_str(),
            type=RawDataType.CONVERSATION,
        )

        runtime_memcell = comment_variable(
            {
                "original_data": memcell.original_data,
                "timestamp": memcell.timestamp.isoformat(),
                "summary": memcell.summary,
                "event_id": memcell.event_id,
                "episode": None,
                "subject": None,
                "foresights": None,
                "event_log": None,
            },
            variable_name="memcell_result",
            to_runtime=True,
            category="memory_entry",
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            id_strategy="evermemos-dict",
            comment=(
                "An initial memory-unit created from the current message buffer directly. "
                "It contains a timestamp (`timestamp`) and the raw data (`original_data`) derived "
                "from the current message buffer. "
                "The timestamp corresponds to the last message's timestamp in the original data."
            ),
        )
        comment_op(
            inputs=[runtime_message_buffer],
            outputs=[runtime_memcell],
            op_name="memory_system.finalize_memory_unit",
            category="initialization",
            comment=(
                "The memory system finalizes the remaining buffered messages into "
                "a concrete memory unit because the conversation is ending. "
                "It is an initial memory unit that will be used in the "
                "subsequent extraction and indexing stages."
            ),
        )
        
        # Extract Episode
        episode_request = MemoryExtractRequest(
            memcell=memcell,
            user_id=None,
            participants=list(speakers),
            group_id=self.config.group_id,
        )

        # Get the tracing context.
        tracing_ctx = current_context()
        
        # Initialize the extraction operation context.
        foresight_enabled = self._foresight_extractor is not None
        event_log_enabled = self._event_log_extractor is not None
        extraction_comment = (
            "Run the memory-extraction stage on the initial memory-unit. "
            "This stage first extracts an episode-level narrative memory."
        )
        if foresight_enabled:
            extraction_comment += (
                " The foresight extraction is enabled, so the stage also derives "
                "foresight memories from the extracted episode memory."
            )
        if event_log_enabled:
            extraction_comment += (
                " The event-log extraction is enabled, so the stage also derives a "
                "retrieval-oriented event log from the same initial memory-unit."
            )
        with comment_op_scope(
            op_name="memory_system.extract_memory",
            category="extraction",
            comment=extraction_comment,
            metadata={
                "llm_model": self._llm_provider.provider.model,
                "foresight_enabled": foresight_enabled,
                "event_log_enabled": event_log_enabled,
            },
        ):
            episode_memory = await self._episode_extractor.extract_memory(episode_request)

            if episode_memory and episode_memory.episode:
                # Get the current snapshot of the initial memory-unit.
                runtime_episode_memory = None 
                memcell_snapshot = {} 
                if tracing_ctx is not None and is_tracing_enabled():
                    runtime_episode_memory = tracing_ctx.get_variable("episode_memory")
                    memcell_snapshot = runtime_memcell.value
            
                with comment_mutation(
                    target=memcell_snapshot,
                    inputs=[runtime_episode_memory],
                    comment=(
                        "This is a refined memory-unit after "
                        "the memory system enriches it with the extracted episode-level "
                        "narrative memory." 
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    category="memory_entry",
                    decoding_fn=json.loads,
                    id_strategy="evermemos-dict",
                    mutation_comment=(
                        "The memory system refines the initial memory-unit by writing "
                        "the extracted episode-level narrative memory back into it as "
                        "the episode text, the main subject, and a short derived summary."
                    ), 
                    reuse_op=True,
                ) as memcell_mutation_scope:
                    memcell.episode = episode_memory.episode
                    memcell.subject = episode_memory.subject if episode_memory.subject else ""
                    memcell.summary = episode_memory.episode[:200] + "..."

                    memcell_snapshot["episode"] = memcell.episode
                    memcell_snapshot["subject"] = memcell.subject 
                    memcell_snapshot["summary"] = memcell.summary

                runtime_memcell_snapshot = memcell_mutation_scope.result
                # Overwrite the runtime variable of the memory-unit.
                # A new version of memory-unit is registered in the tracing context.
                if tracing_ctx is not None and is_tracing_enabled():
                    tracing_ctx.register_variable(
                        "memcell_result", 
                        runtime_memcell_snapshot,
                        overwrite=True,
                    )

                if runtime_episode_memory is not None:
                    tracing_ctx.remove_variable("episode_memory")

                # 2. Extract Foresight (optional)
                if self._foresight_extractor:
                    input_text = ""
                    for data in memcell.original_data:
                        speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                        content = data['content']
                        msg_ts = data.get('timestamp')
                        ts_str = from_iso_format(msg_ts)
                        input_text += f"[{ts_str}] {speaker}: {content}\n"

                    foresight_memories = (
                        await self._foresight_extractor.generate_foresights_for_conversation(
                            input_text, 
                            episode_memory.timestamp,
                            episode_memory.user_id, 
                        )
                    )
                    if foresight_memories:
                        memcell.foresights = foresight_memories

                        # Simulate an in-place mutation of the memory-unit.
                        memcell_snapshot = {} 
                        runtime_foresights = None 
                        if tracing_ctx is not None and is_tracing_enabled():
                            runtime_foresights = tracing_ctx.get_variable("foresights")
                            memcell_snapshot = runtime_memcell_snapshot.value

                        with comment_mutation(
                            target=memcell_snapshot,
                            inputs=[runtime_foresights],
                            comment=(
                                "This is a further refined memory-unit after the memory "
                                "system enriches it with the structured foresight memories "
                                "derived from the memory unit's original data."
                            ),
                            encoding_fn=partial(
                                json.dumps,
                                ensure_ascii=False,
                                indent=4,
                                sort_keys=True,
                            ),
                            category="memory_entry",
                            decoding_fn=json.loads,
                            id_strategy="evermemos-dict",
                            mutation_comment=(
                                "The memory system further refines the current memory-unit "
                                "by writing the structured foresight memories derived from "
                                "the memory unit's original data back into the memory-unit."
                            ), 
                            reuse_op=True,
                        ) as memcell_mutation_scope: 
                            memcell_snapshot["foresights"] = runtime_foresights.value

                        runtime_memcell_snapshot = memcell_mutation_scope.result
                        if tracing_ctx is not None and is_tracing_enabled():
                            tracing_ctx.register_variable(
                                "memcell_result", 
                                runtime_memcell_snapshot,
                                overwrite=True,
                            )

                        if runtime_foresights is not None:
                            tracing_ctx.remove_variable("foresights")
            else:
                # Episode extraction failed - raise exception, don't hide errors
                raise ValueError(
                    f"❌ Episode extraction failed! conv_id={self.config.group_id}, memcell_id={memcell.event_id}"
                )

            # Extract Event Log (if enabled)
            if self._event_log_extractor and episode_memory and episode_memory.episode:
                event_log = await self._event_log_extractor.extract_event_log(
                    memcell=memcell,
                    timestamp=memcell.timestamp,
                )
                if event_log:
                    memcell.event_log = event_log

                    # Simulate an in-place mutation of the memory-unit.
                    memcell_snapshot = {} 
                    runtime_event_log = None 
                    if tracing_ctx is not None and is_tracing_enabled():
                        runtime_event_log = tracing_ctx.get_variable("event_log")
                        memcell_snapshot = runtime_memcell_snapshot.value

                    with comment_mutation(
                        target=memcell_snapshot,
                        inputs=[runtime_event_log],
                        comment=(
                            "This is a further refined memory-unit after the memory "
                            "system enriches it with the structured retrieval-oriented "
                            "event-log extracted from the memory unit's original data."
                        ),
                        encoding_fn=partial(
                            json.dumps,
                            ensure_ascii=False,
                            indent=4,
                            sort_keys=True,
                        ),
                        category="memory_entry",
                        decoding_fn=json.loads,
                        id_strategy="evermemos-dict",
                        mutation_comment=(
                            "The memory system further refines the current memory-unit "
                            "by writing the structured retrieval-oriented event-log "
                            "back into the memory-unit."
                        ), 
                        reuse_op=True,
                    ) as memcell_mutation_scope: 
                        if runtime_event_log is not None:
                            memcell_snapshot["event_log"] = runtime_event_log.value

                    runtime_memcell_snapshot = memcell_mutation_scope.result
                    if tracing_ctx is not None and is_tracing_enabled():
                        tracing_ctx.register_variable(
                            "memcell_result", 
                            runtime_memcell_snapshot,
                            overwrite=True,
                        )

                    if runtime_event_log is not None:
                        tracing_ctx.remove_variable("event_log")
            
            # Clustering (if enabled)
            if self._cluster_manager and self._cluster_state:
                # Incremental clustering: Update the current cluster state  
                _, self._cluster_state = await self._cluster_manager.cluster_memcell(
                    memcell.to_dict(),
                    self._cluster_state,
                )
            
            # Add to index
            await self._index_manager.add_memcell(memcell)

            if tracing_ctx is not None and is_tracing_enabled():
                memcell_snapshot = runtime_memcell_snapshot.value
                with comment_mutation(
                    target=memcell_snapshot,
                    comment=(
                        "This is the final memory-unit prepared for downstream memory "
                        "use and storage."
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    category="memory_entry",
                    decoding_fn=json.loads,
                    id_strategy="evermemos-dict",
                    mutation_comment=(
                        "After memory extraction is complete, the memory system "
                        "conceptually removes the raw original data from the current "
                        "memory-unit, forms the final memory unit for downstream use, "
                        "and places that final memory unit into the memory store."
                    ),
                    reuse_op=True,
                ):
                    memcell_snapshot.pop("original_data")
            
            # Store MemCell
            self._memcells.append(memcell)
            self._stats["memcells_extracted"] += 1
        
        # Update profiles for user roles
        await self._update_profiles_for_users(memcell)
        
        # Clear buffer
        # The original implementation clears the message buffer by reassigning
        # `self._message_buffer` instead of mutating that list strictly in place.
        # Therefore, this mutation is traced on the parent container `self`,
        # while the encoder projects only the embedded message-buffer field.
        with comment_mutation(
            target=self,
            comment=(
                "A message buffer that stores messages " 
                "not yet processed by the memory system. " 
                "Its data structure is a list."
            ),
            id_strategy=lambda _: "message-buffer",
            encoding_fn=lambda memory_manager: json.dumps(
                memory_manager._message_buffer,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            mutation_name="memory_system.message_buffer_update",
            mutation_category="update",
            mutation_comment=(
                "After the remaining messages are flushed into a final memory unit, "
                "the memory system clears the message buffer."
            ),
            mutation_metadata={
                "flush": True,
            },
        ):
            self._message_buffer = []

        # Remove the tracing variables. 
        if tracing_ctx is not None and is_tracing_enabled():
            tracing_ctx.remove_variable("memcell_result")
        
        return memcell
    
    def flush(self) -> Optional[MemCell]:
        """Synchronous wrapper for flush_async."""
        return asyncio.run(self.flush_async())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._stats,
            "buffer_size": len(self._message_buffer),
            "total_memcells": len(self._memcells),
            "total_profiles": len(self._profiles),
        }
    
    def get_memcells(self) -> List[MemCell]:
        """Get all extracted MemCells."""
        return self._memcells.copy()
    
    def get_profiles(self) -> Dict[str, ProfileMemory]:
        """Get all user profiles as ProfileMemory objects."""
        return self._profiles.copy()
    
    def get_profiles_as_dicts(self) -> Dict[str, Dict[str, Any]]:
        """Get all user profiles converted to dictionaries."""
        return {
            uid: profile.to_dict() 
            for uid, profile in self._profiles.items()
        }
    
    async def save_async(self, save_dir: str) -> None:
        """Save the memory state to disk as pkl files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Directly store objects (pkl handles serialization)
        state = {
            "memcells": self._memcells,
            "profiles": self._profiles,
            "cluster_state": self._cluster_state,
            "message_buffer": self._message_buffer,
            "stats": self._stats,
        }
        
        with open(save_path / "memory_state.pkl", "wb") as f:
            pickle.dump(state, f)
        
        # Save index separately (contains numpy arrays)
        await self._index_manager.save(save_path / "index.pkl")
    
    def save(self, save_dir: str) -> None:
        """Synchronous wrapper for save_async."""
        asyncio.run(self.save_async(save_dir))
    
    async def load_async(self, save_dir: str) -> bool:
        """Load the memory state from disk."""
        save_path = Path(save_dir)
        state_path = save_path / "memory_state.pkl"
        
        if not state_path.exists():
            return False
        
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        
        self._memcells = state["memcells"]
        self._profiles = state["profiles"]
        self._cluster_state = state["cluster_state"]
        self._message_buffer = state["message_buffer"]
        self._stats = state["stats"]
        print("The state of EverMemOS layer is loaded successfully.")
        
        # Load index
        index_path = save_path / "index.pkl"
        if index_path.exists():
            res = await self._index_manager.load(index_path)
            if res:
                print("The index of EverMemOS layer is loaded successfully.")
        
        return True
    
    def load(self, save_dir: str) -> bool:
        """Synchronous wrapper for load_async."""
        return asyncio.run(self.load_async(save_dir))
