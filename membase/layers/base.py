from __future__ import annotations
from abc import ABC, abstractmethod
from ..utils import PatchSpec
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
from typing import Any 


# Note: This base interface will evolve as additional memory systems are integrated.
# As new methods and capabilities are standardized across implementations, this class
# may be updated to ensure a consistent API across different memory backends.
class MemBaseLayer(ABC):
    """
    Abstract base class for memory layers that defines a unified interface for various memory 
    algorithms. This class follows the template method pattern and provides common methods 
    that should be implemented by concrete memory layer classes.
    
    The interface is designed to be compatible with popular memory frameworks like mem0, 
    A-MEM, LangMem, and other memory systems, providing a consistent API for memory 
    operations across different implementations.
    """

    @abstractmethod
    def add_message(self, message: Message, **kwargs: Any) -> None:
        """Add a single message to the memory layer.
        
        Each concrete implementation defines how the message is ingested. Depending on the
        memory system, the message may be directly stored, or transformed into a structured memory unit
        via model-based extraction.

        Args:
            message (`Message`): 
                A message.
            **kwargs (`Any`):
                Additional keyword arguments required by specific implementations.
                These arguments are independent of the given message.
        """
        ... 

    @abstractmethod
    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        """Add a list of messages to the memory layer.

        The default behavior of many implementations is to iterate over ``messages`` and
        delegate to method `add_message` one by one, making the two methods functionally
        equivalent. However, some memory layers treat a batch of messages as a logical
        unit (e.g., a conversation session) and apply specialized extraction logic that
        differs from per-message ingestion. For example, some memory layers perform session-level
        memory summarization or boundary detection that is only meaningful over a sequence of
        messages. Subclasses should document any such batch-level semantics.
        
        Args:
            messages (`list[Message]`): 
                A list of messages.
            **kwargs (`Any`):
                Additional keyword arguments forwarded to the ingestion logic.
                These arguments are independent of the given messages.
        """
        ... 

    @abstractmethod
    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        """
        Retrieve memory entries relevant to the given query.
        
        Subclasses should override this method to implement their own retrieval logic,
        support additional retrieval parameters, and define how each memory entry is 
        formatted for downstream tasks.

        Args:
            query (`str`): 
                The natural language query to find relevant memories.
            k (`int`, defaults to `10`): 
                Maximum number of memory entries to return. Some implementations may
                return fewer if the memory store contains less entries.
            **kwargs (`Any`):
                Additional keyword arguments for retrieval customization, such as
                filters, search parameters, or implementation-specific settings.

        Returns:
            `list[MemoryEntry]`: 
                A list of retrieved memory entries, ranked by relevance or recency
                depending on the implementation.
        """
        ... 

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory entry by its unique identifier. 

        Subclasses should override this method to provide their specific delete logic.
        
        Args:
            memory_id (`str`): 
                The unique identifier of the memory entry to delete.

        Returns:
            `bool`: 
                Whether the memory is successfully deleted.
        """
        ... 

    @abstractmethod
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update an existing memory entry.

        Subclasses should override this method to provide their specific update logic.
        
        Args:
            memory_id (`str`): 
                The unique identifier of the memory entry to update.
            **kwargs (`Any`):
                Keyword arguments containing the fields to update. 

        Returns:
            `bool`: 
                Whether the memory is successfully updated.
        """
        ...

    @abstractmethod
    def save_memory(self) -> None:
        """Save the memory state to the implementation-configured storage location.
        
        Implementations typically persist both configuration and serialized memory data 
        to the directory specified in the layer's configuration.

        Subclasses should override this method to provide their specific save logic.
        """
        ...

    @abstractmethod
    def load_memory(self, user_id: str | None = None) -> bool:
        """Load the memory state for a specific user from persistent storage.
        
        Implementations read configuration and serialized data from the storage
        directory, reconstruct internal state, and validate that the stored user ID
        matches the requested one.

        Subclasses should override this method to provide their specific load logic.

        Args:
            user_id (`str`, optional): 
                The identifier of the user whose memory state should be loaded. If not
                provided, the implementation uses the user ID from the current config.

        Returns:
            `bool`: 
                Whether the user's memory state is successfully loaded.
        """
        ...

    def flush(self) -> None:
        """
        Force trigger pending online operations and ensure internal state consistency.
        
        Some memory layers perform deferred or batched operations (e.g., buffered indexing,
        lazy graph updates, or accumulated message processing) that are only triggered 
        under certain conditions. After feeding a complete trajectory of messages, call 
        this method to:
        
        1. Force process any pending messages
        2. Ensure all messages have been fully ingested into the memory layer
        3. Guarantee internal state consistency before retrieval or saving
    
        The default implementation is a no-op. Subclasses should override this method
        if they have pending operations that need to be flushed.
        """
        ...

    def get_patch_specs(self) -> list[PatchSpec]:
        """
        Return a list of patch rules for token monitoring.
        
        This method allows each memory layer to define its own monkey patch specifications
        for monitoring LLM API calls during memory construction. By default, it returns 
        an empty list, meaning no patching is applied.
        
        Subclasses can override this method to provide their specific patch specs
        for token usage tracking. The specs define how to intercept and wrap LLM calls
        to monitor input/output tokens and operation types. The operation type
        ``'generation'`` indicates creating new memory entries, while ``'update'``
        indicates modifying existing ones.

        Returns:
            `list[PatchSpec]`: 
                A list containing rules defining the monkey patches to apply.
        """ 
        return []

    def cleanup(self) -> None:
        """Release external resources held by this layer.
        
        The default implementation is a no-op. Subclasses that manage external 
        resources (e.g., database connections) should override this method.
        """
        ...
