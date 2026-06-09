from collections import deque
from functools import partial
from litellm import token_counter as litellm_token_counter
from ..utils import get_tokenizer_for_model


class MessageBufferMixin:
    """
    A mixin class that provides online message buffering functionality for memory layers.
    
    This mixin maintains a sliding window of messages, enabling incremental indexing
    while preserving context from recent messages. It is designed for RAG systems that 
    typically expect batch document indexing but need to work in an online, 
    message-by-message scenario (e.g., memory evaluation benchmarks).
    """
    
    def _init_buffer(
        self,
        num_overlap_msgs: int | float = 0,
        max_tokens: int | None = None,
        model_for_tokenizer: str | None = None,
        deferred: bool = False,
    ) -> None:
        """Initialize the message buffer.
        
        Args:
            num_overlap_msgs (`int | float`, defaults to `0`): 
                The number of previous messages to include when indexing a new message.
                If it is `0`, each message is indexed independently without overlap.
            max_tokens (`int | None`, optional): 
                Maximum total token count allowed in the buffer. When it is set, the buffer 
                trims oldest messages until the total token count is within this limit.
                It requires ``model_for_tokenizer`` to be provided.
            model_for_tokenizer (`str | None`, optional): 
                Model name for token counting. It is required when ``max_tokens`` is set.
            deferred (`bool`, defaults to `False`):
                When it is enabled, messages are accumulated in the buffer and a document is
                only emitted when adding the next message would exceed the maximum token count.
        """
        if max_tokens is not None and model_for_tokenizer is None:
            raise ValueError(
                "`model_for_tokenizer` is required when `max_tokens` is set."
            )
        if deferred and (max_tokens is None or model_for_tokenizer is None):
            raise ValueError(
                "Deferred mode requires both `max_tokens` and `model_for_tokenizer`."
            )

        self._message_buffer = deque()
        self._num_overlap_msgs = num_overlap_msgs
        self._max_tokens = max_tokens
        self._deferred = deferred

        if model_for_tokenizer is not None:
            tokenizer = get_tokenizer_for_model(model_for_tokenizer) 
            self._tokenizer = partial(
                litellm_token_counter, 
                model=model_for_tokenizer,
                custom_tokenizer=tokenizer,
            )
            self._buffer_total_tokens = 0
        else:
            self._tokenizer = None
            self._buffer_total_tokens = None 
    
    def _trim_buffer(self) -> None:
        """Drop oldest messages until both overlap and token constraints are met."""
        while len(self._message_buffer) > self._num_overlap_msgs:
            removed = self._message_buffer.popleft()
            if self._tokenizer is not None:
                self._buffer_total_tokens -= self._tokenizer(text=removed)

        if self._max_tokens is not None:
            while self._buffer_total_tokens > self._max_tokens and self._message_buffer:
                removed = self._message_buffer.popleft()
                self._buffer_total_tokens -= self._tokenizer(text=removed)
    
    def _buffer_and_get_doc(
        self,
        message_content: str | None = None,
        separator: str = "\n",
    ) -> str | None:
        """Optionally add a message to the buffer and return the concatenated document.

        In deferred mode, messages accumulate silently unless adding the new message would 
        exceed the maximum token count. In that case the current buffer is emitted as a 
        document (excluding the new message), and the new message is then appended.
         
        Args:
            message_content (`str | None`, optional): 
                The content of the new message. If it is not provided, only the current
                buffer contents are returned.
            separator (`str`, defaults to ``"\\n"``): 
                The separator used to join messages into a document.
            
        Returns:
            `str | None`: 
                The concatenated document, or `None` in deferred mode when the buffer
                has not yet reached its capacity.
        """
        if message_content is None:
            return separator.join(self._message_buffer)

        if not self._deferred:
            doc = separator.join([*self._message_buffer, message_content])
            self._message_buffer.append(message_content)
            if self._tokenizer is not None:
                self._buffer_total_tokens += self._tokenizer(text=message_content)
            self._trim_buffer()
            return doc

        doc = None
        new_tokens = self._tokenizer(text=message_content)

        if self._buffer_total_tokens + new_tokens > self._max_tokens and self._message_buffer:
            doc = separator.join(self._message_buffer)
            self._trim_buffer()

        self._message_buffer.append(message_content)
        self._buffer_total_tokens += new_tokens
        return doc
    
    def _get_buffer_size(self) -> int:
        """Return the current number of messages in the message buffer.
        
        Returns:
            `int`: 
                The current number of messages in the message buffer.
        """
        return len(self._message_buffer)
    
    def _get_buffer_token_count(self) -> int:
        """Return the total token count of messages currently in the buffer.
        
        Returns:
            `int`: 
                The total token count of messages currently in the buffer.
                It returns `-1` if token counting is not enabled.
        """
        if self._tokenizer is None:
            return -1
        return self._buffer_total_tokens
    
    def _flush_buffer(self, separator: str = "\n") -> str | None:
        """Flush the message buffer.
        
        In deferred mode, if there are remaining messages in the buffer, they are
        concatenated into a final document and returned before the buffer is cleared.
        
        Returns:
            `str | None`:
                The concatenated document of remaining messages in deferred mode,
                or ``None`` if the buffer is empty or in eager mode.
        """
        if self._deferred and self._message_buffer:
            doc = separator.join(self._message_buffer)
            self._clear_buffer()
            return doc
        self._clear_buffer()
        return None
    
    def _clear_buffer(self) -> None:
        """Clear the message buffer and reset token accounting."""
        self._message_buffer.clear()
        if self._tokenizer is not None:
            self._buffer_total_tokens = 0
