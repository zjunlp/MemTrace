import uuid
import os
import json
import pickle
from collections import deque
from functools import partial 
from smartcomment import (
    comment_mutation,
    comment_op,
    comment_op_scope,
    comment_variable,
    current_context,
    is_tracing_enabled,
    IdentityRegistry,
)
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from ._mixin import MessageBufferMixin 
from .base import MemBaseLayer
from ..configs.naive_rag import NaiveRAGConfig
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
from typing import Any, ClassVar


def _get_naive_rag_memory_identity(variable: dict) -> str: 
    """Get the identity of a memory unit in the naive RAG layer."""  
    if not isinstance(variable, dict):
        raise TypeError(
            f"The provided variable '{variable}' is not a dictionary."
        )
        
    memory_id = variable.get("id")
    if memory_id is not None:
        return f"memory-unit-{memory_id}"

    raise ValueError(
        f"It is unable to extract the identity from the provided variable '{variable}'."
    )

IdentityRegistry.register(
    "naive-rag-dict",
    _get_naive_rag_memory_identity,
    exist_ok=True,
)


class NaiveRAGLayer(MemBaseLayer, MessageBufferMixin):

    layer_type: ClassVar[str] = "NaiveRAG"

    def __init__(self, config: NaiveRAGConfig) -> None:
        """Create an interface of naive RAG. The implementation is based on the 
        third-party library `langchain`."""
        self._init_buffer(
            num_overlap_msgs=config.num_overlap_msgs,
            max_tokens=config.max_tokens,
            model_for_tokenizer=config.llm_model,
            deferred=config.deferred,
        )
        self.memory_layer = InMemoryStore(
            index={
                "dims": config.retriever_dim, 
                "embed": init_embeddings(
                    config.retriever_name_or_path,
                    **config.embedding_kwargs,
                ), 
                "fields": ["content"], 
            }, 
        ) 
        self.config = config

        # Store each memory unit's id.
        self._memory_ids = set()

    def get_namespace(self) -> tuple[str, str]:
        """Get the namespace of the memory layer.

        Returns:
            tuple[str, str]:
                A tuple containing the namespace prefix and the user identifier.
        """
        return ("memories", self.config.user_id)

    def add_message(self, message: Message, **kwargs: Any) -> None:
        text = f"Speaker {message.name} (role: {message.role}) says: {message.content}\nTimestamp: {message.timestamp}"

        runtime_raw_input = comment_variable(
            {
                "content": text,
                "name": message.name,
                "role": message.role,
                "timestamp": message.timestamp,
            },
            to_runtime=True,
            id_strategy=lambda _: message.id,
            encoding_fn=partial(
                json.dumps, 
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            category="message",
            metadata={
                **message.metadata,
                "timestamp": message.timestamp,
                "speakers": message.name,
            },
            comment=(
                "An input message fed into the memory pipeline. "
                "It triggers the memory system to extract valuable information " 
                "that is worth storing in the memory store. "
                "The name denotes the speaker of the message, the role denotes " 
                "the role of the speaker, and the timestamp denotes the time when " 
                "the message is sent."
            ),
        )

       
        buffer_snapshot = {
            "message_buffer": list(self._message_buffer),
            "buffer_total_tokens": self._buffer_total_tokens,
        }
        # Obtain the runtime handle of the current message buffer so that,
        # even after in-place updates to the buffer, previous snapshots remain accessible.
        runtime_buffer_snapshot = comment_variable(
            buffer_snapshot,
            to_runtime=True,
            id_strategy=lambda _: "message-buffer",
            encoding_fn=partial(
                json.dumps, 
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            category="message_buffer",
            comment=(
                "A message buffer that stores messages " 
                "not yet processed by the memory system. " 
                "Its data structure is a list."
            ), 
        )

        with comment_op_scope(
            op_name="memory_system.message_buffer_update",
            category="update",
            comment=(
                "Merge a new input message into the current message buffer "
                "and, when the buffer is ready, emit a memory unit "
                "that aggregates recent messages for the memory system to "
                "store."
            ),
            metadata={
                "deferred": self.config.deferred,
                "max_tokens": self.config.max_tokens,
                "num_overlap_msgs": self.config.num_overlap_msgs,
                "message_separator": self.config.message_separator,
                "llm_model_for_tokenizer": self.config.llm_model,
            },
        ):
            with comment_mutation(
                target=buffer_snapshot,
                inputs=[runtime_raw_input],
                id_strategy=lambda _: "message-buffer",
                encoding_fn=partial(
                    json.dumps, 
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                category="message_buffer",
                mutation_comment=(
                    "Add a new input message into the current message buffer. "
                    "The message buffer is updated and the oldest messages may be trimmed "
                    f"so that the buffer stays within the configured overlap ({self.config.num_overlap_msgs}) "
                    f"and max-token budgets ({self.config.max_tokens})."
                ),
                mutation_category="update",
                reuse_op=True,
            ):
                doc = self._buffer_and_get_doc(
                    message_content=text,
                    separator=self.config.message_separator,
                )
                buffer_snapshot["message_buffer"] = list(self._message_buffer)
                buffer_snapshot["buffer_total_tokens"] = self._buffer_total_tokens


            if doc is not None:
                # Index the document into naive RAG.
                mem_id = str(uuid.uuid4())
                value = {
                    "content": doc, 
                }
                self.memory_layer.put(self.get_namespace(), mem_id, value) 
                self._memory_ids.add(mem_id)

                runtime_doc = comment_variable(
                    doc,
                    to_runtime=True,
                    class_name="temporary_variable",
                    category="temporary_variable",
                    comment=(
                        "A temporary variable that holds the buffered "
                        "document emitted from the message buffer. "
                        "It will serve as the content of a memory unit, "
                        "forming a new memory unit that is inserted into "
                        "the memory store."
                    ),
                )
                comment_op(
                    inputs=[runtime_raw_input, runtime_buffer_snapshot],
                    outputs=[runtime_doc],
                    category="extraction",
                    comment=(
                        "The new input message joins the previous message "
                        "buffer state to produce a buffered document."
                    ), 
                    reuse_op=True,
                )

                # For the operation which adds the buffered document to the memory store,
                # we create a new operation scope.
                comment_op(
                    inputs=[runtime_doc],
                    outputs=[
                        (
                            {
                                "content": value["content"],
                                "id": mem_id,
                            },
                            {
                                "id_strategy": "naive-rag-dict",
                                "category": "memory_entry",
                                "encoding_fn": partial(
                                    json.dumps, 
                                    ensure_ascii=False,
                                    indent=4,
                                    sort_keys=True,
                                ),
                                "decoding_fn": json.loads,
                            }
                        )
                    ], 
                    op_name="memory_system.add",
                    category="update", 
                    comment=(
                        "The buffered document is embedded into a memory unit. "
                        "The memory unit is added into the memory store."
                    ),
                )

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)

    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        tracing_ctx = current_context()
        runtime_query = None
        if tracing_ctx is not None:
            runtime_query = tracing_ctx.get_variable("query")

        with comment_op_scope(
            op_name="memory_system.retrieve",
            category="retrieval",
            comment=(
                f"A task query searches the memory store for top-{k} relevant memories."
            ),
            metadata={
                "top_k": k,
                "embedding_model": self.config.retriever_name_or_path,
            },
        ):
            # It returns a list of `SearchItem` objects.
            # See https://reference.langchain.com/python/langgraph-sdk/schema/SearchItem.
            memories = self.memory_layer.search(
                self.get_namespace(),
                query=query,
                limit=k,
                **kwargs,
            )
            outputs = []
            for memory in memories:
                memory_dict = memory.dict()
                content = memory_dict["value"]["content"]
                metadata = {
                    key: value
                    for key, value in memory_dict.items() if key != "value"
                }

                if is_tracing_enabled():
                    metadata["trace_id"] = _get_naive_rag_memory_identity(
                        {"id": metadata["key"]}
                    )

                outputs.append(
                    MemoryEntry(
                        content=content,
                        formatted_content=content,
                        metadata=metadata,
                    )
                )

                comment_op(
                    inputs=[runtime_query],
                    outputs=[
                        (
                            {
                                "id": metadata["key"],
                                "content": content,
                            },
                            {
                                "id_strategy": "naive-rag-dict",
                                "category": "memory_entry",
                                "encoding_fn": partial(
                                    json.dumps, 
                                    ensure_ascii=False,
                                    indent=4,
                                    sort_keys=True,
                                ),
                                "decoding_fn": json.loads,
                            },
                        ),
                    ],
                    comment=(
                        f"One memory '{metadata['key']}' is retrieved from the memory store. "
                        f"Its score is {metadata.get('score', 'unknown')}."
                    ),
                    reuse_op=True,
                )
            return outputs

    def delete(self, memory_id: str) -> bool:
        namespace = self.get_namespace()

        item = self.memory_layer.get(namespace, memory_id)
        if item is None:
            return False

        # TODO: The message buffer is synchronized with deletion operations.
        try:
            self.memory_layer.delete(namespace, memory_id)
            self._memory_ids.remove(memory_id)
            return True
        except Exception as e:
            print(f"Error in delete method in NaiveRAGLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        namespace = self.get_namespace()

        item = self.memory_layer.get(namespace, memory_id)
        if item is None:
            return False

        # Existing fields in the memory unit are overwritten by matching
        # keys in `kwargs`. Extra keys in `kwargs` are added as new fields.
        # TODO: The message buffer is synchronized with update operations.
        new_value = {
            **item.value,
            **kwargs,
        }
        try:
            self.memory_layer.put(
                namespace,
                memory_id,
                new_value,
            )
            return True
        except Exception as e:
            print(f"Error in update method in NaiveRAGLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id
        pkl_path = os.path.join(self.config.save_dir, f"{user_id}.pkl")
        config_path = os.path.join(self.config.save_dir, "config.json")
        buffer_path = os.path.join(self.config.save_dir, "buffer_state.json") 
        if (
            not os.path.exists(pkl_path) or 
            not os.path.exists(config_path) or 
            not os.path.exists(buffer_path)
        ):
            return False 

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )

        config = NaiveRAGConfig(**config_dict)
        self._init_buffer(
            num_overlap_msgs=config.num_overlap_msgs,
            max_tokens=config.max_tokens,
            model_for_tokenizer=config.llm_model,
            deferred=config.deferred,
        )
        self.memory_layer = InMemoryStore(
            index={
                "dims": config.retriever_dim,
                "embed": init_embeddings(
                    config.retriever_name_or_path,
                    **config.embedding_kwargs,
                ),
                "fields": ["content"],
            },
        )

        with open(buffer_path, "r", encoding="utf-8") as f:
            buffer_state = json.load(f)
        self._message_buffer = deque(buffer_state["message_buffer"])
        self._buffer_total_tokens = buffer_state["buffer_total_tokens"]

        with open(pkl_path, "rb") as f:
            predefined_memory_units = pickle.load(f)

        self._memory_ids.clear()
        self.config = config
        namespace = self.get_namespace()
        for memory_unit in predefined_memory_units:
            self.memory_layer.put(
                namespace,
                **memory_unit
            )
            self._memory_ids.add(memory_unit["key"])

        return True

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Save layer config.
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump()
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        buffer_path = os.path.join(self.config.save_dir, "buffer_state.json")
        buffer_state = {
            "message_buffer": list(self._message_buffer),
            "buffer_total_tokens": self._buffer_total_tokens,
        }
        with open(buffer_path, "w", encoding="utf-8") as f:
            json.dump(
                buffer_state,
                f,
                ensure_ascii=False,
                indent=4,
            )

        # In NaiveRAG, we don't store the vector embeddings.
        preserved_memory_units = []
        namespace = self.get_namespace()
        for memory_id in self._memory_ids:
            item = self.memory_layer.get(namespace, memory_id)
            if item is not None:
                preserved_memory_units.append(
                    {
                        "key": memory_id,
                        "value": item.value,
                    }
                )

        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(preserved_memory_units, f)

    def flush(self) -> None:
        buffer_snapshot = {
            "message_buffer": list(self._message_buffer),
            "buffer_total_tokens": self._buffer_total_tokens,
        }
        runtime_message_buffer = comment_variable(
            buffer_snapshot,
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

        with comment_op_scope(
            op_name="memory_system.message_buffer_update",
            category="update",
            comment=(
                "Finalize the current message buffer at the end of the "
                "conversation. Any remaining buffered messages are "
                "aggregated into a last buffered document that forms a new "
                "memory unit for the memory system to store, and the "
                "message buffer is then cleared."
            ),
            metadata={
                "deferred": self.config.deferred,
                "max_tokens": self.config.max_tokens,
                "num_overlap_msgs": self.config.num_overlap_msgs,
                "message_separator": self.config.message_separator,
                "llm_model_for_tokenizer": self.config.llm_model,
            },
        ):
            doc = self._flush_buffer(separator=self.config.message_separator)
            if doc is not None:
                mem_id = str(uuid.uuid4())
                value = {
                    "content": doc, 
                }
                self.memory_layer.put(self.get_namespace(), mem_id, value) 
                self._memory_ids.add(mem_id)

                runtime_doc = comment_variable(
                    doc,
                    to_runtime=True,
                    class_name="temporary_variable",
                    category="temporary_variable",
                    comment=(
                        "A temporary variable that holds the buffered "
                        "document emitted from the message buffer. "
                        "It will serve as the content of a memory unit, "
                        "forming a new memory unit that is inserted into "
                        "the memory store."
                    ),
                )
                comment_op(
                    inputs=[runtime_message_buffer],
                    outputs=[runtime_doc],
                    op_name="memory_system.finalize_memory_unit",
                    category="extraction",
                    comment=(
                        "The memory system finalizes the remaining buffered messages into "
                        "a concrete buffered document because the conversation is ending."
                    ), 
                    reuse_op=True,
                )

                # Create a new operation scope for the memory unit addition.
                comment_op(
                    inputs=[runtime_doc],
                    outputs=[
                        (
                            {
                                "content": value["content"],
                                "id": mem_id,
                            },
                            {
                                "id_strategy": "naive-rag-dict",
                                "category": "memory_entry",
                                "encoding_fn": partial(
                                    json.dumps, 
                                    ensure_ascii=False,
                                    indent=4,
                                    sort_keys=True,
                                ),
                                "decoding_fn": json.loads,
                            }
                        )
                    ], 
                    op_name="memory_system.add",
                    category="update", 
                    comment=(
                        "The buffered document is embedded into a memory unit. "
                        "The memory unit is added into the memory store."
                    ),
                )

                with comment_mutation(
                    target=buffer_snapshot,
                    id_strategy=lambda _: "message-buffer",
                    encoding_fn=partial(
                        json.dumps, 
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="message_buffer",
                    mutation_comment=(
                        "After the remaining messages are flushed into a buffered document, "
                        "the memory system clears the message buffer."
                    ),
                    mutation_category="update",
                    reuse_op=True,
                ):
                    buffer_snapshot["message_buffer"] = list(self._message_buffer)
                    buffer_snapshot["buffer_total_tokens"] = self._buffer_total_tokens
