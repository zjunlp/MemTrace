from langgraph.store.memory import InMemoryStore
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    ToolMessage,
)
from langmem import create_memory_store_manager
from .base import MemBaseLayer
from ..configs.langmem import LangMemConfig
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
import pickle
import os
import json
from typing import Any, ClassVar


def _normalize_langmem_messages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """A helper function to process the messages of LangMem."""
    messages = kwargs.get("messages", args[0])
    assert len(messages) == 1, "Unconsidered Case where the number of messages is not equal to 1."
    messages = messages[0]
    normalized_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            normalized_messages.append(
                {
                    "role": "system",
                    "content": message.content
                }
            )
        elif isinstance(message, HumanMessage):
            normalized_messages.append(
                {
                    "role": "user",
                    "content": message.content
                }
            )
        elif isinstance(message, AIMessage):
            if message.content is not None and not isinstance(message.content, str):
                raise ValueError(
                    f"The content of the message is '{type(message.content)}' instead of 'string'."
                )
            normalized_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": str(tool_call["args"]),
                            }
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            )
        elif isinstance(message, ToolMessage):
            # See https://platform.openai.com/docs/guides/function-calling
            normalized_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
        else:
            raise ValueError(f"Message type '{type(message)}' is not supported.")
    
    return normalized_messages


def _extract_langmem_model(
    llm_model: str,
    query_model: str | None,
    *args: Any,
    **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """A helper function to extract the model name and metadata for LangMem."""
    llm_model = llm_model.split(':', 1)[1]
    query_model = query_model.split(':', 1)[1] if query_model is not None else None

    messages = kwargs.get("messages", args[0])
    assert len(messages) == 1, "Unconsidered Case where the number of messages is not equal to 1."
    messages = messages[0]
    # The following parameters are used in LiteLLM's token counter.
    metadata = {
        "tools": kwargs.get("tools"),
        "tool_choice": kwargs.get("tool_choice"),
    }
    if isinstance(messages[0], HumanMessage) and messages[0].content.startswith(
        "Use parallel tool calling to search for distinct memories relevant to this conversation."
    ):
        if query_model is None:
            raise ValueError("Query model is not provided.")
        return query_model, metadata
    return llm_model, metadata


def _extract_langmem_output(response: Any) -> dict[str, list[dict[str, str]] | str | float | int]:
    """A helper function to extract the output for LangMem."""
    assert len(response.generations) == 1, "Unconsidered Case."
    assert len(response.generations[0]) == 1, "Unconsidered Case."
    return {
        "messages": _normalize_langmem_messages(
            [[response.generations[0][0].message]]
        )
    }


class LangMemLayer(MemBaseLayer):

    layer_type: ClassVar[str] = "LangMem"

    def __init__(self, config: LangMemConfig) -> None:
        """Create an interface of LangMem. The implementation is based on the
        third-party library `langmem`."""
        self._llm_model = init_chat_model(
            config.llm_model,
            **config.llm_kwargs,
        )
        self._query_model = (
            None
            if config.query_model is None
            else init_chat_model(
                config.query_model,
                **config.llm_kwargs,
            )
        )
        self._store = InMemoryStore(
            index={
                "dims": config.retriever_dim,
                "embed": init_embeddings(
                    config.retriever_name_or_path,
                    **config.embedding_kwargs,
                ),
                "fields": ["content"],   # `kind` is ignored as there is only one kind of memory.
            },
        )
        self.memory_layer = create_memory_store_manager(
            self._llm_model,
            enable_inserts=config.enable_inserts,
            enable_deletes=config.enable_deletes,
            query_model=self._query_model,
            query_limit=config.query_limit,
            namespace=("memories", config.user_id),
            store=self._store,
        )
        self.config = config

        # Store each memory unit's id.
        self._memory_ids = {}
    
    @property
    def llm_model(self) -> BaseChatModel:
        return self._llm_model

    def add_message(self, message: Message, **kwargs: Any) -> None:
        message_dict = {
            "role": message.role,
            "name": message.name,
            "content": f"{message.content}\nTimestamp: {message.timestamp}",
        }
        # See https://langchain-ai.github.io/langmem/background_quickstart/
        # `kwargs` can include some optional parameters, e.g., `max_steps`.
        final_puts = self.memory_layer.invoke({"messages": [message_dict]}, **kwargs)
        # Some operations update contents of previous memory units.
        for final_put in final_puts:
            self._memory_ids[final_put["key"]] = final_put["value"]

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        message_level = kwargs.pop("message_level", True)
        if message_level not in [True, False]:
            raise TypeError(
                "`message_level` must be a boolean to indicate whether the messages "
                "are added to the memory layer message by message or as a whole."
            )
        
        if message_level:
            for message in messages:
                self.add_message(message, **kwargs)
        else:
            message_dicts = [
                {
                    "role": m.role,
                    "name": m.name,
                    "content": f"{m.content}\nTimestamp: {m.timestamp}",
                }
                for m in messages
            ]
            final_puts = self.memory_layer.invoke({"messages": message_dicts}, **kwargs)
            for final_put in final_puts:
                self._memory_ids[final_put["key"]] = final_put["value"]
    
    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        memories = self.memory_layer.search(query=query, limit=k, **kwargs)
        outputs = []
        for memory in memories:
            memory_dict = memory.dict()
            content = memory_dict["value"]["content"]
            metadata = {
                key: value
                for key, value in memory_dict.items() if key != "value"
            }
            outputs.append(
                MemoryEntry(
                    content=content,
                    formatted_content=content,
                    metadata=metadata,
                )
            )
        return outputs

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory_layer.delete(memory_id)
            if memory_id in self._memory_ids:
                del self._memory_ids[memory_id]
            return True
        except Exception as e:
            print(f"Error in delete method in LangMemLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        if "content" not in kwargs:
            raise KeyError("`content` is required in `kwargs` for LangMem layer.")
        content = kwargs.pop("content")
        try:
            self.memory_layer.put(
                memory_id,
                {"content": content},
                **kwargs
            )
            return True
        except Exception as e:
            print(f"Error in update method in LangMemLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id
        pkl_path = os.path.join(self.config.save_dir, f"{user_id}.pkl")
        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(pkl_path) or not os.path.exists(config_path):
            return False
        
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = json.load(f)
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )
        config = LangMemConfig(**config_dict)
        self._llm_model = init_chat_model(
            config.llm_model,
            **config.llm_kwargs,
        )
        self._query_model = (
            None
            if config.query_model is None
            else init_chat_model(
                config.query_model,
                **config.llm_kwargs,
            )
        )
        self._store = InMemoryStore(
            index={
                "dims": config.retriever_dim,
                "embed": init_embeddings(
                    config.retriever_name_or_path,
                    **config.embedding_kwargs,
                ),
                "fields": ["content"],
            }
        )
        self.memory_layer = create_memory_store_manager(
            self._llm_model,
            enable_inserts=config.enable_inserts,
            enable_deletes=config.enable_deletes,
            query_model=self._query_model,
            query_limit=config.query_limit,
            namespace=("memories", config.user_id),
            store=self._store,
        )
        self.config = config
        
        with open(pkl_path, "rb") as f:
            predefined_memory_units = pickle.load(f)
        self._memory_ids.clear()

        for memory_unit in predefined_memory_units:
            self.memory_layer.put(**memory_unit)
            self._memory_ids[memory_unit["key"]] = memory_unit["value"]

        return True

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump()
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # In LangMem, we don't store the vector embeddings.
        preserved_memory_units = []
        for key, value in self._memory_ids.items():
            # Note that some memory units have been deleted.
            if self.memory_layer.get(key) is not None:
                memory_unit = {
                    "key": key,
                    "value": value,
                }
                preserved_memory_units.append(memory_unit)

        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(preserved_memory_units, f)
    
    def get_patch_specs(self) -> list[PatchSpec]:
        getter, setter = make_attr_patch(self.llm_model, "generate")
        spec = PatchSpec(
            name=f"{self.llm_model.__class__.__name__}.generate",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: _extract_langmem_model(
                    self.config.llm_model,
                    self.config.query_model,
                    *args,
                    **kwargs
                ),
                extract_input_dict=lambda *args, **kwargs: {
                    # NOTE: LangMem uses the same prompt to generate and update memories.
                    # These two types of operations are handled by the same forward pass of LLMs.
                    "messages": _normalize_langmem_messages(*args, **kwargs),
                    "metadata": {
                        "op_type": "generation, update"
                    }
                },
                extract_output_dict=lambda response: _extract_langmem_output(response)
            )
        )
        specs = [spec]
        return specs 
