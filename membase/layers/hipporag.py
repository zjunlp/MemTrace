import os
import json
from collections import deque 
from .base import MemBaseLayer
from ._mixin import MessageBufferMixin
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from ..baselines.hipporag import HippoRAG
from ..baselines.hipporag.utils.config_utils import BaseConfig as HippoRAGBaseConfig
from ..configs.hipporag import HippoRAGConfig
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
from typing import Any, ClassVar


class HippoRAGLayer(MemBaseLayer, MessageBufferMixin):
    
    layer_type: ClassVar[str] = "HippoRAG2"
    
    def __init__(self, config: HippoRAGConfig) -> None:
        """Create an interface of HippoRAG 2. The implementation is based on the 
        [official implementation](https://github.com/OSU-NLP-Group/HippoRAG)."""
        self._init_layer(config)
        self.config = config
    
    def _init_layer(self, config: HippoRAGConfig) -> None:
        """Initialize HippoRAG 2 layer.
        
        Args:
            config (`HippoRAGConfig`): 
                The configuration for the HippoRAG 2 layer.
        """
        config_dict = config.model_dump(mode="python")
        config_dict.pop("user_id")
        config_dict.pop("num_overlap_msgs")
        config_dict.pop("message_separator")
        config_dict.pop("max_tokens")
        config_dict.pop("deferred")
        global_config = HippoRAGBaseConfig(**config_dict)
        self.memory_layer = HippoRAG(global_config)
        self._init_buffer(
            num_overlap_msgs=config.num_overlap_msgs,
            max_tokens=config.max_tokens,
            model_for_tokenizer=config.llm_name,
            deferred=config.deferred,
        )
    
    def add_message(self, message: Message, **kwargs: Any) -> None:
        text = f"Speaker {message.name} (role: {message.role}) says: {message.content}\nTimestamp: {message.timestamp}"

        # Add the current message into the buffer and get the document to index.
        doc = self._buffer_and_get_doc(
            message_content=text, 
            separator=self.config.message_separator,
        )
        if doc is not None:
            # Index the document into HippoRAG 2.
            self.memory_layer.index([doc]) 
        
    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)
    
    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        query_solutions = self.memory_layer.retrieve(
            queries=[query],
            num_to_retrieve=k
        )
        solution = query_solutions[0]

        outputs = []
        for i, doc in enumerate(solution.docs):
            score = solution.doc_scores[i]
            outputs.append(
                MemoryEntry(
                    content=doc,
                    metadata={
                        "score": float(score),
                    },
                    formatted_content=doc,
                ) 
            ) 
        return outputs
    
    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by its content.
        
        Note that HippoRAG 2 uses the document content as identifier, not a separate ID.
        
        Args:
            memory_id (`str`): 
                The document content to delete.
                
        Returns:
            `bool`: 
                Whether the memory is successfully deleted.
        """
        try:
            self.memory_layer.delete([memory_id])
            return True
        except Exception as e:
            print(f"Error in delete method in HippoRAGLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update the memory by deleting the old content and indexing the new content.
        
        Args:
            memory_id (`str`): 
                The document content to update. 
                HippoRAG 2 uses the document content as identifier, not a separate ID.
            **kwargs (`Any`):
                Keyword arguments for updating the memory. For HippoRAG 2, `content` is required.
                
        Returns:
            `bool`: 
                Whether the memory is successfully updated.
        """
        if "content" not in kwargs:
            raise KeyError("`content` is required in `kwargs` for HippoRAG 2.") 
        new_content = kwargs.pop("content")

        try:
            self.delete(memory_id)
            self.memory_layer.index([new_content])
            return True
        except Exception as e:
            print(f"Error in update method in HippoRAGLayer: \n\t{e.__class__.__name__}: {e}")
            return False
    
    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save layer config.
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)
        
        # Save buffer state.
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
            
    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id
        
        config_path = os.path.join(self.config.save_dir, "config.json")
        buffer_path = os.path.join(self.config.save_dir, "buffer_state.json")
        
        if not os.path.exists(config_path) or not os.path.exists(buffer_path):
            return False
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        try:
            if config_dict["user_id"] != user_id:
                raise ValueError(
                    f"The user id in the config file ({config_dict['user_id']}) "
                    f"does not match the user id ({user_id}) in the function call."
                )
            
            config = HippoRAGConfig(**config_dict) 
            self._init_layer(config)
            
            with open(buffer_path, "r", encoding="utf-8") as f:
                buffer_state = json.load(f)
            self._message_buffer = deque(buffer_state["message_buffer"])
            self._buffer_total_tokens = buffer_state["buffer_total_tokens"]

            self.config = config 
            return True
        except Exception as e:
            print(f"Error in load_memory method in HippoRAGLayer: \n\t{e.__class__.__name__}: {e}")
            return False 

    def flush(self) -> None:
        doc = self._flush_buffer(separator=self.config.message_separator)
        if doc is not None:
            self.memory_layer.index([doc])
    
    def get_patch_specs(self) -> list[PatchSpec]:
        def _extract_input_dict(*args, **kwargs):
            # For HippoRAG, the input is a list of `TextChatMessage` which is a dict with "role" and "content" keys.
            # Note there are some cases where all arguments are passed as keyword arguments in practice.
            messages = kwargs.get(
                "messages_list", 
                kwargs.get("messages", args[0] if len(args) > 0 else None)
            )
            if messages is None:
                raise ValueError("The wrapped function should have a `messages` argument.")

            if "messages_list" in kwargs: 
                if len(messages) > 1:
                    raise ValueError(
                        "Memory evaluation does not support delayed operations, so batch memory "
                        "addition will not occur. For HippoRAG, each indexing operation processes "
                        "only a single document at a time."
                    )
                messages = messages[0] 
            return {
                "messages": messages, 
                "metadata": {
                    "op_type": "generation"
                }
            }
        
        def _extract_output_dict(result):
            response = result[0] 
            if isinstance(response, list): 
                if len(response) > 1:
                    raise ValueError(
                        "Memory evaluation does not support delayed operations, so batch memory "
                        "addition will not occur. For HippoRAG, each indexing operation processes "
                        "only a single document at a time."
                    )
                response = response[0]

            return {
                "messages": response,
            }

        specs = [] 
        for method_name in ["batch_infer", "infer"]:
            getter, setter = make_attr_patch(self.memory_layer.openie.llm_model, method_name)
            spec = PatchSpec(
                name=f"{self.memory_layer.openie.llm_model.__class__.__name__}.{method_name}",
                getter=getter,
                setter=setter,
                wrapper=token_monitor(
                    extract_model_name=lambda *args, **kwargs: (self.config.llm_name, {}),
                    extract_input_dict=_extract_input_dict,
                    extract_output_dict=_extract_output_dict,
                )
            )
            specs.append(spec)
        return specs