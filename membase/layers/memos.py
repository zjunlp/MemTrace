import os
import json
from memos.configs.llm import LLMConfigFactory
from memos.configs.memory import TreeTextMemoryConfig
from memos.memories.textual.tree import TreeTextMemory
from memos.configs.mem_reader import MemReaderConfigFactory
from memos.mem_reader.factory import MemReaderFactory
from memos.llms.factory import LLMFactory
from memos.memories.textual.tree_text_memory.organize.manager import MemoryManager
from .base import MemBaseLayer 
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from ..configs.memos import MemOSConfig
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
from typing import Any, ClassVar


class MemOSLayer(MemBaseLayer):

    layer_type: ClassVar[str] = "MemOS"

    def __init__(self, config: MemOSConfig) -> None:
        """Create an interface of MemOS. The implementation is based on the 
        third-party library `MemoryOS`."""
        self._init_layer(config)
        self.config = config 

    def add_message(self, message: Message, **kwargs: Any) -> None:
        # MemOS does not use the 'name' field directly. 
        # Therefore, we incorporate the name information into the message content to retain speaker identity.
        text = f"Speaker {message.name} (role: {message.role}) says: {message.content}"
        message_dict = {
            "role": message.role,
            "name": message.name,
            "content": text,
            "chat_time": message.timestamp,
        }
        
        # Message object supported by MemOS can contain role, name, content and timestamp 
        mode = kwargs.get("mode", "fine")
        session_id = kwargs.get("session_id", "")
        info = {
            "user_id": self.config.user_id, 
            "session_id": session_id,
        }

        # The memory reader will assign a memory type (either long-term memory or user memory) 
        # to each generated memory unit. 
        # See https://github.com/MemTensor/MemOS/blob/v2.0.1/src/memos/templates/mem_reader_prompts.py. 
        memory_items = self.reader.get_memory(
            [[message_dict]], 
            "chat", 
            info, 
            mode=mode,
        )[0]
        self.memory_layer.add(memory_items)

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
            mode = kwargs.get("mode", "fine")
            session_id = kwargs.get("session_id", "")
            info = {
                "user_id": self.config.user_id, 
                "session_id": session_id,
            }

            message_dicts = [] 
            for m in messages:
                text = f"Speaker {m.name} (role: {m.role}) says: {m.content}"
                message_dicts.append(
                    {
                        "role": m.role, 
                        "name": m.name, 
                        "content": text, 
                        "chat_time": m.timestamp,
                    }
                )

            memory_items = self.reader.get_memory(
                [message_dicts], 
                "chat", 
                info, 
                mode=mode,
            )[0]
            self.memory_layer.add(memory_items)
    
    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        memories = self.memory_layer.search(query, top_k=k, **kwargs)
        outputs = [] 
        for memory in memories:
            memory_dict = memory.model_dump()
            outputs.append(
                MemoryEntry(
                    content=memory_dict["memory"], 
                    metadata={
                        key: value
                        for key, value in memory_dict["metadata"].items()
                        if key != "embedding"
                    }, 
                    formatted_content=memory_dict["memory"]
                )
            )
        return outputs  

    def delete(self, memory_id: str) -> bool:
        try:
            self.memory_layer.delete([memory_id])
            return True
        except Exception as e:
            print(f"Error in delete method in MemOSLayer: \n\t{e.__class__.__name__}: {e}")
            return False

    def update(self, memory_id: str, **kwargs) -> bool:
        raise NotImplementedError(
            "Currently, the tree memory in MemOS does not support updating existing memories."
        )

    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id
        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(config_path):
            return False 

        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = json.load(f) 
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )

        file_path = os.path.join(self.config.save_dir, self.config.memory_filename)
        if not os.path.exists(file_path):
            return False 

        config = MemOSConfig(**config_dict) 
        self._init_layer(config)
        self.config = config
        
        try:
            self.memory_layer.load(self.config.save_dir)
            return True 
        except Exception as e:
            print(f"Error in load_memory method in MemOSLayer: \n\t{e.__class__.__name__}: {e}")
            return False 

    def _init_layer(self, config: MemOSConfig) -> None:
        """Initialize MemOS layer.
        
        Args:
            config (`MemOSConfig`): 
                The configuration for the MemOS layer.
        """
        config_dict = config.model_dump(mode="python") 

        tree_mem_config = TreeTextMemoryConfig(
            extractor_llm=config_dict["extractor_config"],
            dispatcher_llm=config_dict["dispatcher_config"],
            embedder=config_dict["embedding_config"],
            reranker=config_dict["reranker_config"],
            graph_db=config_dict["graph_db"],
            internet_retriever=config_dict["internet_retriever"],
            reorganize=config_dict["reorganize"],
            memory_size=config_dict["memory_size"],
            search_strategy=config_dict["search_strategy"],
            mode=config_dict["mode"],
            include_embedding=config_dict["include_embedding"],
            memory_filename=config_dict["memory_filename"],
        )
        self.memory_layer = TreeTextMemory(tree_mem_config)

        # Reset some fields to avert singleton issues.
        llm_config = LLMConfigFactory(**config_dict["extractor_config"])
        llm_class = LLMFactory.backend_to_class[llm_config.backend]
        self.memory_layer.extractor_llm = llm_class(llm_config.config)
        self.memory_layer.dispatcher_llm = llm_class(llm_config.config)
        self.memory_layer.memory_manager = MemoryManager(
            self.memory_layer.graph_store,
            self.memory_layer.embedder,
            self.memory_layer.extractor_llm,
            memory_size=tree_mem_config.memory_size or {
                "WorkingMemory": 20,
                "LongTermMemory": 1500,
                "UserMemory": 480,
            },
            is_reorganize=self.memory_layer.is_reorganize,
        )

        reader_config = MemReaderConfigFactory(
            backend="simple_struct",
            config={
                "llm": config_dict["extractor_config"],
                "embedder": config_dict["embedding_config"],
                "chunker": config_dict["chunker_config"],
            },
        )
        reader_class = MemReaderFactory.backend_to_class[reader_config.backend]
        self.reader = reader_class(reader_config.config)
        
        # IMPORTANT: Due to `singleton_factory` in `LLMFactory.from_config`, `reader.llm` and 
        # `memory_layer.extractor_llm` may be the same object instance when using identical 
        # configs. This causes issues when applying separate monkey patches for token 
        # monitoring (e.g., distinguishing "generation" vs "update" operations).
        # Solution: Create an independent LLM instance for reader by directly instantiating 
        # the LLM class, bypassing the singleton cache.
        # See https://github.com/MemTensor/MemOS/blob/v2.0.1/src/memos/llms/factory.py. 
        self.reader.llm = llm_class(llm_config.config)

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json.
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python")
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        self.memory_layer.dump(self.config.save_dir, include_embedding=False)

    def get_patch_specs(self) -> list[PatchSpec]:
        specs = []

        # The memory reader in MemOS is used to generate memory units from raw content. 
        getter, setter = make_attr_patch(self.reader.llm, "generate")
        specs.append(
            PatchSpec(
                name = f"{self.reader.llm.__class__.__name__}.generate",
                getter=getter,
                setter=setter,
                wrapper=token_monitor(
                    extract_model_name=lambda *args, **kwargs: (
                        self.config.extractor_config.config.model_name_or_path, {}
                    ),
                    extract_input_dict=lambda *args, **kwargs: {
                        "messages": kwargs.get("messages", args[0] if len(args) > 0 else ""),
                        "metadata": {
                            "op_type": "generation"
                        }
                    },
                    extract_output_dict=lambda result: {
                        "messages": result
                    },
                ),
            )
        )

        # The extractor in MemOS is used to reorganize the memory units. 
        # It is used to update the current memory tree by merging similar or conflicting memory units. 
        # If there is a conflict between two memory units and it is cannot be resolved, the old memory unit will be deleted. 
        # Thus, we consider this operation as an update operation. 
        getter, setter = make_attr_patch(self.memory_layer.extractor_llm, "generate")
        specs.append(
            PatchSpec(
                name = f"{self.memory_layer.extractor_llm.__class__.__name__}.generate",
                getter=getter,
                setter=setter,
                wrapper=token_monitor(
                    extract_model_name=lambda *args, **kwargs: (self.config.extractor_config.config.model_name_or_path, {}),
                    extract_input_dict=lambda *args, **kwargs: {
                        "messages": kwargs.get("messages", args[0] if len(args) > 0 else ""),
                        "metadata": {
                            "op_type": "update"
                        }
                    },
                    extract_output_dict=lambda result: {
                        "messages": result
                    },
                ),
            )
        )
        return specs 
    
    def cleanup(self) -> None:
        self.memory_layer.delete_all()