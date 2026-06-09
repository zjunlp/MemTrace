from .base import MemBaseLayer 
from ..baselines.amem.memory_system import AgenticMemorySystem, MemoryNote
from ..configs.amem import AMEMConfig
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


class AMEMLayer(MemBaseLayer):

    layer_type: ClassVar[str] = "A-MEM"

    def __init__(self, config: AMEMConfig) -> None:
        """Create an interface of A-MEM. The implementation is based on the 
        [official implementation](https://github.com/WujiangXu/A-mem-sys)."""
        self.memory_layer = AgenticMemorySystem(
            model_name=config.retriever_name_or_path,
            llm_backend=config.llm_backend,
            llm_model=config.llm_model,
            evo_threshold=config.evo_threshold,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            embedder_provider=config.embedding_provider,
            embedding_api_key=config.embedding_api_key,
            embedding_base_url=config.embedding_base_url,
            user_id=config.user_id, 
        )
        self.config = config 
    
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
        self.config = AMEMConfig(**config_dict)
        self.memory_layer = AgenticMemorySystem(
            model_name=self.config.retriever_name_or_path,
            llm_backend=self.config.llm_backend,
            llm_model=self.config.llm_model,
            evo_threshold=self.config.evo_threshold,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url,
            embedder_provider=self.config.embedding_provider,
            embedding_api_key=self.config.embedding_api_key,
            embedding_base_url=self.config.embedding_base_url,
            user_id=self.config.user_id,
        )
        
        with open(pkl_path, "rb") as f:
            predefined_states = pickle.load(f)
        self.memory_layer.evo_cnt = predefined_states["evo_cnt"]
        predefined_notes = predefined_states["notes"]

        documents, metadatas, ids, embeddings = [], [], [], [] 
        for note in predefined_notes:
            self.memory_layer.memories[note["id"]] = MemoryNote(
                content=note["content"],
                id=note["id"],
                keywords=note["keywords"],
                links=note["links"],
                retrieval_count=note["retrieval_count"],
                timestamp=note["timestamp"],
                last_accessed=note["last_accessed"],
                context=note["context"], 
                evolution_history=note["evolution_history"],
                category=note["category"],
                tags=note["tags"],
            )
            documents.append(note["database"]["document"])
            metadatas.append(note["database"]["metadata"])
            ids.append(note["database"]["id"])
            embeddings.append(note["database"]["embedding"])

        self.memory_layer.retriever.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        return True 

    def add_message(self, message: Message, **kwargs: Any) -> None:
        # See https://github.com/WujiangXu/A-mem/blob/main/test_advanced.py#L296. 
        text = f"Speaker {message.name} (role: {message.role}) says: {message.content}"
        self.memory_layer.add_note(text, time=message.timestamp)

    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages: 
            self.add_message(message, **kwargs)
    
    def retrieve(self, query: str, k: int = 10, **kwargs: Any) -> list[MemoryEntry]:
        memories = self.memory_layer.search_agentic(query, k=k)
        outputs = [] 
        for memory in memories:
            formatted_content = {
                "memory content": memory["content"], 
                "memory context": memory["context"],
                "memory keywords": str(memory["keywords"]),
                "memory tags": str(memory["tags"]),
                "talk start time": memory["timestamp"],
            }
            # See https://github.com/WujiangXu/A-mem/blob/main/memory_layer.py#L690.
            formatted_content = '\n'.join(
                [f"{key}: {value}" for key, value in formatted_content.items()]
            )
            metadata = {
                key: value
                for key, value in memory.items() if key != "content"
            }
            outputs.append(
                MemoryEntry(
                    content=memory["content"], 
                    formatted_content=formatted_content,
                    metadata=metadata,
                ),
            )
        return outputs 

    def delete(self, memory_id: str) -> bool:
        return self.memory_layer.delete(memory_id)

    def update(self, memory_id: str, **kwargs: Any) -> bool:
        return self.memory_layer.update(memory_id, **kwargs)

    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)

        # Write config.json.
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(),
        }
        with open(config_path, 'w', encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4)

        # Serialize notes with associated vector store entries.
        notes_serialized = []
        collection = self.memory_layer.retriever.collection

        for note in self.memory_layer.memories.values():
            fetched = collection.get(
                ids=[note.id], include=["documents", "metadatas", "embeddings"]
            )
            doc_value = fetched["documents"][0]
            meta_value = fetched["metadatas"][0]
            # Note that it is a numpy array, which is picklable. 
            emb_value = fetched["embeddings"][0]

            note_dict = {
                "content": note.content,
                "id": note.id,
                "keywords": note.keywords,
                "links": note.links,
                "retrieval_count": note.retrieval_count,
                "timestamp": note.timestamp,
                "last_accessed": note.last_accessed,
                "context": note.context,
                "evolution_history": note.evolution_history,
                "category": note.category,
                "tags": note.tags,
                "database": {
                    "document": doc_value,
                    "metadata": meta_value, 
                    "id": note.id,
                    "embedding": emb_value,
                },
            }
            notes_serialized.append(note_dict)

        payload = {
            "evo_cnt": self.memory_layer.evo_cnt,
            "notes": notes_serialized,
        }

        pkl_path = os.path.join(self.config.save_dir, f"{self.config.user_id}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(payload, f)

    def flush(self) -> None:
        self.memory_layer.consolidate_memories()

    def get_patch_specs(self) -> list[PatchSpec]:
        # In this case, we modify an instance's method. 
        # Other instances are not affected. 
        # Note that there is no need to check `response_format` parameter. 
        getter, setter = make_attr_patch(self.memory_layer.llm_controller.llm, "get_completion")
        spec = PatchSpec(
            name=f"{self.memory_layer.llm_controller.llm.__class__.__name__}.get_completion",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (self.config.llm_model, {}),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You must respond with a JSON object."
                        }, 
                        {
                            "role": "user",
                            "content": kwargs.get("prompt", args[0] if len(args) > 0 else "") 
                        }
                    ],
                    "metadata": {
                        "op_type": (
                            "generation"
                            if kwargs.get("prompt", args[0] if len(args) > 0 else "").startswith(
                                "Generate a structured analysis"
                            ) 
                            else "update"
                        )
                    }
                },
                extract_output_dict=lambda result: {
                    "messages": result
                },
            ),
        )
        return [spec] 
