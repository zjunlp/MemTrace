from .base import MemBaseConfig 
from ..baselines.evermemos.online_memory.config import (
    LLMConfig,
    EmbeddingConfig,
    BoundaryDetectionConfig,
    ClusteringConfig,
    ProfileConfig,
    OnlineRetrieverConfig,
    MemoryExtractionConfig,
)
from pydantic import Field


class EverMemOSConfig(MemBaseConfig):
    """Configuration for EverMemOS (online version).
    
    There are some other parameters that are not included in the configuration, 
    such as the language of the prompts. These parameters are set by the environment variable.
    For example, MEMORY_LANGUAGE=zh will set the language of the prompts to Chinese.
    """
    
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration for memory cell extraction and processing.",
        examples=[
            {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "YOUR_OPENAI_API_KEY",
                "base_url": "https://api.openai.com/v1",
                "temperature": 0.3,
                "max_tokens": 16384,
            },
        ],
    )
    embedding_config: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration.",
        examples=[
            {
                # You can run the following code to serve the embedding model: 
                #   vllm serve all-MiniLM-L6-v2 --port 8000 \
                #   --tensor-parallel-size 1 \
                #   --served-model-name all-MiniLM-L6-v2 \
                #   --gpu-memory-utilization 0.1
                "provider": "vllm",
                "model": "all-MiniLM-L6-v2",
                "api_key": "EMPTY",
                "base_url": "http://localhost:8000/v1",
                "embedding_dims": 384,
            },
        ],
    )
    boundary_config: BoundaryDetectionConfig = Field(
        default_factory=BoundaryDetectionConfig,
        description="Boundary detection configuration.",
        examples=[
            {
                "hard_token_limit": 8192,
                "hard_message_limit": 50,
                "use_smart_mask": True,
                "smart_mask_threshold": 5,
            },
        ],
    )
    clustering_config: ClusteringConfig = Field(
        default_factory=ClusteringConfig,
        description="Clustering configuration.",
        examples=[
            {
                "enabled": True,
                "similarity_threshold": 0.65,
                "max_time_gap_days": 7.0,
            },
        ],
    )
    profile_config: ProfileConfig = Field(
        default_factory=ProfileConfig,
        description="Profile extraction configuration.",
        examples=[
            {
                "enabled": False,
                "scenario": "assistant",
                "min_confidence": 0.6,
                "batch_size": 50,
            },
        ],
    )
    retrieval_config: OnlineRetrieverConfig = Field(
        default_factory=OnlineRetrieverConfig,
        description="Retrieval configuration.",
        examples=[
            {
                "retrieval_mode": "agentic",
                "bm25_top_n": 50,
                "emb_top_n": 50,
                "rrf_k": 60,
                "use_reranker": True,
                "reranker_provider": "vllm",
                "reranker_model": "Qwen3-Reranker-4B",
                "reranker_api_key": "EMPTY",
                "reranker_base_url": "http://localhost:8001/v1/rerank",
                "reranker_instruction": (
                    "Determine if the passage contains specific facts, entities (names, dates, "
                    "locations), or details that directly answer the question."
                ), 
                "reranker_batch_size": 20,
                "reranker_max_retries": 3,
                "reranker_retry_delay": 0.8,
                "reranker_timeout": 60.0,
                "reranker_fallback_threshold": 0.3,
                "reranker_concurrent_batches": 5,
                "use_multi_query": True,
                "final_top_k": 30,
                "sufficiency_check_docs": 10, 
            },
        ],
    )
    extraction_config: MemoryExtractionConfig = Field(
        default_factory=MemoryExtractionConfig,
        description="Memory extraction configuration.",
        examples=[
            {
                "enable_foresight": False,
                "enable_event_log": True,
            },
        ],
    )
    
    def get_llm_models(self) -> list[str]:
        return [self.llm_config.model]