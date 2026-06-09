from __future__ import annotations
from pydantic import (
    BaseModel, 
    Field, 
    model_validator,
)
from memory_layer.profile_manager.config import ScenarioType
from typing import Literal, Optional


class LLMConfig(BaseModel):
    """
    LLM configuration for memory extraction and processing.
    
    This configuration controls the language model used for various memory operations
    including boundary detection, MemCell extraction, episode generation, and profile
    extraction.
    """
    
    provider: Literal["openai"] = Field(
        default="openai",
        description="LLM provider type.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Model name.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for authentication. If not provided, will attempt to read from "
            "environment variable 'OPENAI_API_KEY' for OpenAI provider."
        ),
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API base URL. Override for custom endpoints.",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: int = Field(
        default=16384,
        ge=1,
        description="Maximum number of tokens for model output.",
    )


class EmbeddingConfig(BaseModel):
    """
    Embedding model configuration for semantic search and similarity computation.
    
    Embeddings are used for:
    - Semantic retrieval of memory content
    - MemCell clustering based on semantic similarity
    """
    
    provider: Literal["deepinfra", "vllm"] = Field(
        default="vllm",
        description="Embedding provider type.",
    )
    model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name. For huggingface models, you need use vLLM to serve the model.",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for embedding service. Falls back to LLM api_key if not set.",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="API base URL for embedding service. Falls back to LLM base_url if not set.",
    )
    embedding_dims: int = Field(
        default=384,
        ge=1,
        description=(
            "Embedding vector dimension. Must match the model's output dimension. "
            "Incorrect dimension will cause runtime errors."
        ),
    )


class BoundaryDetectionConfig(BaseModel):
    """
    Configuration for conversation boundary detection.
    
    Boundary detection determines when to split a conversation into MemCells.
    The system uses LLM-based semantic boundary detection combined with hard limits
    to ensure MemCells are appropriately sized for memory extraction.
    """
    
    hard_token_limit: int = Field(
        default=8192,
        ge=1,
        description=(
            "Maximum token count before forcing a MemCell split, regardless of semantic "
            "boundaries. This prevents excessively large MemCells that may exceed LLM context windows during extraction."
            "Set based on your LLM's context window and computational resources."
        ),
    )
    hard_message_limit: int = Field(
        default=50,
        ge=2,
        description="Maximum message count before forcing a MemCell split.",
    )
    use_smart_mask: bool = Field(
        default=True,
        description=(
            "Enable smart masking for boundary detection. When this is on and the message count reaches the threshold, "
            "if a boundary is detected, the last message (before the new one) will be added to the just-created MemCell, "
            "and also kept in the buffer for the next MemCell. "
            "This means the last message appears in two adjacent MemCells, like overlap in RAG chunking.",
        ),
    )
    smart_mask_threshold: int = Field(
        default=5,
        ge=2,
        description=(
            "Number of recent messages to include in boundary detection when smart masking "
            "is enabled. Higher values provide more context but increase token usage."
        ),
    )


class ClusteringConfig(BaseModel):
    """
    Configuration for MemCell clustering.
    
    The clustering algorithm uses embedding similarity and temporal proximity
    to determine cluster membership.
    """
    
    enabled: bool = Field(
        default=False,
        description="Enable MemCell clustering."
    )
    similarity_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity (0.0-1.0) for adding a MemCell to an existing "
            "cluster. Higher values create more focused clusters but may fragment "
            "related content."
        ),
    )
    max_time_gap_days: float = Field(
        default=7.0,
        ge=0.0,
        description=(
            "Maximum time gap in days between MemCells in the same cluster. "
            "MemCells separated by more than this duration will form separate clusters "
            "even if semantically similar. Set to 0 to disable temporal constraints."
        ),
    )


class ProfileConfig(BaseModel):
    """
    Configuration for user profile extraction.
    
    Profile extraction builds and maintains user profiles from conversation history.
    Profiles capture user characteristics, preferences, and behaviors that help
    personalize AI responses.
    """
    
    enabled: bool = Field(
        default=False,
        description=(
            "Enable automatic profile extraction. When disabled, no profile updates "
            "occur during message processing. You can still manually trigger profile "
            "extraction if needed."
        ),
    )
    scenario: ScenarioType = Field(
        default=ScenarioType.ASSISTANT,
        description=(
            "Profile extraction scenario, affects extraction prompts and focus areas." 
        ),
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence threshold (0.0-1.0) for accepting extracted profile "
            "attributes. The LLM provides confidence scores for each extraction:\n"
            "- 0.4-0.5: Accept uncertain extractions (may include noise)\n"
            "- 0.6-0.7: Balanced (recommended)\n"
            "- 0.8+: Conservative, only high-confidence extractions"
        ),
    )
    batch_size: int = Field(
        default=50,
        ge=1,
        description=(
            "Maximum number of MemCells to process in a single profile extraction "
            "batch. If the number of MemCells exceeds this threshold, only the most recent MemCells " 
            "up to this limit will be processed. This parameter is relevant when running EverMemOS offline."
        ),
    )


class OnlineRetrieverConfig(BaseModel):
    """
    Configuration for online memory retrieval.
    
    The retriever supports multiple search modes.
    """
    
    retrieval_mode: Literal["agentic", "hybrid", "bm25_only", "emb_only"] = Field(
        default="agentic",
        description=(
            "Primary retrieval mode. Choose based on your requirements:\n"
            "- 'agentic': Highest quality retrieval. It uses an LLM-guided multi-round process " 
            "where each retrieval round employs hybrid search. " 
            "The LLM assesses sufficiency and can generate refined queries for a second round if needed. " 
            "It is best for critical use cases that require strong recall and precision, but involves higher latency and LLM token costs.\n"
            "- 'hybrid': It balances semantic (embedding-based) and lexical (BM25) retrieval. It fuses both results using Reciprocal Rank Fusion (RRF), " 
            "which does not require normalization or calibration between the different score ranges of each retriever. " 
            "This approach offers both diversity and relevance, and is ideal for scenarios requiring a tradeoff between speed and quality.\n"
            "- 'bm25_only': It performs lexical (keyword-based) retrieval only (fast, ideal for exact keyword matches).\n"
            "- 'emb_only': It performs semantic (embedding-based) retrieval only (best for descriptive, paraphrased, or synonym-rich queries)."
        ),
    )
    bm25_top_n: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of candidate documents to retrieve from BM25 index before fusion. "
            "It should be greater or equal to `final_top_k`."
        ),
    )
    emb_top_n: int = Field(
        default=50,
        ge=1,
        description=(
            "Number of candidate documents to retrieve from embedding index before fusion. "
            "It should be greater or equal to `final_top_k`."
        ),
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        description=(
            "Reciprocal rank fusion parameter k. It controls how quickly scores decay "
            "with rank: score(d) = sum(1 / (k + rank_i)). Higher k values give more weight "
            "to lower-ranked documents, smoothing the fusion."
        ),
    )
    final_top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "Maximum number of documents to return from retrieval (before any reranking). "
            "This is the primary result limit." 
        ),
    )
    use_reranker: bool = Field(
        default=False,
        description="Enable neural reranking of retrieved results.",
    )
    reranker_provider: Literal["deepinfra", "vllm"] = Field(
        default="vllm",
        description="Reranker service provider: 'deepinfra' or 'vllm'.",
    )
    reranker_model: str = Field(
        default="",
        description=(
            "Reranker model name. If empty, reads from RERANK_MODEL env var "
            "or defaults to 'Qwen/Qwen3-Reranker-4B'."
        ),
    )
    reranker_api_key: str = Field(
        default="",
        description=(
            "Reranker API key. If empty, reads from RERANK_API_KEY env var. "
            "For vLLM provider, defaults to 'EMPTY' if not set."
        ),
    )
    reranker_base_url: str = Field(
        default="",
        description=(
            "Reranker API base URL. If empty, reads from RERANK_BASE_URL env var "
            "or defaults based on provider."
        ),
    )
    reranker_instruction: str = Field(
        default=(
            "Determine if the passage contains specific facts, entities (names, dates, "
            "locations), or details that directly answer the question."
        ),
        description=(
            "Instruction prompt for the reranker model. It guides the reranker on what "
            "aspects to prioritize when scoring relevance. It is customizable based on your "
            "retrieval goals."
        ),
    )
    reranker_batch_size: int = Field(
        default=20,
        ge=1,
        description="Number of documents to process per reranker API batch.",
    )
    reranker_max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retry attempts for each reranker batch.",
    )
    reranker_retry_delay: float = Field(
        default=0.8,
        ge=0.0,
        description="Base retry delay in seconds (exponential backoff is used).",
    )
    reranker_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="Timeout in seconds for each reranker batch.",
    )
    reranker_fallback_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fall back to original ranking when batch success rate is below this threshold.",
    )
    reranker_concurrent_batches: int = Field(
        default=5,
        ge=1,
        description="Maximum number of concurrent reranker batches.",
    )
    use_multi_query: bool = Field(
        default=True,
        description=(
            "Enable multi-query generation in agentic mode. When the initial retrieval "
            "is insufficient, generates multiple complementary queries to approach the "
            "question from different angles. It improves recall for complex questions."
        ),
    )
    sufficiency_check_docs: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of top documents to include in the sufficiency check prompt. "
            "The LLM evaluates these documents to determine if they contain enough "
            "information to answer the query. Higher values provide more context but "
            "increase token usage."
        ),
    )
    
    @model_validator(mode="after")
    def validate_config_constraints(self) -> OnlineRetrieverConfig:
        """Validate configuration constraints."""
        if self.sufficiency_check_docs > self.final_top_k:
            raise ValueError(
                f"The number of documents for sufficiency check must be less than or equal to the final top k ({self.final_top_k}) "
                f"but {self.sufficiency_check_docs} is provided."
            )
        
        if self.bm25_top_n < self.final_top_k:
            raise ValueError(
                f"The number of documents for sparse search must be greater than or equal to the final top k ({self.final_top_k}) "
                f"but {self.bm25_top_n} is provided."
            )
        
        if self.emb_top_n < self.final_top_k:
            raise ValueError(
                f"The number of documents for dense search must be greater than or equal to the final top k ({self.final_top_k}) "
                f"but {self.emb_top_n} is provided."
            )
        
        return self


class MemoryExtractionConfig(BaseModel):
    """
    Configuration for memory extraction from MemCells.
    
    It controls which types of memories are extracted from conversation segments:
    - Episode Memory: Narrative summary of the conversation segment
    - Event Log: Atomic facts extracted from the episode (who, what, when, where)
    - Foresight: Predicted future needs or intentions
    """
    
    enable_foresight: bool = Field(
        default=False,
        description=(
            "Enable foresight (prospective memory) extraction. Foresight predicts "
            "future user needs, intentions, or scheduled events based on conversation "
            "content. Experimental feature - may increase extraction time and LLM costs. "
            "Useful for proactive AI assistants."
        ),
    )
    enable_event_log: bool = Field(
        default=True,
        description=(
            "Enable event log (atomic fact) extraction. Event logs decompose episode "
            "narratives into discrete facts (e.g., 'User mentioned their birthday is "
            "March 15th'). These atomic facts enable:\n"
            "- Fine-grained retrieval\n"
            "- Fact-based question answering\n"
            "- Structured knowledge accumulation\n"
            "Recommended to keep enabled for most use cases."
        ),
    )


class OnlineMemoryManagerConfig(BaseModel):
    """Configuration for the Online Memory Manager."""
    
    group_id: str = Field(
        default="default",
        description="Group/conversation identifier.",
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration.",
    )
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Embedding model configuration.",
    )
    boundary: BoundaryDetectionConfig = Field(
        default_factory=BoundaryDetectionConfig,
        description="Boundary detection settings.",
    )
    clustering: ClusteringConfig = Field(
        default_factory=ClusteringConfig,
        description="MemCell clustering configuration.",
    )
    profile: ProfileConfig = Field(
        default_factory=ProfileConfig,
        description="Profile extraction configuration.",
    )
    retrieval: OnlineRetrieverConfig = Field(
        default_factory=OnlineRetrieverConfig,
        description="Retrieval configuration.",
    )
    extraction: MemoryExtractionConfig = Field(
        default_factory=MemoryExtractionConfig,
        description="Memory extraction configuration.",
    )
