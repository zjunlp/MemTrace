from .base import MemBaseConfig
from pydantic import Field
from typing import Literal


class HippoRAGConfig(MemBaseConfig):
    """Configuration for HippoRAG2 (online version)."""
    
    max_tokens: int | None = Field(
        default=None,
        description=(
            "Maximum total token count allowed in the buffer. When it is set, the message buffer "
            "trims oldest messages until the total token count is within this limit."
        ),
        gt=0,
    )
    num_overlap_msgs: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of previous messages to include when indexing a new message. "
            "This enables contextual continuity in online indexing scenarios. "
            "Set to 0 for independent message indexing."
        )
    )
    message_separator: str = Field(
        default="\n",
        description="Separator used to join messages into a document."
    )
    deferred: bool = Field(
        default=False,
        description=(
            "When it is enabled, messages are accumulated in the buffer and a document is "
            "only emitted when adding the next message would exceed the maximum token count. "
            "It requires `max_tokens` to be set. For example, if `max_tokens` is set to 1200, "
            "adding messages to the buffer will not trigger indexing until the next incoming "
            "message would cause the buffer's total token count to exceed 1200. At that point, "
            "the current buffer contents are concatenated into a single document for indexing, "
            "and the new message starts a fresh buffer. Combined with `num_overlap_msgs`, this "
            "achieves a chunking-with-overlap effect similar to offline chunking strategies."
        )
    )

    llm_name: str = Field(
        default="gpt-4o-mini",
        description=(
            "LLM model name for OpenIE extraction. It supports three modes:\n"
            "1. OpenAI-compatible API (default): Use model names like 'gpt-4o-mini', 'gpt-4o', "
            "or HuggingFace model ID like 'meta-llama/Llama-3.1-8B-Instruct' for vLLM-deployed models. "
            "Set `llm_base_url` to customize the API endpoint (e.g., 'http://localhost:8000/v1' for vLLM, "
            "or a proxy URL for OpenAI). If `llm_base_url` is None, it uses official OpenAI endpoint.\n"
            "2. Transformers local inference: Prefix with 'Transformers/', e.g., "
            "'Transformers/Qwen/Qwen2.5-7B-Instruct'. The model is loaded locally via "
            "HuggingFace Transformers with device_map='auto' and torch.bfloat16.\n"
            "3. AWS Bedrock: Prefix with 'bedrock', e.g., 'bedrock/anthropic.claude-3-5-haiku-20241022-v1:0'."
        ), 
    )
    llm_base_url: str | None = Field(
        default=None,
        description="Base URL for the LLM API."
    )

    embedding_model_name: str = Field(
        default="nvidia/NV-Embed-v2",
        description=(
            "Embedding model name. Model selection is based on name pattern matching "
            "(checked in order):\n"
            "- If 'GritLM' is in the name: GritLM embedding model.\n"
            "- If 'NV-Embed-v2' is in the name: Local NV-Embed-v2 (default, requires GPU).\n"
            "- If 'contriever' is in the name: Contriever embedding model.\n"
            "- If 'text-embedding' is in the name: OpenAI embedding API (e.g., 'text-embedding-3-small'), "
            "it uses `embedding_base_url` for custom endpoint.\n"
            "- If 'cohere' is in the name: Cohere embedding API.\n"
            "- If 'Transformers/' is in the name: the local sentence transformer, e.g., "
            "'Transformers/BAAI/bge-large-en-v1.5'.\n"
            "- If 'VLLM/' is in the name: vLLM-deployed embedding service, e.g., "
            "'VLLM/BAAI/bge-large-en-v1.5', it requires `embedding_base_url`."
        )
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for the embedding API. It is used when:\n"
            "- `embedding_model_name` starts with 'text-embedding' (OpenAI API): proxy or custom endpoint.\n"
            "- `embedding_model_name` starts with 'VLLM/': vLLM embedding endpoint, "
            "e.g., 'http://localhost:8001/v1/embeddings'."
        )
    )
    embedding_batch_size: int = Field(
        default=16,
        description=(
            "Batch size for embedding model calls. Larger values speed up encoding "
            "but require more GPU memory. Reduce if OOM occurs."
        ), 
    )
    
    save_openie: bool = Field(
        default=True,
        description="Save OpenIE results to disk."
    )
    
    openie_mode: Literal["online", "offline", "Transformers-offline"] = Field(
        default="online",
        description=(
            "OpenIE (Open Information Extraction) mode for entity and triple extraction. "
            "'online' (default): it uses OpenAI API for real-time extraction. "
            "'offline': it uses vLLM offline batch inference, more efficient for large corpora. "
            "'Transformers-offline': it uses local HuggingFace Transformers model for batch inference."
        )
    )
    
    retrieval_top_k: int = Field(
        default=10,
        description="Number of documents to retrieve."
    )
    linking_top_k: int = Field(
        default=5,
        description=(
            "Number of top-ranked phrases to use as seed nodes for Personalized PageRank (PPR). "
            "HippoRAG retrieval flow: (1) compute query-to-fact similarity scores; "
            "(2) LLM reranks top facts (triples like [subject, predicate, object]); "
            "(3) extract phrases (subject and object) from reranked facts and score them; "
            "(4) select top `linking_top_k` phrases as PPR seed nodes. "
            "In the original paper of HippoRAG 2, `linking_top_k` is set to 5."
        )
    )
    damping: float = Field(
        default=0.5,
        description=(
            "Damping factor for Personalized PageRank (PPR). "
            "In the original paper of HippoRAG 2, `damping` is set to 0.5."
        )
    ) 
    passage_node_weight: float = Field(
        default=0.05,
        description=(
            "Scaling factor for passage (document) node weights in Personalized PageRank (PPR). "
            "Each passage node's score is computed via query-passage embedding dot product. "
            "A small value (e.g., 0.05) reduces direct semantic similarity influence, "
            "letting PPR rely more on graph structure propagation from entity nodes. "
            "Higher values give more weight to dense retrieval signals. "
            "In the original paper of HippoRAG 2, `passage_node_weight` is set to 0.05. "
            "This achieves better retrieval performance empirically." 
        )
    )
    
    is_directed_graph: bool = Field(
        default=False,
        description="Whether the graph is directed."
    )
    synonymy_edge_sim_threshold: float = Field(
        default=0.8,
        description=(
            "Minimum embedding similarity score to create a synonymy edge between entity nodes. "
            "HippoRAG builds synonymy edges by: (1) computing embeddings for all entity nodes; "
            "(2) running K nearest neighbors (KNN) to find similar entities for each entity; "
            "(3) adding edges between entity pairs whose similarity is equal or greater than this threshold."
        )
    )

    def get_llm_models(self) -> list[str]:
        return [self.llm_name]
