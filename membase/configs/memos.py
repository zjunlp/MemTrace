from .base import MemBaseConfig 
from pydantic import (
    Field, 
    field_validator,
    ValidationInfo, 
)
from copy import deepcopy 
from memos.configs.embedder import EmbedderConfigFactory
from memos.configs.graph_db import GraphDBConfigFactory
from memos.configs.internet_retriever import InternetRetrieverConfigFactory
from memos.configs.llm import LLMConfigFactory
from memos.configs.chunker import ChunkerConfigFactory
from memos.configs.reranker import RerankerConfigFactory
from typing import Any, Literal


class MemOSConfig(MemBaseConfig):
    """The default configuration for MemOS.
    It is composed of multiple configurations for the MemOS system."""

    extractor_config: LLMConfigFactory = Field(
        ..., 
        description=(
            "The configuration for the extractor "
            "which is used to generate memory units from raw content."
        ),
        examples=[
            {
                "backend": "openai", 
                "config": {
                    # The following fields belong to the basic LLM configuration. 
                    "model_name_or_path": "gpt-4.1-mini", 
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "top_p": 0.95,
                    "top_k": 50,
                    # The following fields belong to the openai configuration. 
                    "api_key": "YOUR_OPENAI_API_KEY", 
                    "api_base": "YOUR_OPENAI_API_BASE",
                }, 
            },
        ], 
    )
    dispatcher_config: LLMConfigFactory = Field(
        ..., 
        description=(
            "The configuration for the dispatcher "
            "which is used to rephrase the user' query. "
            "It is also used to extract related keywords and tags from the user's query."
        ),
        examples=[
            {
                "backend": "openai", 
                "config": {
                    # The following fields belong to the basic LLM configuration. 
                    "model_name_or_path": "gpt-4.1-mini", 
                    "temperature": 0.7,
                    "max_tokens": 8192,
                    "top_p": 0.95,
                    "top_k": 50,
                    # The following fields belong to the openai configuration. 
                    "api_key": "YOUR_OPENAI_API_KEY", 
                    "api_base": "YOUR_OPENAI_API_BASE",
                }, 
            },
        ],
    )
    embedding_config: EmbedderConfigFactory = Field(
        ..., 
        description="The configuration for the embedding model.",
        examples=[
            {
                "backend": "sentence_transformer",
                "config": {
                    # The following fields belong to the basic embedding configuration. 
                    "embedding_dims": 384,
                    "model_name_or_path": "all-MiniLM-L6-v2",
                    "max_tokens": 256,
                    # The following fields belong to the sentence transformer configuration. 
                    "trust_remote_code": True, 
                },
            },
        ], 
    )
    reranker_config: RerankerConfigFactory | None = Field(
        default=None, 
        description="The configuration for the reranker model.",
        examples=[
            {
                "backend": "cosine_local",
                "config": {
                    # The following fields belong to the cosine local reranker configuration. 
                    "level_weights": {"topic": 1.0, "concept": 1.0, "fact": 1.0},
                    "level_field": "background",
                },
            }, 
        ], 
    )
    graph_db: GraphDBConfigFactory = Field(
        ...,
        description="Graph database configuration for the tree-memory storage",
        examples=[
            {
                "backend": "neo4j", 
                "config": {
                    # The following fields belong to the basic graph database configuration. 
                    # For neo4j, you need to first run 
                    #   docker run -p7474:7474 -p7687:7687 -d -e NEO4J_AUTH=neo4j/password neo4j:latest
                    # to start the server, then run the MemOS memory construction script.
                    # You can use 
                    #   docker exec -it YOUR_CONTAINER_ID cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"
                    # to clear the database.
                    # To get a list of running Docker containers, you can use:
                    #   docker ps
                    "uri": "bolt://localhost:7687", 
                    "user": "neo4j", 
                    "password": "password", 
                    # The following fields belong to the neo4j configuration. 
                    "db_name": "alice", 
                    "use_multi_db": True,  
                    "user_name": None, 
                    "embedding_dimension": 384,
                }, 
            }, 
        ], 
    )
    internet_retriever: InternetRetrieverConfigFactory | None = Field(
        default=None, 
        description="The configuration for the internet retriever.",
        examples=[
            {
                "backend": "google",
                "config": {
                    # The following fields belong to the basic internet retriever configuration. 
                    "api_key": "YOUR_GOOGLE_API_KEY", 
                    "search_engine_id": "YOUR_SEARCH_ENGINE_ID", 
                    # The following fields belong to the google custom search configuration. 
                    "max_results": 20, 
                    "num_per_request": 10, 
                }, 
            }
        ], 
    )

    # This field is used in the memory extraction module. 
    chunker_config: ChunkerConfigFactory = Field(
        ...,
        description="The configuration for the chunker.",
        examples=[
            {
                "backend": "sentence",
                "config": {
                    # The following fields belong to the basic chunker configuration. 
                    "tokenizer_or_token_counter": "gpt2",
                    "chunk_size": 256,
                    "chunk_overlap": 64,
                    "min_sentences_per_chunk": 1,
                }, 
            },
        ],
    )

    memory_size: dict[str, int] = Field(
        default_factory=lambda: {
            "WorkingMemory": 20,
            "LongTermMemory": 1500,
            "UserMemory": 480,
        }, 
        description=(
            "Maximum item counts for each memory bucket. "
            "There are three memory buckets in MemOS: WorkingMemory, LongTermMemory, and UserMemory."
        ),
        examples=[
            {
                "WorkingMemory": 20, 
                "LongTermMemory": 10000, 
                "UserMemory": 10000
            }, 
        ], 
    )
    search_strategy: dict[str, bool] = Field(
        default_factory=lambda: {
            "bm25": False, 
            "cot": False, 
            "fast_graph": False, 
        }, 
        description=(
            "The search strategy for the memory retrieval. "
            "The dictionary keys represent different strategies and the boolean values indicate whether they are enabled. "
            "The basic retrieval strategies include graph retrieval and dense retrieval. "
            "Graph retrieval matches extracted keywords and tags from the goal parser against node metadata (keywords and tags). "
            "If BM25 is enabled, sparse retrieval will also be applied. "
            "This strategy uses both the original query and extracted keywords for searching. "
            "If CoT is enabled, MemOS will first decompose complex queries into several sub-queries before retrieval, " 
            "applying dense retrieval for all sub-queries. "
            "If fast graph search is enabled, the goal parser tokenizes the query and uses the resulting tokens as keywords and "
            "tags for graph retrieval. "
            "Otherwise (unless in 'fine' mode), the original query itself is used as keywords/tags without tokenization. "
            "Additionally, when fast graph search is active, the system performs keyword-based search in the database using the extracted keywords."
        ),
        examples=[
            {
                "bm25": True, 
                "cot": True, 
                "fast_graph": True, 
            }, 
        ], 
    )
    reorganize: bool = Field(
        default=False, 
        description=(
            "Whether to reorganize the tree memory. " 
            "If enabled, MemOS will periodically perform hierarchical clustering on graph nodes "
            "and use a large language model for summarization. "
            "When adding new graph nodes, if a conflict is detected, MemOS will also reorganize the graph structure."
        ),
    )
    mode: Literal["sync", "async"] = Field(
        default="sync",
        description=(
            "Whether to use asynchronous mode in memory add. "
            "If `sync` mode is used, MemOS will refresh and clear the working memory after adding new memories. "
            "Otherwise, the working memory will not be refreshed and cleared."
        ),
    )
    include_embedding: bool = Field(
        default=False,
        description="Whether to include embedding in the memory retrieval result.",
    )

    memory_filename: str = Field(
        default="textual_memory.json", 
        description="The filename for storing memory items."
    )

    @field_validator("graph_db", mode="before")
    @classmethod
    def validate_db_config(
        cls, 
        v: dict[str, Any] | GraphDBConfigFactory, 
        info: ValidationInfo,
    ) -> dict[str, Any]:
        """Ensure the database name is equal to the user id.

        If the memory database is not multi-database mode, a shared database is used with 
        the user name as the partition key. If the memory database is in multi-database mode,
        each user serves as a separate database.
        
        Args:
            v (`dict[str, Any] | GraphDBConfigFactory`): 
                The graph database configuration to validate.
            info (`ValidationInfo`): 
                The validation information.
                
        Returns:
            `dict[str, Any]`: 
                The validated graph database configuration.
        """
        user = info.data["user_id"]
        if isinstance(v, dict):
            v = deepcopy(v)
            if "config" not in v:
                v["config"] = {}
        else:
            v = v.model_dump()

        config = v["config"]
        backend = v.get("backend", "neo4j")
        if backend == "nebular":
            db_field = "space"
        else:
            db_field = "db_name"

        use_multi_db = config.get("use_multi_db", True)
        if use_multi_db:
            config[db_field] = user
            config["user_name"] = None
        else:
            if config.get(db_field) is None:  
                raise ValueError(
                    "The database name must be provided when multi-database mode is disabled."
                )
            config["user_name"] = user 

        embedding_config = info.data["embedding_config"]
        if embedding_config.config.embedding_dims is None: 
            raise ValueError(
                "The embedding dimension must be provided in the embedding configuration."
            )
        config["embedding_dimension"] = embedding_config.config.embedding_dims

        return v

    @field_validator("memory_filename", mode="before")
    @classmethod
    def validate_memory_filename(cls, v: str) -> str:
        """Ensure the memory filename is not same as the config filename.
        
        Args:
            v (`str`): 
                The memory filename to validate.
                
        Returns:
            `str`: 
                The validated memory filename.
        """
        if v == "config.json":
            raise ValueError(
                "The memory filename must be different from the config filename ('config.json')."
            )
        return v 

    def get_llm_models(self) -> list[str]:
        res = [
            self.extractor_config.config.model_name_or_path, 
            self.dispatcher_config.config.model_name_or_path, 
        ]
        return sorted(set(res))
