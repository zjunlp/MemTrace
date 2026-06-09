import asyncio
import json
import time
import inspect 
import uuid
from functools import partial
from smartcomment import (
    comment_link,
    comment_op,
    comment_op_scope,
    comment_variable,
    current_context,
    is_tracing_enabled,
)
from online_memory.config import OnlineRetrieverConfig
from online_memory.index_manager import InMemoryIndexManager
from online_memory.prompts import (
    SUFFICIENCY_CHECK_PROMPT, 
    MULTI_QUERY_GENERATION_PROMPT, 
    REFINED_QUERY_PROMPT, 
) 
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple,
)


class OnlineRetriever:
    """Online retriever with multiple retrieval modes."""
    
    def __init__(
        self,
        index_manager: InMemoryIndexManager,
        config: Optional[OnlineRetrieverConfig] = None,
        llm_provider: Optional[Any] = None,
    ) -> None:
        """
        Initialize the retriever.
        
        Parameters
        ----------
        index_manager : InMemoryIndexManager
            The index manager for BM25 and embedding search.
        config : OnlineRetrieverConfig, optional
            Retriever configuration.
        llm_provider : Any, optional
            LLM provider for agentic retrieval.
        """
        self.index_manager = index_manager
        self.config = config or OnlineRetrieverConfig()
        self.llm_provider = llm_provider
        
        # Initialize reranker service
        self._init_reranker_service()
    
    def _init_reranker_service(self) -> None:
        """Initialize the reranker service from OnlineRetrieverConfig."""
        if self.config.use_reranker:
            from agentic_layer.rerank_service import (
                HybridRerankConfig,
                HybridRerankService,
            )
            
            # Create `HybridRerankConfig` from input parameters.
            rerank_config = HybridRerankConfig(
                primary_provider=self.config.reranker_provider,
                primary_api_key=self.config.reranker_api_key,
                primary_base_url=self.config.reranker_base_url,
                model=self.config.reranker_model,
            )
            self._reranker = HybridRerankService(config=rerank_config)
            print(f"  [Retriever] Initialized HybridRerankService | provider={rerank_config.primary_provider} | model={rerank_config.model}")
        else:
            self._reranker = None
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents for a query.
        
        Parameters
        ----------
        query : str
            Search query.
        k : int, optional
            Maximum number of results to return.
        **kwargs
            Additional parameters such as the mode of retrieval.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples.
        """
        if not query or len(self.index_manager) == 0:
            return []
        
        effective_k = k if k is not None else self.config.final_top_k
        mode = kwargs.get("retrieval_mode", self.config.retrieval_mode)
        
        if mode == "agentic":
            if self.llm_provider is None:
                print("Warning: Agentic mode requires LLM provider. Falling back to hybrid.")

                with comment_op_scope(
                    op_name="memory_system.hybrid_retrieval",
                    comment=(
                        f"A task query searches the memory store for top-{effective_k} relevant memories. "
                        "Note that the retrieval mode is set to 'agentic' but the large language model provider "
                        "is not available. Therefore, it falls back to hybrid retrieval. "
                    ),
                    category="retrieval",
                    metadata={
                        "mode": "hybrid_retrieval",
                        "top_k": effective_k,
                        "embedding_model": self.index_manager.embedding_config.model,
                        "rrf_k": self.config.rrf_k,
                    }, 
                ):
                    return await self._search_hybrid(query, effective_k)

            # For agentic retrieval, we don't need to wrap it in an operation scope.
            # Internally, it creates three operation scopes.
            return await self._search_agentic(query, effective_k)
        elif mode == "bm25_only":
            with comment_op_scope(
                op_name="memory_system.bm25_retrieval",
                comment=(
                    f"A task query searches the memory store for top-{effective_k} relevant memories."
                ),
                category="retrieval",
                metadata={
                    "mode": "sparse_retrieval",
                    "top_k": effective_k,
                }, 
            ):
                return self._search_bm25_only(query, effective_k)
        elif mode == "emb_only":
            with comment_op_scope(
                op_name="memory_system.embedding_retrieval",
                comment=(
                    f"A task query searches the memory store for top-{effective_k} relevant memories."
                ),
                category="retrieval",
                metadata={
                    "mode": "dense_retrieval",
                    "top_k": effective_k,
                    "embedding_model": self.index_manager.embedding_config.model,
                }, 
            ):
                return await self._search_emb_only(query, effective_k)
        else:  
            with comment_op_scope(
                op_name="memory_system.hybrid_retrieval",
                comment=(
                    f"A task query searches the memory store for top-{effective_k} relevant memories."
                ),
                category="retrieval",
                metadata={
                    "mode": "hybrid_retrieval",
                    "top_k": effective_k,
                    "embedding_model": self.index_manager.embedding_config.model,
                    "rrf_k": self.config.rrf_k,
                }, 
            ):
                return await self._search_hybrid(query, effective_k)
    
    def _search_bm25_only(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """BM25-only search."""
        results = self.index_manager.search_bm25(query, top_n=k)

        tracing_ctx = current_context()
        runtime_query = None
        if tracing_ctx is not None and is_tracing_enabled():
            # At the first place, we try to get the active query from the tracing context.
            query_candidates = tracing_ctx.query_variables(name_pattern="active_query")
            assert len(query_candidates) <= 1, (
                "Only one active query is allowed."
            )
            if not query_candidates:
                # If no active query is found, we use the original query.
                runtime_query = tracing_ctx.get_variable("query")

        for (doc, score) in results:
            if runtime_query is not None:
                # Note: A `None` runtime query can arise in two cases:
                # (1) tracing is not enabled, or
                # (2) this function is not invoked directly.
                comment_op(
                    inputs=[runtime_query],
                    outputs=[
                        (
                            {
                                "event_id": doc["event_id"],
                                "timestamp": doc["timestamp"],
                                "subject": doc["subject"],
                                "summary": doc["summary"],
                                "episode": doc["episode"],
                                "event_log": {
                                    "atomic_fact": doc["event_log"]["atomic_fact"],
                                } if doc["event_log"] is not None else None,
                                "foresights": {
                                    "foresight": doc["foresights"]["foresight"],
                                    "evidence": doc["foresights"]["evidence"],
                                    "start_time": doc["foresights"]["start_time"],
                                    "end_time": doc["foresights"]["end_time"],
                                    "duration_days": doc["foresights"]["duration_days"],
                                } if doc["foresights"] is not None else None,
                            },
                            {
                                "id_strategy": "evermemos-dict",
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
                        "Based on the lexical similarity (BM25) between the query and the memory, "
                        f"one memory '{doc['event_id']}' is retrieved from the memory store. "
                        f"Its score is {score}."
                    ),
                    reuse_op=True,
                ) 

        return results
    
    async def _search_emb_only(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Embedding-only search."""
        results = await self.index_manager.search_embedding(query, top_n=k)

        tracing_ctx = current_context()
        runtime_query = None
        if tracing_ctx is not None and is_tracing_enabled():
            # At the first place, we try to get the active query from the tracing context.
            query_candidates = tracing_ctx.query_variables(name_pattern="active_query")
            assert len(query_candidates) <= 1, (
                "Only one active query is allowed."
            )
            if not query_candidates:
                # If no active query is found, we use the original query.
                runtime_query = tracing_ctx.get_variable("query")

        for (doc, score) in results:
            if runtime_query is not None:
                # Note: A `None` runtime query can arise in two cases:
                # (1) tracing is not enabled, or
                # (2) this function is not invoked directly.
                comment_op(
                    inputs=[runtime_query],
                    outputs=[
                        (
                            {
                                "event_id": doc["event_id"],
                                "timestamp": doc["timestamp"],
                                "subject": doc["subject"],
                                "summary": doc["summary"],
                                "episode": doc["episode"],
                                "event_log": {
                                    "atomic_fact": doc["event_log"]["atomic_fact"],
                                } if doc["event_log"] is not None else None,
                                "foresights": {
                                    "foresight": doc["foresights"]["foresight"],
                                    "evidence": doc["foresights"]["evidence"],
                                    "start_time": doc["foresights"]["start_time"],
                                    "end_time": doc["foresights"]["end_time"],
                                    "duration_days": doc["foresights"]["duration_days"],
                                } if doc["foresights"] is not None else None,
                            },
                            {
                                "id_strategy": "evermemos-dict",
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
                        "Based on the semantic similarity between the query and the memory, "
                        f"one memory '{doc['event_id']}' is retrieved from the memory store. "
                        f"Its score is {score}."
                    ),
                    reuse_op=True,
                ) 

        return results
    
    async def _search_hybrid(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Hybrid search with RRF fusion.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L491. 
        """
        # Adjust `bm25_top_n` and `emb_top_n` if `k` is larger
        bm25_n = max(self.config.bm25_top_n, k)
        emb_n = max(self.config.emb_top_n, k)

        # `_search_hybrid()` can be invoked multiple times in sequence during
        # agentic retrieval. Even when two invocations produce the same
        # list-level retrieval results, we still give each list node a fresh
        # identity so different retrieval rounds do not collapse to one node
        # and form cycles in the trace graph.
        list_level_result_id_strategy = lambda results: (
            f"{', '.join(result['event_id'] for result in results) or 'empty'}"
            f" | ts={time.time_ns()}"
            f" | rand={uuid.uuid4().hex[:8]}"
        )

        tracing_ctx = current_context()
        runtime_query = None
        curret_is_active = False 
        if tracing_ctx is not None and is_tracing_enabled():
            # At the first place, we try to get the active query from the tracing context.
            query_candidates = tracing_ctx.query_variables(name_pattern="active_query")
            assert len(query_candidates) <= 1, (
                "Only one active query is allowed."
            )
            if not query_candidates:
                # If no active query is found, we use the original query.
                runtime_query = tracing_ctx.get_variable("query")

                # It denotes that this function is invoked directly, rather than being triggered via agentic search.
                # We simply view it as an active query.
                if tracing_ctx is not None and is_tracing_enabled():
                    tracing_ctx.register_variable(
                        "active_query", 
                        runtime_query,
                        overwrite=True,
                    )
                curret_is_active = True
            else:
                runtime_query = query_candidates[0]
        
        # Run BM25 and embedding search in parallel
        bm25_task = asyncio.to_thread(
            self.index_manager.search_bm25,
            query,
            bm25_n,
        )
        emb_task = self.index_manager.search_embedding(
            query,
            emb_n,
        )
        
        bm25_results, emb_results = await asyncio.gather(bm25_task, emb_task)

        runtime_bm25_results = comment_variable(
            [
                {
                    "event_id": doc["event_id"],
                    "timestamp": doc["timestamp"],
                    "subject": doc["subject"],
                    "summary": doc["summary"],
                    "episode": doc["episode"],
                    "event_log": {
                        "atomic_fact": doc["event_log"]["atomic_fact"],
                    } if doc["event_log"] is not None else None,
                    "foresights": {
                        "foresight": doc["foresights"]["foresight"],
                        "evidence": doc["foresights"]["evidence"],
                        "start_time": doc["foresights"]["start_time"],
                        "end_time": doc["foresights"]["end_time"],
                        "duration_days": doc["foresights"]["duration_days"],
                    } if doc["foresights"] is not None else None,
                    "score": score,
                }
                for doc, score in bm25_results
            ],
            to_runtime=True, 
            id_strategy=list_level_result_id_strategy,
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            category="sparse_retrieval_results",
            class_name="sparse_retrieval_results",
            comment=(
                f"The results of the top-{bm25_n} BM25 search. "
                "It is a list. Each element in the list is a retrieved memory unit. "
                "The score field in each memory unit represents the lexical similarity score " 
                "between the query and the retrieved memory unit."
            ),
        ) 
        runtime_emb_results = comment_variable(
            [
                {
                    "event_id": doc["event_id"],
                    "timestamp": doc["timestamp"],
                    "subject": doc["subject"],
                    "summary": doc["summary"],
                    "episode": doc["episode"],
                    "event_log": {
                        "atomic_fact": doc["event_log"]["atomic_fact"],
                    } if doc["event_log"] is not None else None,
                    "foresights": {
                        "foresight": doc["foresights"]["foresight"],
                        "evidence": doc["foresights"]["evidence"],
                        "start_time": doc["foresights"]["start_time"],
                        "end_time": doc["foresights"]["end_time"],
                        "duration_days": doc["foresights"]["duration_days"],
                    } if doc["foresights"] is not None else None,
                    "score": score,
                }
                for doc, score in emb_results
            ],
            to_runtime=True, 
            id_strategy=list_level_result_id_strategy,
            encoding_fn=partial(
                json.dumps,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            ),
            decoding_fn=json.loads,
            category="dense_retrieval_results",
            class_name="dense_retrieval_results",
            comment=(
                f"The results of the top-{emb_n} embedding search. "
                "It is a list. Each element in the list is a retrieved memory unit. "
                "The score field in each memory unit represents the semantic similarity score " 
                "between the query and the retrieved memory unit."
            ),
        ) 
        comment_op(
            inputs=[runtime_query],
            outputs=[runtime_bm25_results, runtime_emb_results],
            comment=(
                "The query is used to run hybrid retrieval in parallel. "
                "It produces two candidate memory lists. BM25 results are " 
                "ranked by lexical similarity and embedding results are ranked " 
                "by semantic similarity."
            ), 
            metadata={
                "bm25_top_n": bm25_n,
                "emb_top_n": emb_n,
            },
            reuse_op=True, 
        )
        
        # Handle edge cases
        if not bm25_results and not emb_results:
            print(f"Warning: Both BM25 and embedding search returned no results for query: {query}")
            runtime_hybrid_results = comment_variable(
                [],
                to_runtime=True,
                variable_name="hybrid_retrieval_results",
                id_strategy=list_level_result_id_strategy,
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                category="hybrid_retrieval_results",
                class_name="hybrid_retrieval_results",
                comment=(
                    "The final hybrid retrieval results. "
                    "It is a list. Each element in the list is a retrieved memory unit. "
                    "The score field in each memory unit represents the hybrid retrieval score " 
                    "between the query and the retrieved memory unit."
                ),
            )
            comment_op(
                inputs=[runtime_bm25_results, runtime_emb_results],
                outputs=[runtime_hybrid_results],
                comment=(
                    "Hybrid retrieval returns no final memory units because both "
                    "the BM25 search and the embedding search return no candidates."
                ),
                reuse_op=True,
            )

            return []
        elif not bm25_results:
            print(f"Warning: BM25 search returned no results for query: {query}")

            if curret_is_active:
                for doc, score in emb_results[:k]:
                    comment_op(
                        inputs=[runtime_bm25_results, runtime_emb_results],
                        outputs=[
                            (
                                {
                                    "event_id": doc["event_id"],
                                    "timestamp": doc["timestamp"],
                                    "subject": doc["subject"],
                                    "summary": doc["summary"],
                                    "episode": doc["episode"],
                                    "event_log": {
                                        "atomic_fact": doc["event_log"]["atomic_fact"],
                                    } if doc["event_log"] is not None else None,
                                    "foresights": {
                                        "foresight": doc["foresights"]["foresight"],
                                        "evidence": doc["foresights"]["evidence"],
                                        "start_time": doc["foresights"]["start_time"],
                                        "end_time": doc["foresights"]["end_time"],
                                        "duration_days": doc["foresights"]["duration_days"],
                                    } if doc["foresights"] is not None else None,
                                },
                                {
                                    "id_strategy": "evermemos-dict",
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
                            "Because the BM25 search returns no candidates, hybrid "
                            "retrieval falls back to the embedding results and keeps "
                            f"memory '{doc['event_id']}' as a final retrieved memory "
                            f"unit. Its semantic similarity score is {score}."
                        ),
                        reuse_op=True,
                    )
            else:
                runtime_hybrid_results = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in emb_results[:k]
                    ],
                    to_runtime=True,
                    variable_name="hybrid_retrieval_results",
                    id_strategy=list_level_result_id_strategy,
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="hybrid_retrieval_results",
                    class_name="hybrid_retrieval_results",
                    comment=(
                        "The final hybrid retrieval results. "
                        "It is a list. Each element in the list is a retrieved memory unit. "
                        "The score field in each memory unit represents the hybrid retrieval score " 
                        "between the query and the retrieved memory unit."
                    ),
                )
                comment_op(
                    inputs=[runtime_bm25_results, runtime_emb_results],
                    outputs=[runtime_hybrid_results],
                    comment=(
                        "Because the BM25 search returns no candidates, hybrid "
                        "retrieval directly uses the embedding candidate list as "
                        "the final retrieval results."
                    ),
                    reuse_op=True,
                )

            return emb_results[:k]
        elif not emb_results:
            print(f"Warning: Embedding search returned no results for query: {query}")

            if curret_is_active:
                for doc, score in bm25_results[:k]:
                    comment_op(
                        inputs=[runtime_bm25_results, runtime_emb_results],
                        outputs=[
                            (
                                {
                                    "event_id": doc["event_id"],
                                    "timestamp": doc["timestamp"],
                                    "subject": doc["subject"],
                                    "summary": doc["summary"],
                                    "episode": doc["episode"],
                                    "event_log": {
                                        "atomic_fact": doc["event_log"]["atomic_fact"],
                                    } if doc["event_log"] is not None else None,
                                    "foresights": {
                                        "foresight": doc["foresights"]["foresight"],
                                        "evidence": doc["foresights"]["evidence"],
                                        "start_time": doc["foresights"]["start_time"],
                                        "end_time": doc["foresights"]["end_time"],
                                        "duration_days": doc["foresights"]["duration_days"],
                                    } if doc["foresights"] is not None else None,
                                },
                                {
                                    "id_strategy": "evermemos-dict",
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
                            "Because the embedding search returns no candidates, "
                            "hybrid retrieval falls back to the BM25 results and "
                            f"keeps memory '{doc['event_id']}' as a final retrieved "
                            f"memory unit. Its lexical similarity score is {score}."
                        ),
                        reuse_op=True,
                    )
            else:
                runtime_hybrid_results = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in bm25_results[:k]
                    ],
                    to_runtime=True,
                    variable_name="hybrid_retrieval_results",
                    id_strategy=list_level_result_id_strategy,
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="hybrid_retrieval_results",
                    class_name="hybrid_retrieval_results",
                    comment=(
                        "The final hybrid retrieval results. "
                        "It is a list. Each element in the list is a retrieved memory unit. "
                        "The score field in each memory unit represents the hybrid retrieval score " 
                        "between the query and the retrieved memory unit."
                    ),
                )
                comment_op(
                    inputs=[runtime_bm25_results, runtime_emb_results],
                    outputs=[runtime_hybrid_results],
                    comment=(
                        "Because the embedding search returns no candidates, hybrid "
                        "retrieval directly uses the BM25 candidate list as the "
                        "final retrieval results."
                    ),
                    reuse_op=True,
                )

            return bm25_results[:k]
        
        # RRF fusion
        fused = self._rrf_fusion(bm25_results, emb_results)

        if curret_is_active:
            for doc, score in fused[:k]:
                comment_op(
                    inputs=[runtime_bm25_results, runtime_emb_results],
                    outputs=[
                        (
                            {
                                "event_id": doc["event_id"],
                                "timestamp": doc["timestamp"],
                                "subject": doc["subject"],
                                "summary": doc["summary"],
                                "episode": doc["episode"],
                                "event_log": {
                                    "atomic_fact": doc["event_log"]["atomic_fact"],
                                } if doc["event_log"] is not None else None,
                                "foresights": {
                                    "foresight": doc["foresights"]["foresight"],
                                    "evidence": doc["foresights"]["evidence"],
                                    "start_time": doc["foresights"]["start_time"],
                                    "end_time": doc["foresights"]["end_time"],
                                    "duration_days": doc["foresights"]["duration_days"],
                                } if doc["foresights"] is not None else None,
                            },
                            {
                                "id_strategy": "evermemos-dict",
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
                        "Using constant "
                        f"{self.config.rrf_k} for Reciprocal Rank Fusion (RRF), "
                        "hybrid retrieval fuses the BM25 and embedding candidate "
                        f"lists and returns memory '{doc['event_id']}' with fused score {score}."
                    ),
                    metadata={
                        "rrf_k": self.config.rrf_k,
                    },
                    reuse_op=True,
                )
            tracing_ctx.remove_variable("active_query")
        else:
            runtime_hybrid_results = comment_variable(
                [
                    {
                        "event_id": doc["event_id"],
                        "timestamp": doc["timestamp"],
                        "subject": doc["subject"],
                        "summary": doc["summary"],
                        "episode": doc["episode"],
                        "event_log": {
                            "atomic_fact": doc["event_log"]["atomic_fact"],
                        } if doc["event_log"] is not None else None,
                        "foresights": {
                            "foresight": doc["foresights"]["foresight"],
                            "evidence": doc["foresights"]["evidence"],
                            "start_time": doc["foresights"]["start_time"],
                            "end_time": doc["foresights"]["end_time"],
                            "duration_days": doc["foresights"]["duration_days"],
                        } if doc["foresights"] is not None else None,
                        "score": score,
                    }
                    for doc, score in fused[:k]
                ],
                to_runtime=True,
                variable_name="hybrid_retrieval_results",
                id_strategy=list_level_result_id_strategy,
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                category="hybrid_retrieval_results",
                class_name="hybrid_retrieval_results",
                comment=(
                    "The final hybrid retrieval results are produced by Reciprocal "
                    f"Rank Fusion (RRF) with constant {self.config.rrf_k}. It is a "
                    "list. Each element in the list is a retrieved memory unit. The "
                    "score field in each memory unit represents the fused retrieval "
                    "score between the query and the retrieved memory unit."
                ),
                metadata={
                    "rrf_k": self.config.rrf_k,
                },
            )
            comment_op(
                inputs=[runtime_bm25_results, runtime_emb_results],
                outputs=[runtime_hybrid_results],
                comment=(
                    "Using constant "
                    f"{self.config.rrf_k} for Reciprocal Rank Fusion (RRF), "
                    "hybrid retrieval fuses the BM25 and embedding candidate "
                    "lists into the final retrieval results."
                ),
                metadata={
                    "rrf_k": self.config.rrf_k,
                },
                reuse_op=True,
            )
        
        print(f"Hybrid search: Emb={len(emb_results)}, BM25={len(bm25_results)}, Fused={len(fused)}, Returning top-{k}")
        return fused[:k]
    
    async def _search_agentic(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Agentic multi-round LLM-guided retrieval.
        
        Process:
        1. Round 1: Hybrid search -> Top N -> Rerank -> Sufficiency check
        2. If sufficient: return reranked results
        3. If insufficient:
           - Generate refined queries
           - Round 2: Retrieve and merge
           - Rerank combined results -> return final
        
        Parameters
        ----------
        query : str
            Search query.
        k : int
            Maximum number of results to return.
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Agentic Retrieval: {query[:60]}...")
        print(f"{'='*60}")
        print(f"  [Start] Time: {time.strftime('%H:%M:%S')}")
        
        # Determine retrieval sizes based on k
        # Round 1 retrieves k documents for candidate pool
        round1_size = k 
        sufficiency_check_size = min(self.config.sufficiency_check_docs, k)


        tracing_ctx = current_context()
        # Collect the variables that need to be cleaned up after the execution.
        vars_need_cleaned = [] 

        runtime_query = None
        if tracing_ctx is not None and is_tracing_enabled():
            # Get the original query from the tracing context.
            runtime_query = tracing_ctx.get_variable("query")
            # We mark it as the active query 
            # so that the execution graph produced by hybrid search changes accordingly.
            if tracing_ctx is not None and is_tracing_enabled():
                tracing_ctx.register_variable(
                    "active_query", 
                    runtime_query,
                    overwrite=True,
                )
            vars_need_cleaned.append("active_query")

        if self.config.use_reranker:
            round1_bundle_comment = (
                "It runs the first hybrid retrieval round, reranks the retrieved "
                "candidates to build the sufficiency-check set, and asks an LLM "
                "whether the current memory evidence is already sufficient or "
                "whether another retrieval round is needed."
            )
        else:
            round1_bundle_comment = (
                "It runs the first hybrid retrieval round, directly uses the top "
                "retrieved candidates to build the sufficiency-check set, and asks "
                "an LLM whether the current memory evidence is already sufficient "
                "or whether another retrieval round is needed."
            )
        with comment_op_scope(
            op_name="memory_system.agentic_round1_bundle",
            category="retrieval",
            comment=round1_bundle_comment,
            metadata={
                "round": 1,
                "retrieval_mode": "agentic",
                "top_k": k,
                "sufficiency_check_docs": sufficiency_check_size,
                "use_reranker": self.config.use_reranker,
            },
        ):
            # Round 1: Hybrid search
            print(f"  [Round 1] Hybrid search for Top {round1_size}...")
            round1_results = await self._search_hybrid(query, round1_size)

            runtime_round1_results = None
            if tracing_ctx is not None and is_tracing_enabled():
                runtime_round1_results = tracing_ctx.get_variable("hybrid_retrieval_results")
                vars_need_cleaned.append("hybrid_retrieval_results")

            if not round1_results:
                print(f"  [Warning] No results from Round 1")
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
                
                comment_link(
                    source=runtime_round1_results,
                    target=(
                        [],
                        {
                            "encoding_fn": partial(
                                json.dumps,
                                ensure_ascii=False,
                                indent=4,
                                sort_keys=True,
                            ),
                            "decoding_fn": json.loads,
                            "category": "agentic_retrieval_results",
                            "class_name": "agentic_retrieval_results",
                        }
                    ),
                    comment=(
                        "The first-round hybrid retrieval returns no candidate "
                        "memory units, so the final agentic retrieval results are "
                        "empty and the workflow stops before the sufficiency check."
                    ),
                )
                if tracing_ctx is not None and is_tracing_enabled():
                    for var in vars_need_cleaned:
                        tracing_ctx.remove_variable(var)
                return []
        
            print(f"  [Round 1] Retrieved {len(round1_results)} documents")
        
            # Rerank for sufficiency check
            round1_results_ = round1_results 
            if self.config.use_reranker:
                print(f"  [Rerank] Reranking to get Top {sufficiency_check_size} for sufficiency check...")
                round1_results_ = await self._rerank_results(query, round1_results, top_n=len(round1_results))
                reranked_for_check = round1_results_[:sufficiency_check_size]
                print(f"  [Rerank] Got {len(reranked_for_check)} documents for sufficiency check")

                # Note: Our identity strategy does not rely solely on the combination of event identifiers,
                # as identical combinations may still yield different scores. Therefore, we also incorporate
                # the score into the identity definition, so that such cases are treated as distinct results.
                runtime_round1_results_ = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in round1_results_
                    ],
                    to_runtime=True,
                    id_strategy=lambda results: ", ".join(
                        [f"{result['event_id']}-{result['score']}" for result in results]
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="reranked_round1_results",
                    class_name="reranked_round1_results",
                    comment=(
                        "The full Round-1 hybrid retrieval results after reranking. "
                        "It is a list. Each element in the list is a retrieved memory "
                        "unit. The score field in each memory unit represents the "
                        "reranking score assigned to the memory unit."
                    ),
                )
                comment_link(
                    source=runtime_round1_results,
                    target=runtime_round1_results_,
                    comment=(
                        "The Round-1 hybrid retrieval results are reranked so the full "
                        "candidate list is reordered by reranking score before the "
                        "sufficiency-check set is selected."
                    ),
                )
                runtime_round1_results = runtime_round1_results_

                reranked_for_check_comment = (
                    "The memory units selected for the Round-1 sufficiency check "
                    "after reranking. It is a list. Each element in the list is a "
                    "retrieved memory unit. The score field in each memory unit "
                    "represents the reranking score used to choose the "
                    "sufficiency-check set."
                )
                reranked_for_check_link_comment = (
                    "The Round-1 hybrid retrieval results are reranked, and the "
                    f"top-{len(reranked_for_check)} candidates are selected as the memory evidence for the "
                    "sufficiency check."
                )
            else:
                reranked_for_check = round1_results[:sufficiency_check_size]
                print(f"  [No Rerank] Using original Top {sufficiency_check_size} for sufficiency check")
                reranked_for_check_comment = (
                    "The memory units selected for the Round-1 sufficiency check. "
                    "It is a list. Each element in the list is a retrieved memory "
                    "unit. The score field in each memory unit represents the "
                    "hybrid retrieval score used to choose the sufficiency-check set."
                )
                reranked_for_check_link_comment = (
                    "The Round-1 hybrid retrieval results are directly truncated to "
                    f"the top-{len(reranked_for_check)} candidates and used as the memory evidence for the "
                    "sufficiency check because reranking is disabled."
                )

            runtime_reranked_for_check = comment_variable(
                [
                    {
                        "event_id": doc["event_id"],
                        "timestamp": doc["timestamp"],
                        "subject": doc["subject"],
                        "summary": doc["summary"],
                        "episode": doc["episode"],
                        "event_log": {
                            "atomic_fact": doc["event_log"]["atomic_fact"],
                        } if doc["event_log"] is not None else None,
                        "foresights": {
                            "foresight": doc["foresights"]["foresight"],
                            "evidence": doc["foresights"]["evidence"],
                            "start_time": doc["foresights"]["start_time"],
                            "end_time": doc["foresights"]["end_time"],
                            "duration_days": doc["foresights"]["duration_days"],
                        } if doc["foresights"] is not None else None,
                        "score": score,
                    }
                    for doc, score in reranked_for_check
                ],
                to_runtime=True,
                id_strategy=lambda results: ", ".join( 
                    [f"{result['event_id']}-{result['score']}" for result in results]
                ),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                category="reranked_for_check",
                class_name="reranked_for_check",
                comment=reranked_for_check_comment,
            )
            comment_link(
                source=runtime_round1_results,
                target=runtime_reranked_for_check,
                comment=reranked_for_check_link_comment,
            )
            
            if not reranked_for_check:
                print(f"  [Warning] Reranking failed, falling back to original results")
                print(f"Time taken: {time.time() - start_time:.2f} seconds")

                for doc, score in round1_results[:k]:
                    comment_op(
                        inputs=[runtime_round1_results, runtime_reranked_for_check],
                        outputs=[
                            (
                                {
                                    "event_id": doc["event_id"],
                                    "timestamp": doc["timestamp"],
                                    "subject": doc["subject"],
                                    "summary": doc["summary"],
                                    "episode": doc["episode"],
                                    "event_log": {
                                        "atomic_fact": doc["event_log"]["atomic_fact"],
                                    } if doc["event_log"] is not None else None,
                                    "foresights": {
                                        "foresight": doc["foresights"]["foresight"],
                                        "evidence": doc["foresights"]["evidence"],
                                        "start_time": doc["foresights"]["start_time"],
                                        "end_time": doc["foresights"]["end_time"],
                                        "duration_days": doc["foresights"]["duration_days"],
                                    } if doc["foresights"] is not None else None,
                                },
                                {
                                    "id_strategy": "evermemos-dict",
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
                            "Because the Round-1 sufficiency-check set is empty, "
                            "agentic retrieval falls back to the original Round-1 "
                            f"hybrid results and keeps memory '{doc['event_id']}' "
                            "as a final retrieved memory unit. Its hybrid retrieval "
                            f"score is {score}."
                        ),
                        reuse_op=True,
                    )
                if tracing_ctx is not None and is_tracing_enabled():
                    for var in vars_need_cleaned:
                        tracing_ctx.remove_variable(var)

                return round1_results[:k]

            # The corresponding runtime variables have been assigned in the same manner.
            round1_results = round1_results_

            # Put the documents for sufficiency check in the tracing context.
            if tracing_ctx is not None and is_tracing_enabled():
                tracing_ctx.register_variable(
                    "reranked_for_check", 
                    runtime_reranked_for_check,
                    overwrite=True,
                )
    
        
            # Sufficiency check
            print(f"  [LLM] Checking sufficiency on Top {len(reranked_for_check)}...")
            is_sufficient, reasoning, missing_info, key_info = await self._check_sufficiency(
                query, reranked_for_check
            )
        
            print(f"  [LLM] Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
            print(f"  [LLM] Reasoning: {reasoning}")
            if key_info:
                print(f"  [LLM] Key Info Found: {', '.join(key_info)}")

            runtime_sufficiency_check_result = None 
            if tracing_ctx is not None and is_tracing_enabled():
                runtime_sufficiency_check_result = tracing_ctx.get_variable("sufficiency_check_result")
                vars_need_cleaned.append("sufficiency_check_result")
            
            if is_sufficient:
                # If sufficient, rerank full round1 results to get final k documents
                final_results = round1_results[:k]

                for doc, score in final_results:
                    comment_op(
                        inputs=[
                            runtime_sufficiency_check_result,
                            runtime_round1_results,
                        ],
                        outputs=[
                            (
                                {
                                    "event_id": doc["event_id"],
                                    "timestamp": doc["timestamp"],
                                    "subject": doc["subject"],
                                    "summary": doc["summary"],
                                    "episode": doc["episode"],
                                    "event_log": {
                                        "atomic_fact": doc["event_log"]["atomic_fact"],
                                    } if doc["event_log"] is not None else None,
                                    "foresights": {
                                        "foresight": doc["foresights"]["foresight"],
                                        "evidence": doc["foresights"]["evidence"],
                                        "start_time": doc["foresights"]["start_time"],
                                        "end_time": doc["foresights"]["end_time"],
                                        "duration_days": doc["foresights"]["duration_days"],
                                    } if doc["foresights"] is not None else None,
                                },
                                {
                                    "id_strategy": "evermemos-dict",
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
                            "Because the Round-1 sufficiency check concludes that "
                            "the current memory evidence is already sufficient, "
                            "agentic retrieval stops after Round 1 and keeps "
                            f"memory '{doc['event_id']}' as a final retrieved "
                            f"memory unit. Its Round-1 retrieval score is {score}."
                        ),
                        reuse_op=True,
                    )
                
                print(f"  [Complete] Sufficient! Final: {len(final_results)} docs")
                print(f"Time taken: {time.time() - start_time:.2f} seconds")

                if tracing_ctx is not None and is_tracing_enabled():
                    for var in vars_need_cleaned:
                        tracing_ctx.remove_variable(var)

                return final_results
        
        if self.config.use_multi_query:
            round2_bundle_comment = (
                "The first round is insufficient. It generates multiple "
                "complementary follow-up queries, executes the second retrieval "
                "round for each query, and fuses their candidate memories."
            )
        else:
            round2_bundle_comment = (
                "The first round is insufficient. It generates a single "
                "refined follow-up query and executes the second retrieval round "
                "with that query."
            )

        # Initialize the second retrieval round. 
        with comment_op_scope(
            op_name="memory_system.agentic_round2_bundle",
            category="retrieval",
            comment=round2_bundle_comment,
            metadata={
                "round": 2,
                "retrieval_mode": "agentic",
                "top_k": k,
                "multi_query": self.config.use_multi_query,
            },
        ):
            # Round 2: Generate refined queries and search
            print(f"  [Round 2] Insufficient, generating refined queries...")
            print(f"  [Missing] {', '.join(missing_info) if missing_info else 'N/A'}")
            
            if self.config.use_multi_query:
                refined_queries, _ = await self._generate_multi_queries(
                    query, reranked_for_check, missing_info, key_info
                )
                print(f"  [Round 2] Generated {len(refined_queries)} queries")
            else:
                refined_query = await self._generate_refined_query(
                    query, reranked_for_check, missing_info
                )
                refined_queries = [refined_query]
                print(f"  [Round 2] Generated refined query: {refined_query}...")
            
            runtime_refined_queries = None 
            all_runtime_round2_results = [] 
            if tracing_ctx is not None and is_tracing_enabled():
                runtime_refined_queries = tracing_ctx.get_variable("refined_queries")
                vars_need_cleaned.append("refined_queries")
            
            # Execute refined queries
            # Each query retrieves k candidates
            round2_retrieval_size = k
            all_round2_results = []
            for i, rq in enumerate(refined_queries, 1):
                print(f"  [Round 2] Searching query {i}: {rq}...")

                # We overwrite the current active query in the tracing context 
                # with the refined query. 
                # The hybrid search will track the refined query.
                runtime_refined_query = comment_variable(
                    rq,
                    to_runtime=True,
                    variable_name="active_query",
                    category="refined_query",
                    class_name="refined_query",
                    comment=(
                        f"The {i}-th refined follow-up query from a list of refined queries."
                    ), 
                )
                comment_link(
                    source=runtime_refined_queries,
                    target=runtime_refined_query,
                    comment=(
                        f"Currently, the memory system selects the {i}-th query from the generated "
                        "Round-2 follow-up query list."
                    ),
                )

                r2_results = await self._search_hybrid(rq, round2_retrieval_size)

                if tracing_ctx is not None and is_tracing_enabled():
                    runtime_round2_results = tracing_ctx.get_variable("hybrid_retrieval_results")
                    all_runtime_round2_results.append(runtime_round2_results)

                all_round2_results.append(r2_results)
                print(f"    Query {i}: Retrieved {len(r2_results)} documents")
            
            # Multi-query RRF fusion
            print(f"  [Multi-RRF] Fusing results from {len(refined_queries)} queries...")
            runtime_round2_results = None
            if len(all_round2_results) > 1:
                round2_results = self._multi_rrf_fusion(all_round2_results)
                
                runtime_round2_results = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in round2_results
                    ],
                    to_runtime=True,
                    id_strategy=lambda results: ", ".join(
                        [f"{result['event_id']}-{result['score']}" for result in results]
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="round2_retrieval_results",
                    class_name="round2_retrieval_results",
                    comment=(
                        "The final Round-2 retrieval results."
                    ),
                )
                for runtime_branch_round2_results in all_runtime_round2_results:
                    comment_link(
                        source=runtime_branch_round2_results,
                        target=runtime_round2_results,
                        comment=(
                            "One Round-2 candidate memory list contributes to "
                            "the final Round-2 retrieval results through "
                            "multi-query Reciprocal Rank Fusion (RRF). "
                            f"The constant k for RRF is set to {self.config.rrf_k}."
                        ),
                    )
            elif all_round2_results:
                round2_results = all_round2_results[0]

                runtime_round2_results = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in round2_results
                    ],
                    to_runtime=True,
                    id_strategy=lambda results: ", ".join(
                        [f"{result['event_id']}-{result['score']}" for result in results]
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="round2_retrieval_results",
                    class_name="round2_retrieval_results",
                    comment=(
                        "The final Round-2 retrieval results."
                    ),
                )
                comment_link(
                    source=all_runtime_round2_results[0],
                    target=runtime_round2_results,
                    comment=(
                        "Because there is only one Round-2 retrieval branch, "
                        "its hybrid retrieval results are directly used as "
                        "the final Round-2 retrieval results."
                    ),
                )
            else:
                round2_results = []

                runtime_round2_results = comment_variable(
                    [],
                    to_runtime=True,
                    id_strategy=lambda results: ", ".join(
                        [f"{result['event_id']}-{result['score']}" for result in results]
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="round2_retrieval_results",
                    class_name="round2_retrieval_results",
                    comment=(
                        "The final Round-2 retrieval results."
                    ),
                )
                comment_link(
                    source=runtime_refined_queries,
                    target=runtime_round2_results,
                    comment=(
                        "The generated Round-2 follow-up query list does "
                        "not yield any usable second-round candidate memory "
                        "list, so the final Round-2 retrieval results are "
                        "empty."
                    ),
                )
        
        if self.config.use_reranker:
            final_rerank_comment = (
                "It merges the first- and second-round candidate pools, "
                "removes duplicate memories, and reranks the combined "
                "candidates to select the final top-k agentic retrieval "
                "results."
            )
        else:
            final_rerank_comment = (
                "It merges the first- and second-round candidate pools, "
                "removes duplicate memories, and directly uses the top merged "
                "candidates as the final top-k agentic retrieval results "
                "because reranking is disabled."
            )

        # Initialize the third operation context. 
        with comment_op_scope(
            op_name="memory_system.agentic_final_rerank",
            category="retrieval",
            comment=final_rerank_comment,
            metadata={
                "top_k": k,
                "use_reranker": self.config.use_reranker,
            },
        ):
            # Merge Round 1 and Round 2 results
            # Target merge size: 2x k for reranking
            merge_target = k * 2
            
            print(f"  [Merge] Combining Round 1 and Round 2 to {merge_target} documents...")
            round1_ids = {doc.get("event_id", id(doc)) for doc, _ in round1_results}
            round2_unique = [
                (doc, score) for doc, score in round2_results
                if doc.get("event_id", id(doc)) not in round1_ids
            ]
            
            combined = round1_results.copy()
            needed_from_round2 = merge_target - len(combined)
            combined.extend(round2_unique[:needed_from_round2])
            
            duplicates_removed = len(round2_results) - len(round2_unique)
            round2_added = len(round2_unique[:needed_from_round2])
            
            print(
                f"  [Merge] Round1: {len(round1_results)}, Round2 unique added: {round2_added}, duplicates removed: {duplicates_removed}"
            )
            print(f"  [Merge] Combined total: {len(combined)} documents")

            runtime_combined = comment_variable(
                [
                    {
                        "event_id": doc["event_id"],
                        "timestamp": doc["timestamp"],
                        "subject": doc["subject"],
                        "summary": doc["summary"],
                        "episode": doc["episode"],
                        "event_log": {
                            "atomic_fact": doc["event_log"]["atomic_fact"],
                        } if doc["event_log"] is not None else None,
                        "foresights": {
                            "foresight": doc["foresights"]["foresight"],
                            "evidence": doc["foresights"]["evidence"],
                            "start_time": doc["foresights"]["start_time"],
                            "end_time": doc["foresights"]["end_time"],
                            "duration_days": doc["foresights"]["duration_days"],
                        } if doc["foresights"] is not None else None,
                        "score": score,
                    }
                    for doc, score in combined
                ],
                to_runtime=True,
                id_strategy=lambda results: ", ".join(
                    [f"{result['event_id']}-{result['score']}" for result in results]
                ),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                category="combined_retrieval_results",
                class_name="combined_retrieval_results",
                comment=(
                    "The merged candidate pool before final selection."
                ),
            )
            comment_op(
                inputs=[runtime_round1_results, runtime_round2_results],
                outputs=[runtime_combined],
                comment=(
                    "The merged candidate pool starts with all Round-1 retrieval "
                    "results, then scans the final Round-2 retrieval results, "
                    "discards any memory who already appears in Round 1, "
                    f"and appends the first {round2_added} remaining unique Round-2 "
                    f"memories toward a target merged size of {merge_target}. In this "
                    f"merge, {duplicates_removed} duplicate Round-2 memories are "
                    f"removed and the final combined pool contains {len(combined)} "
                    "memories."
                ),
                reuse_op=True,
            )
            
            # Final rerank
            runtime_final_results = None
            if self.config.use_reranker and len(combined) > 0:
                print(f"  [Rerank] Reranking {len(combined)} combined documents to get Top {k}...")
                final_results = await self._rerank_results(query, combined, top_n=k)
                print(f"  [Rerank] Final Top {len(final_results)} selected")

                runtime_final_results = comment_variable(
                    [
                        {
                            "event_id": doc["event_id"],
                            "timestamp": doc["timestamp"],
                            "subject": doc["subject"],
                            "summary": doc["summary"],
                            "episode": doc["episode"],
                            "event_log": {
                                "atomic_fact": doc["event_log"]["atomic_fact"],
                            } if doc["event_log"] is not None else None,
                            "foresights": {
                                "foresight": doc["foresights"]["foresight"],
                                "evidence": doc["foresights"]["evidence"],
                                "start_time": doc["foresights"]["start_time"],
                                "end_time": doc["foresights"]["end_time"],
                                "duration_days": doc["foresights"]["duration_days"],
                            } if doc["foresights"] is not None else None,
                            "score": score,
                        }
                        for doc, score in final_results
                    ],
                    to_runtime=True,
                    id_strategy=lambda results: ", ".join(
                        [f"{result['event_id']}-{result['score']}" for result in results]
                    ),
                    encoding_fn=partial(
                        json.dumps,
                        ensure_ascii=False,
                        indent=4,
                        sort_keys=True,
                    ),
                    decoding_fn=json.loads,
                    category="final_reranked_results",
                    class_name="final_reranked_results",
                    comment=(
                        "The final agentic retrieval results after reranking. "
                        "It is a list. Each element in the list is a retrieved "
                        "memory unit. The score field in each memory unit "
                        "represents the reranking score assigned to the memory "
                        "unit."
                    ),
                )
                comment_link(
                    source=runtime_combined,
                    target=runtime_final_results,
                    comment=(
                        "The merged candidate pool is reranked, and the "
                        f"top-{len(final_results)} candidates are selected as the "
                        "final agentic retrieval results."
                    ),
                )
            else:
                final_results = combined[:k]
                print(f"  [No Rerank] Returning Top {k} from combined results")

            for doc, score in final_results:
                if runtime_final_results is not None:
                    final_result_comment = (
                        "The final reranked retrieval results select memory "
                        f"'{doc['event_id']}' as a final retrieved memory unit. "
                        f"Its reranking score is {score}."
                    )
                    final_result_inputs = [runtime_final_results]
                else:
                    final_result_comment = (
                        "Because reranking is disabled, the merged candidate pool "
                        f"is directly truncated to the top-{k} candidates and keeps "
                        f"memory '{doc['event_id']}' as a final retrieved memory "
                        f"unit. Its score in the merged candidate pool is {score}."
                    )
                    final_result_inputs = [runtime_combined]
                comment_op(
                    inputs=final_result_inputs,
                    outputs=[
                        (
                            {
                                "event_id": doc["event_id"],
                                "timestamp": doc["timestamp"],
                                "subject": doc["subject"],
                                "summary": doc["summary"],
                                "episode": doc["episode"],
                                "event_log": {
                                    "atomic_fact": doc["event_log"]["atomic_fact"],
                                } if doc["event_log"] is not None else None,
                                "foresights": {
                                    "foresight": doc["foresights"]["foresight"],
                                    "evidence": doc["foresights"]["evidence"],
                                    "start_time": doc["foresights"]["start_time"],
                                    "end_time": doc["foresights"]["end_time"],
                                    "duration_days": doc["foresights"]["duration_days"],
                                } if doc["foresights"] is not None else None,
                            },
                            {
                                "id_strategy": "evermemos-dict",
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
                    comment=final_result_comment,
                    reuse_op=True,
                )
            
            print(f"  [Complete] Final: {len(final_results)} docs")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            print(f"{'='*60}\n")
        
        return final_results
    
    async def _check_sufficiency(
        self,
        query: str,
        results: List[Tuple[Dict[str, Any], float]],
    ) -> Tuple[bool, str, List[str], List[str]]:
        """Check if retrieval results are sufficient.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L172. 
        """
        runtime_reranked_for_check = runtime_query = runtime_prompt = None
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_reranked_for_check = tracing_ctx.get_variable("reranked_for_check")
            runtime_query = tracing_ctx.get_variable("active_query")

        try:
            # Format documents
            docs_text = self._format_documents(results)
            runtime_docs_text = comment_variable(
                docs_text,
                to_runtime=True,
                class_name="sufficiency_check_docs_text",
                category="formatted_documents",
                comment=(
                    "The Round-1 sufficiency-check memory evidence formatted into "
                    "a plain-text document block for the large language model."
                ),
            )
            comment_link(
                source=runtime_reranked_for_check,
                target=runtime_docs_text,
                comment=(
                    "The memory system formats the reranked sufficiency-check "
                    "memory evidence into plain text before sending it to the "
                    "large language model."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(self._format_documents),
                },
            )

            runtime_prompt_template = comment_variable(
                SUFFICIENCY_CHECK_PROMPT,
                to_runtime=True,
                id_strategy=lambda _: "sufficiency-check-prompt-template",
                class_name="sufficiency_check_prompt_template",
                category="prompt",
                comment=(
                    "A prompt template that asks the large language model to "
                    "judge whether the current Round-1 memory evidence is "
                    "already sufficient to answer the task query."
                ),
            )
            
            prompt = SUFFICIENCY_CHECK_PROMPT.format(
                query=query,
                retrieved_docs=docs_text,
            )
            
            runtime_prompt = comment_variable(
                prompt,
                to_runtime=True,
                class_name="sufficiency_check_prompt",
                category="prompt",
                comment=(
                    "A concrete sufficiency-check prompt assembled from the task "
                    "query, the formatted memory evidence, and the sufficiency-"
                    "check prompt template."
                ),
            )
            comment_op(
                inputs=[
                    runtime_query,
                    runtime_docs_text,
                    runtime_prompt_template,
                ],
                outputs=[runtime_prompt],
                comment=(
                    "The memory system assembles the concrete sufficiency-check "
                    "prompt from the task query, the formatted memory evidence, "
                    "and the sufficiency-check prompt template."
                ),
                reuse_op=True,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=500,
            )
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            runtime_response = comment_variable(
                response,
                to_runtime=True,
                class_name="raw_sufficiency_check_response",
                category="llm_response",
                comment=(
                    "A raw sufficiency-check response from the large language "
                    "model."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_response,
                comment=(
                    "The large language model judges whether the current memory "
                    "evidence is sufficient for answering the task query."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(self._parse_json_response),
                },
            )

            runtime_sufficiency_check_result = comment_variable(
                {
                    "is_sufficient": result["is_sufficient"],
                    "reasoning": result["reasoning"],
                    "missing_information": result.get("missing_information", []),
                    "key_information_found": result.get("key_information_found", []),
                },
                to_runtime=True,
                variable_name="sufficiency_check_result",
                class_name="sufficiency_check_result",
                category="sufficiency_check_result",
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                comment=(
                    "A structured sufficiency-check result parsed from the "
                    "large language model's response."
                ),
            )
            comment_link(
                source=runtime_response,
                target=runtime_sufficiency_check_result,
                comment=(
                    "The memory system parses the raw sufficiency-check response "
                    "into a structured sufficiency-check result."
                ),
            )
            
            return (
                result["is_sufficient"],
                result["reasoning"],
                result.get("missing_information", []),
                result.get("key_information_found", [])
            )

        except asyncio.TimeoutError:
            print(f"  ❌ Sufficiency check timeout (30s)")
            
            runtime_timeout_result = comment_variable(
                {
                    "is_sufficient": True,
                    "reasoning": "Timeout: LLM took too long",
                    "missing_information": [],
                    "key_information_found": [],
                },
                to_runtime=True,
                variable_name="sufficiency_check_result",
                class_name="sufficiency_check_result",
                category="sufficiency_check_result",
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                comment=(
                    "A fallback sufficiency-check result."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_timeout_result,
                comment=(
                    "The memory system returns a fallback sufficiency-check "
                    "result because the large language model times out."
                ),
                edge_metadata={
                    "error": "Sufficiency check timeout (30s).",
                },
            )

            # Timeout fallback: assume sufficient
            return True, "Timeout: LLM took too long", [], []
        except Exception as e:
            print(f"  ❌ Sufficiency check failed: {e}")
            import traceback
            traceback.print_exc()

            runtime_error_result = comment_variable(
                {
                    "is_sufficient": True,
                    "reasoning": f"Error: {str(e)}",
                    "missing_information": [],
                    "key_information_found": [],
                },
                to_runtime=True,
                variable_name="sufficiency_check_result",
                class_name="sufficiency_check_result",
                category="sufficiency_check_result",
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                comment=(
                    "A fallback sufficiency-check result."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_error_result,
                comment=(
                    "The memory system returns a fallback sufficiency-check "
                    "result because the sufficiency-check workflow raises an "
                    "exception."
                ),
                edge_metadata={
                    "error": traceback.format_exc(),
                },
            )

            # Conservative fallback: assume sufficient
            return True, f"Error: {str(e)}", [], []

    async def _generate_refined_query(
        self,
        original_query: str,
        results: List[Tuple[Dict[str, Any], float]],
        missing_info: List[str],
    ) -> str:
        """Generate a refined query using self.llm_provider."""
        runtime_query = runtime_sufficiency_check_result = None
        runtime_reranked_for_check = runtime_prompt = None
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_query = tracing_ctx.get_variable("active_query")
            runtime_sufficiency_check_result = tracing_ctx.get_variable(
                "sufficiency_check_result"
            )
            runtime_reranked_for_check = tracing_ctx.get_variable("reranked_for_check")

        try:
            docs_text = self._format_documents(results)
            runtime_docs_text = comment_variable(
                docs_text,
                to_runtime=True,
                class_name="refined_query_docs_text",
                category="formatted_documents",
                comment=(
                    "The insufficiency evidence formatted into a plain-text "
                    "document block for refined-query generation."
                ),
            )
            comment_link(
                source=runtime_reranked_for_check,
                target=runtime_docs_text,
                comment=(
                    "The memory system formats the Round-1 insufficiency "
                    "evidence into plain text before generating a refined "
                    "follow-up query."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(self._format_documents),
                },
            )

            runtime_prompt_template = comment_variable(
                REFINED_QUERY_PROMPT,
                to_runtime=True,
                id_strategy=lambda _: "refined-query-prompt-template",
                class_name="refined_query_prompt_template",
                category="prompt",
                comment=(
                    "A prompt template that asks the large language model to "
                    "generate a single refined follow-up query when the first "
                    "retrieval round is insufficient."
                ),
            )
            missing_str = ", ".join(missing_info) if missing_info else "N/A"
            
            prompt = REFINED_QUERY_PROMPT.format(
                original_query=original_query,
                retrieved_docs=docs_text,
                missing_info=missing_str,
            )
            
            runtime_prompt = comment_variable(
                prompt,
                to_runtime=True,
                class_name="refined_query_prompt",
                category="prompt",
                comment=(
                    "A concrete refined-query prompt assembled from the "
                    "original task query, the formatted insufficiency "
                    "evidence, the refined-query prompt template, and the "
                    "missing information field in the structured "
                    "sufficiency-check result."
                ),
            )
            comment_op(
                inputs=[
                    runtime_query,
                    runtime_sufficiency_check_result,
                    runtime_docs_text,
                    runtime_prompt_template,
                ],
                outputs=[runtime_prompt],
                comment=(
                    "The memory system assembles the concrete refined-query "
                    "prompt from the original task query, the formatted "
                    "Round-1 insufficiency evidence, and the "
                    "missing information field from the structured "
                    "sufficiency-check result."
                ),
                reuse_op=True,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150,
            )
            
            refined_query = self._parse_refined_query(response, original_query)
            
            runtime_response = comment_variable(
                response,
                to_runtime=True,
                class_name="raw_refined_query_response",
                category="llm_response",
                comment=(
                    "A raw refined-query-generation response from the large "
                    "language model."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_response,
                comment=(
                    "The large language model generates a refined follow-up "
                    "query based on the concrete refined-query prompt."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(self._parse_refined_query),
                },
            )

            runtime_refined_queries = comment_variable(
                [refined_query],
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda queries: " | ".join(queries),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A one-element list containing the refined follow-up query "
                    "for the second retrieval round."
                ),
            )
            comment_link(
                source=runtime_response,
                target=runtime_refined_queries,
                comment=(
                    "The memory system parses the raw refined-query response "
                    "and wraps the resulting query as a one-element query list "
                    "for the second retrieval round."
                ),
            )
            
            return refined_query
        
        except asyncio.TimeoutError:
            print(f"  ❌ Query refinement timeout (30s)")
            runtime_timeout_queries = comment_variable(
                [original_query],
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda queries: " | ".join(queries),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A fallback one-element query list containing the original "
                    "task query."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_timeout_queries,
                comment=(
                    "The memory system falls back to the original task "
                    "query because refined-query generation times out."
                ),
                edge_metadata={
                    "error": "Query refinement timeout (30s).",
                },
            )

            # Timeout fallback: use original query
            return original_query
        except Exception as e:
            print(f"  ❌ Query refinement failed: {e}")
            import traceback
            traceback.print_exc()

            runtime_error_queries = comment_variable(
                [original_query],
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda queries: " | ".join(queries),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A fallback one-element query list containing the original "
                    "task query."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_error_queries,
                comment=(
                    "The memory system falls back to the original task "
                    "query because refined-query generation raises an "
                    "exception."
                ),
                edge_metadata={
                    "error": traceback.format_exc(),
                },
            )

            # Fall back to original query
            return original_query
    
    async def _generate_multi_queries(
        self,
        original_query: str,
        results: List[Tuple[Dict[str, Any], float]],
        missing_info: List[str],
        key_info: List[str],
    ) -> Tuple[List[str], str]:
        """Generate multiple complementary queries (2-3 queries as per prompt).
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L355. 
        """
        runtime_query = runtime_sufficiency_check_result = None
        runtime_reranked_for_check = runtime_prompt = None
        tracing_ctx = current_context()
        if tracing_ctx is not None and is_tracing_enabled():
            runtime_query = tracing_ctx.get_variable("active_query")
            runtime_sufficiency_check_result = tracing_ctx.get_variable(
                "sufficiency_check_result"
            )
            runtime_reranked_for_check = tracing_ctx.get_variable("reranked_for_check")

        try:
            docs_text = self._format_documents(results)
            runtime_docs_text = comment_variable(
                docs_text,
                to_runtime=True,
                class_name="multi_query_docs_text",
                category="formatted_documents",
                comment=(
                    "The insufficiency evidence formatted into a plain-text "
                    "document block for multi-query generation."
                ),
            )
            comment_link(
                source=runtime_reranked_for_check,
                target=runtime_docs_text,
                comment=(
                    "The memory system formats the Round-1 insufficiency "
                    "evidence into plain text before generating multiple "
                    "follow-up queries."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(self._format_documents),
                },
            )

            runtime_prompt_template = comment_variable(
                MULTI_QUERY_GENERATION_PROMPT,
                to_runtime=True,
                id_strategy=lambda _: "multi-query-generation-prompt-template",
                class_name="multi_query_generation_prompt_template",
                category="prompt",
                comment=(
                    "A prompt template that asks the large language model to "
                    "generate multiple complementary follow-up queries when "
                    "the first retrieval round is insufficient."
                ),
            )
            missing_str = ", ".join(missing_info) if missing_info else "N/A"
            key_str = ", ".join(key_info) if key_info else "N/A"
            
            prompt = MULTI_QUERY_GENERATION_PROMPT.format(
                original_query=original_query,
                retrieved_docs=docs_text,
                missing_info=missing_str,
                key_info=key_str,
            )
            
            runtime_prompt = comment_variable(
                prompt,
                to_runtime=True,
                class_name="multi_query_generation_prompt",
                category="prompt",
                comment=(
                    "A concrete multi-query-generation prompt assembled from "
                    "the original task query, the formatted insufficiency "
                    "evidence, the multi-query prompt template, and the "
                    "missing information field plus key information field "
                    "fields in the structured sufficiency-check result."
                ),
            )
            comment_op(
                inputs=[
                    runtime_query,
                    runtime_sufficiency_check_result,
                    runtime_docs_text,
                    runtime_prompt_template,
                ],
                outputs=[runtime_prompt],
                comment=(
                    "The memory system assembles the concrete multi-query "
                    "generation prompt from the original task query, the "
                    "formatted Round-1 insufficiency evidence, and the "
                    "missing information field plus key information field "
                    "fields from the structured sufficiency-check result."
                ),
                reuse_op=True,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=300,
            )
            
            queries, reasoning = self._parse_multi_query_response(response, original_query)
            runtime_response = comment_variable(
                response,
                to_runtime=True,
                class_name="raw_multi_query_generation_response",
                category="llm_response",
                comment=(
                    "A raw multi-query-generation response from the large "
                    "language model."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_response,
                comment=(
                    "The large language model generates multiple "
                    "complementary follow-up queries based on the concrete "
                    "multi-query-generation prompt."
                ),
                edge_metadata={
                    "source_code": inspect.getsource(
                        self._parse_multi_query_response
                    ),
                },
            )

            runtime_refined_queries = comment_variable(
                queries,
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda generated_queries: " | ".join(generated_queries),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A list of complementary follow-up queries for the second "
                    "retrieval round."
                ),
            )
            comment_link(
                source=runtime_response,
                target=runtime_refined_queries,
                comment=(
                    "The memory system parses the raw multi-query response "
                    "into a list of complementary follow-up queries for the "
                    "second retrieval round."
                ),
            )
            
            print(f"  [Multi-Query] Generated {len(queries)} queries:")
            for i, q in enumerate(queries, 1):
                print(f"    Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
            print(f"  [Multi-Query] Strategy: {reasoning}")
            
            return queries, reasoning
        
        except asyncio.TimeoutError:
            print(f"  ❌ Multi-query generation timeout (30s)")
            runtime_timeout_queries = comment_variable(
                [original_query],
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda generated_queries: " | ".join(
                    generated_queries
                ),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A fallback one-element query list containing the original "
                    "task query."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_timeout_queries,
                comment=(
                    "The memory system falls back to the original task "
                    "query because multi-query generation times out."
                ),
                edge_metadata={
                    "error": "Multi-query generation timeout (30s).",
                },
            )

            return [original_query], "Timeout: used original query"
        except Exception as e:
            print(f"  ❌ Multi-query generation failed: {e}")
            import traceback
            traceback.print_exc()

            runtime_error_queries = comment_variable(
                [original_query],
                to_runtime=True,
                variable_name="refined_queries",
                id_strategy=lambda generated_queries: " | ".join(
                    generated_queries
                ),
                encoding_fn=partial(
                    json.dumps,
                    ensure_ascii=False,
                    indent=4,
                    sort_keys=True,
                ),
                decoding_fn=json.loads,
                class_name="refined_queries",
                category="refined_queries",
                comment=(
                    "A fallback one-element query list containing the original "
                    "task query."
                ),
            )
            comment_link(
                source=runtime_prompt,
                target=runtime_error_queries,
                comment=(
                    "The memory system falls back to the original task "
                    "query because multi-query generation raises an "
                    "exception."
                ),
                edge_metadata={
                    "error": traceback.format_exc(),
                },
            )

            # Fall back to original query
            return [original_query], f"Error: {str(e)}"
    
    def _format_documents(
        self,
        results: List[Tuple[Dict[str, Any], float]],
    ) -> str:
        """Format documents for LLM consumption.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L21. 
        """
        formatted = []
        
        for i, (doc, _) in enumerate(results, start=1):
            subject = doc.get("subject", "N/A")
            episode = doc.get("episode", "N/A")
            
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            formatted.append(
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
        
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L95. 
        """
        try:
            # Extract JSON (LLM may add extra text before/after)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "is_sufficient" not in result:
                raise ValueError("Missing 'is_sufficient' field")
            
            # Set default values
            result.setdefault("reasoning", "No reasoning provided")
            result.setdefault("missing_information", [])
            result.setdefault("key_information_found", [])
            
            return result
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Failed to parse LLM response: {e}")
            print(f"  Raw response: {response[:200]}...")
            
            # Conservative fallback: assume sufficient to avoid unnecessary second round
            return {
                "is_sufficient": True,
                "reasoning": f"Failed to parse: {str(e)}",
                "missing_information": [],
                "key_information_found": []
            }
    
    def _parse_multi_query_response(self, response: str, original_query: str) -> Tuple[List[str], str]:
        """Parse multi-query generation JSON response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L299. 
        """
        try:
            # Extract JSON
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "queries" not in result or not isinstance(result["queries"], list):
                raise ValueError("Missing or invalid 'queries' field")
            
            queries = result["queries"]
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Filter and validate queries
            valid_queries = []
            for q in queries:
                if isinstance(q, str) and 5 <= len(q) <= 300:
                    # Avoid identical to original query
                    if q.lower().strip() != original_query.lower().strip():
                        valid_queries.append(q.strip())
            
            # Return at least 1 query
            if not valid_queries:
                print(f"  ⚠️  No valid queries generated, using original")
                return [original_query], "Fallback: used original query"
            
            # Limit to maximum 3 queries
            valid_queries = valid_queries[:3]
            
            print(f"  ✅ Generated {len(valid_queries)} valid queries")
            return valid_queries, reasoning
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Failed to parse multi-query response: {e}")
            print(f"  Raw response: {response[:200]}...")
            
            # Fallback: return original query
            return [original_query], f"Parse error: {str(e)}"

    def _parse_refined_query(self, response: str, original_query: str) -> str:
        """
        Parse refined query from LLM response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L140.
        """
        refined = response.strip()
        
        # Remove common prefixes
        prefixes = ["Refined Query:", "Output:", "Answer:", "Query:"]
        for prefix in prefixes:
            if refined.startswith(prefix):
                refined = refined[len(prefix):].strip()
        
        # Validate length
        if len(refined) < 5 or len(refined) > 300:
            print(f"  ⚠️  Invalid refined query length ({len(refined)}), using original")
            return original_query
        
        # Avoid identical query
        if refined.lower() == original_query.lower():
            print(f"  ⚠️  Refined query identical to original, using original")
            return original_query
        
        return refined
    
    def _rrf_fusion(
        self,
        results1: List[Tuple[Dict[str, Any], float]],
        results2: List[Tuple[Dict[str, Any], float]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Fuse two result lists using Reciprocal Rank Fusion (RRF).
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L191. 
        """
        k = self.config.rrf_k
        
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Process first result list
        for rank, (doc, _) in enumerate(results1, start=1):
            doc_id = doc.get("event_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        # Process second result list
        for rank, (doc, _) in enumerate(results2, start=1):
            doc_id = doc.get("event_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    def _multi_rrf_fusion(
        self,
        results_list: List[List[Tuple[Dict[str, Any], float]]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fuse multiple result lists using RRF."""
        if not results_list:
            return []
        
        if len(results_list) == 1:
            return results_list[0]
        
        k = self.config.rrf_k
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for results in results_list:
            for rank, (doc, _) in enumerate(results, start=1):
                doc_id = doc.get("event_id", id(doc))
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    # ========================
    # Reranker Methods
    # ========================
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Dict[str, Any], float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank retrieval results using neural reranker.
        
        For documents containing event_log:
        - Format as multi-line text: time + each atomic_fact on separate line

        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L869. 
        
        Features:
        - Batch processing with configurable batch size
        - Retry with exponential backoff
        - Timeout protection per batch
        - Fallback to original ranking when success rate is too low
        - Controlled concurrent batch processing
        
        Parameters
        ----------
        query : str
            The query to rerank against.
        results : List[Tuple[Dict, float]]
            Initial retrieval results.
        top_n : int, optional
            Number of documents to return after reranking.
            Defaults to config.final_top_k.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            Reranked results with reranker scores.
        """
        if not results:
            return []
        
        # Default to `final_top_k` if `top_n` is not specified
        effective_top_n = top_n if top_n is not None else self.config.final_top_k
        
        batch_size = self.config.reranker_batch_size
        max_retries = self.config.reranker_max_retries
        retry_delay = self.config.reranker_retry_delay
        timeout = self.config.reranker_timeout
        fallback_threshold = self.config.reranker_fallback_threshold
        max_concurrent = self.config.reranker_concurrent_batches
        reranker_instruction = self.config.reranker_instruction
        
        # Step 1: Format documents for reranker
        docs_with_text = []
        doc_texts = []
        original_indices = []
        
        for idx, (doc, _) in enumerate(results):
            # Prefer event_log to format text (if exists)
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                event_log = doc["event_log"]
                time_str = event_log.get("time", "")
                atomic_facts = event_log.get("atomic_fact", [])
                
                if isinstance(atomic_facts, list) and atomic_facts:
                    formatted_lines = []
                    if time_str:
                        formatted_lines.append(time_str)
                    
                    for fact in atomic_facts:
                        if isinstance(fact, dict) and "fact" in fact:
                            formatted_lines.append(fact["fact"])
                        elif isinstance(fact, str):
                            formatted_lines.append(fact)
                    
                    formatted_text = "\n".join(formatted_lines)
                    docs_with_text.append(doc)
                    doc_texts.append(formatted_text)
                    original_indices.append(idx)
                    continue
            
            # Fallback to episode field
            if episode_text := doc.get("episode"):
                docs_with_text.append(doc)
                doc_texts.append(episode_text)
                original_indices.append(idx)
        
        if not doc_texts:
            return results[:effective_top_n]
        
        # Check if reranker is available
        if self._reranker is None:
            print("  [Rerank] Warning: Reranker not initialized, using original ranking")
            return results[:effective_top_n]
        
        print(f"  [Rerank] Reranking {len(doc_texts)} documents in batches of {batch_size}...")
        
        # Step 2: Split into batches
        batches = []
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i + batch_size]
            batches.append((i, batch))
        
        print(f"  [Rerank] Split into {len(batches)} batches")
        
        # Process single batch with retry
        async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
            for attempt in range(max_retries):
                try:
                    batch_results = await asyncio.wait_for(
                        self._reranker.rerank_documents(
                            query, batch_texts, instruction=reranker_instruction
                        ),
                        timeout=timeout
                    )
                    
                    # Adjust indices to global indices
                    for item in batch_results["results"]:
                        item["global_index"] = start_idx + item["index"]
                    
                    if attempt > 0:
                        print(f"    ✓ Batch at {start_idx} succeeded on attempt {attempt + 1}")
                    return batch_results["results"]
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⏱️  Batch at {start_idx} timeout (attempt {attempt + 1}), retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    ❌ Batch at {start_idx} timeout after {max_retries} attempts")
                        return []
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⚠️  Batch at {start_idx} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    ❌ Batch at {start_idx} failed after {max_retries} attempts: {e}")
                        return []
        
        # Step 3: Process batches with controlled concurrency
        batch_results_list = []
        successful_batches = 0
        
        for group_start in range(0, len(batches), max_concurrent):
            group_batches = batches[group_start:group_start + max_concurrent]
            
            print(f"    Processing batch group {group_start // max_concurrent + 1} ({len(group_batches)} batches in parallel)...")
            
            tasks = [
                process_batch_with_retry(start_idx, batch)
                for start_idx, batch in group_batches
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in group_results:
                if isinstance(result, list) and result:
                    batch_results_list.append(result)
                    successful_batches += 1
                else:
                    batch_results_list.append([])
            
            # Inter-group delay
            if group_start + max_concurrent < len(batches):
                await asyncio.sleep(0.3)
        
        # Step 4: Merge results and apply fallback strategy
        all_rerank_results = []
        for batch_results in batch_results_list:
            all_rerank_results.extend(batch_results)
        
        success_rate = successful_batches / len(batches) if batches else 0.0
        print(f"  [Rerank] Success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)")
        
        # Fallback: complete failure
        if not all_rerank_results:
            print("  [Rerank] ⚠️ All batches failed, using original ranking")
            return results[:effective_top_n]
        
        # Fallback: success rate too low
        if success_rate < fallback_threshold:
            print(f"  [Rerank] ⚠️ Success rate too low ({success_rate:.1%} < {fallback_threshold:.1%}), using original ranking")
            return results[:effective_top_n]
        
        print(f"  [Rerank] Complete: {len(all_rerank_results)} documents scored")
        
        # Step 5: Sort by reranker score and return top-N
        sorted_results = sorted(
            all_rerank_results,
            key=lambda x: x["score"],
            reverse=True
        )[:effective_top_n]
        
        # Map back to original documents
        final_results = [
            (results[original_indices[item["global_index"]]][0], item["score"])
            for item in sorted_results
        ]
        
        return final_results