import asyncio
import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import replace
from rank_bm25 import BM25Okapi
from pydantic import (
    BaseModel, 
    Field,
    JsonValue,
)
from agentscope.agent import ReActAgent
from agentscope.embedding import OpenAITextEmbedding
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.rag import (
    Document,
    DocMetadata,
    QdrantStore,
    SimpleKnowledge,
)
from agentscope.token import OpenAITokenCounter
from graphtrace import (
    ChatUsageTokenMonitor,
    GraphTraceAgent,
    GraphTraceNotebook,
    StudioServer,
    agentscope_token_monitor,
)
from obstrace import (
    CCTraceNotebook,
    ObsTraceAttributionAgent,
    flatten_execution_graph,
)
from smartcomment.runtime import ExecNetwork
from smartcomment.runtime.variable import RuntimeVariable
from .bench_utils import FailedQueryCase
from ._base import AgentBaseConfig, AgentBaseRunner
from typing import Any, Literal


SYSTEM_PRIOR_INSTRUCTIONS = {
    "rag": """System-specific prior (RAG):
- Memory Pipeline:
  - During memory construction, each incoming turn is appended to a temporary buffer with token-count tracking.
  - When appending a new turn would exceed the max-token limit (and the buffer is non-empty), the pre-overflow buffered content is emitted as a document.
  - The emitted document is converted into a memory unit and written into the retrievable memory store.
  - During retrieval, the system uses the user query to search the memory store and returns top-k relevant memory units.
  - The retrieved memory units are then passed to the answer model to generate the final response.
- Attribution Priority:
  - In this traced RAG setup, prioritize retrieval-stage verification first: check whether retrieved context already includes all required evidence.
  - If required evidence is present in retrieval output, move directly to response-stage diagnosis.
  - Keep using the global earliest-decisive-fault criterion.
""",
    "mem0": """System-specific prior (mem0):
- Memory Pipeline:
  - During memory construction, given a message, the system uses an LLM to extract a list of key facts.
  - Each extracted fact is then directly used as a query to search the memory store at the current time step for similar memory units.
  - Based on the extracted facts and retrieved similar memory units, the LLM decides whether to create new memory units, update existing ones, or delete existing ones.
  - During retrieval, the system directly uses the user's input as the query to retrieve relevant memory units from the memory store.
  - The retrieved memory units are then passed to a downstream question-answering model to generate the final response.
- Attribution Priority:
  - First confirm whether the required information is correctly extracted.
  - If required information enters memory units, follow the same memory-unit ids downstream and verify whether later update/delete effects degrade or remove that information before blaming retrieval.
  - If memory-unit evolution is locally correct, then inspect retrieval completeness, and finally response generation.
  - Keep using the global earliest-decisive-fault criterion.
  """,
    "evermemos": """System-specific prior (evermemos):
- Memory Pipeline:
  - During memory construction, each incoming message is appended to a running buffer, and the system checks boundary conditions to decide whether to keep buffering or close the current segment.
  - When a segment is closed, the buffered segment is finalized into an identifiable memory unit (event id), then enriched into episode memory and optional retrieval fields (for example, event-log or foresight).
  - The refined memory unit is indexed for retrieval, transient raw fields are cleaned up, and the message buffer is reset under smart-mask behavior.
  - During retrieval Round 1, the system runs hybrid retrieval to build the first candidate pool, then builds a sufficiency-check evidence set (reranked subset when reranker is enabled, otherwise direct top subset).
  - The system runs an LLM sufficiency check; if Round-1 evidence is sufficient, it returns final top-k directly from Round-1 ordering.
  - If Round-1 evidence is insufficient, retrieval enters Round 2: it generates follow-up query or multi-query set from missing-information signals, runs hybrid retrieval for each follow-up query, and fuses second-round branches (typically via multi-query RRF for multi-query cases).
  - During final selection, the system merges Round-1 and Round-2 pools, deduplicates by event identity, keeps a bounded combined pool, and applies final rerank (or direct truncation when reranker is disabled) to produce final top-k context.
  - The final retrieved context is then passed to the answer model to generate the response.
- Attribution Priority:
  - First validate segmentation and memory formation quality (boundary decisions, finalize, extraction), since retrieval quality depends on these outputs.
  - For retrieval-stage diagnosis, follow the actually executed branch end-to-end rather than a single intermediate list; verify candidate generation, fusion, second-round expansion (if any), merge/dedup, and final selection output.
  - Only after confirming final retrieval context quality should response-stage attribution be considered.
  - Keep using the global earliest-decisive-fault criterion.
""",
    "long_context": """System-specific prior (long-context):
- Memory Pipeline:
  - During memory construction, each new turn is merged into one persistent long-context memory unit instead of creating multiple units.
  - If the accumulated context exceeds the configured window budget, older content is trimmed in the same update so the single unit stays within limits.
  - During retrieval, the system returns this single retained long-context unit directly.
  - The retained long-context unit is then passed to the answer model to generate the final response.
- Attribution Priority:
  - First check whether the required evidence was dropped when older messages were trimmed to keep the context within the window budget.
  - In this single-unit setting, retrieval is typically pass-through of retained context, so prioritize update-stage diagnosis over retrieval-stage blame.
  - If no decisive update-stage fault is found, move directly to response-stage diagnosis.
  - Keep using the global earliest-decisive-fault criterion.
""",
}


ATTRIBUTION_INSTRUCTIONS = """Your task is to inspect the execution graph and identify the earliest decisive faulty operation.

How to read the execution graph:
- The execution graph consists of operations. Each operation forms a single-hop or multi-hop subgraph.
- Within one operation subgraph, variables with in-degree 0 relative to that subgraph are the operation's input variables.
- Variables with in-degree >= 1 relative to that subgraph are variables produced by the operation.
- Variables with out-degree 0 relative to that subgraph are the operation's final output variables.
- To judge an operation, inspect its operation name, category, comment, input variables, produced variables, final output variables, variable values, variable comments, variable relationships, and creation timestamps.
- If the current operation is locally correct for its role, it is not the error. Continue downstream by adding one or more final output variables from the current operation to the exploration frontier and inspect the subsequent operations that consume them.
- Pay close attention to creation time. The earliest decisive fault is based on operation execution order, not on which source message is created first.

Ideal memory-system behavior:
- When a raw input containing critical information arrives, the memory construction stage should create or update memory-unit containers so that the critical information enters the memory store.
- Later operations over a memory-unit container should preserve the critical information. They should not accidentally remove, overwrite, over-summarize, or degrade it during updates, consolidation, or deletion.
- When a query requiring that critical information arrives, retrieval should surface memory units or context containing the needed information.
- The downstream answer model should then use the retrieved context to answer the question correctly.
- Compare the observed trace against this ideal behavior to decide where the first decisive break occurs.

Definition of earliest decisive faulty operation:
- An operation is faulty only if its outputs are wrong or insufficient for the role it is supposed to perform, given its operation name, category, comment, inputs, and available evidence at that point in the workflow.
- Do not mark an operation faulty merely because it appears early or because a downstream failure can be traced back to one of its inputs. First decide whether the operation itself behaved incorrectly under its own local responsibility.
- Decisive means that correcting this operation's faulty output, while keeping all strictly earlier operations unchanged and assuming ideal downstream behavior, would have prevented the final wrong answer.
- Earliest means the first such locally faulty and decisive operation in execution order.

Required reasoning checklist before returning the final answer:
1. Identify the critical information needed to answer the question.
2. Inspect candidate operations in execution order and decide whether each one is locally correct for its own function.
3. If an operation is locally correct, follow its final output variables downstream instead of labeling it as the error.
4. Once the critical information enters a memory unit, do not stop at that operation. You must keep following downstream operations involving that memory unit and check whether the critical information remains present throughout the lifetime of the memory unit. If an operation causes this critical information to be transferred from one memory unit to another memory unit or to some intermediate variables, and the original memory unit no longer contains this information, then track the other memory unit or intermediate variables (that contain this key information) to ensure that there exists a memory unit that will hold this key information.
5. If the critical information never enters any memory unit, label the responsible construction operation as ExtractionError.
6. If the critical information enters a memory unit but a later operation removes or degrades it, label that later operation as UpdateError or DeletionError depending on whether it updates the unit or explicitly deletes it.
7. For retrieval candidates, verify that the memory store contains the required information before retrieval, but the retrieval pipeline fails to include it in the final retrieved context.
8. For response candidates, verify that the final retrieved context contains all necessary evidence and the answer model still answers incorrectly.

Return exactly one error attribution using the required structured schema:
- error_type: one of ExtractionError, UpdateError, DeletionError, RetrievalError, ResponseError.
- op_id: the operation identifier of the earliest decisive faulty operation.
- reason: a concise explanation grounded in graph evidence. The reason must state why the chosen operation is locally faulty, why correcting it would rescue the answer, and why earlier candidate operations are not the decisive fault when relevant.

Error type definitions:
- ExtractionError: The critical information is never captured into any memory unit during memory construction, and thus never enters the memory store.
- UpdateError: A memory unit initially contains the critical information, but a subsequent update removes or degrades it.
- DeletionError: A memory unit containing critical information is explicitly removed.
- RetrievalError: The memory store contains the required information, but retrieval fails to include it in the final retrieved context.
- ResponseError: The retrieved context contains all necessary evidence, but the final LLM still answers incorrectly.
"""


OBS_ATTRIBUTION_INSTRUCTIONS = """Your task is to inspect the execution trace and identify the earliest decisive faulty operation.

Ideal memory-system behavior:
- When a raw input containing critical information arrives, the memory construction stage should create or update memory-unit containers so that the critical information enters the memory store.
- Later operations over a memory-unit container should preserve the critical information. They should not accidentally remove, overwrite, over-summarize, or degrade it during updates, consolidation, or deletion.
- When a query requiring that critical information arrives, retrieval should surface memory units or context containing the needed information.
- The downstream answer model should then use the retrieved context to answer the question correctly.
- Compare the observed trace against this ideal behavior to decide where the first decisive break occurs.

Definition of earliest decisive faulty operation:
- An operation is faulty only if its outputs are wrong or insufficient for the role it is supposed to perform, given its operation name, category, comment, inputs, and available evidence at that point in the workflow.
- Do not mark an operation faulty merely because it appears early or because a downstream failure can be traced back to one of its inputs. First decide whether the operation itself behaved incorrectly under its own local responsibility.
- Decisive means that correcting this operation's faulty output, while keeping all strictly earlier operations unchanged and assuming ideal downstream behavior, would have prevented the final wrong answer.
- Earliest means the first such locally faulty and decisive operation in execution order.

Required reasoning checklist before returning the final answer:
1. Identify the critical information needed to answer the question.
2. Inspect candidate operations in execution order and decide whether each one is locally correct for its own function.
3. If an operation is locally correct, continue downstream instead of labeling it as the error.
4. Once the critical information enters a memory unit, do not stop at that operation. You must keep following downstream operations involving that memory unit and check whether the critical information remains present throughout the lifetime of the memory unit. If an operation causes this critical information to be transferred from one memory unit to another memory unit or to some intermediate variables, and the original memory unit no longer contains this information, then track the other memory unit or intermediate variables that contain this key information to ensure that there exists a memory unit that will hold this key information.
5. If the critical information never enters any memory unit, label the responsible construction operation as ExtractionError.
6. If the critical information enters a memory unit but a later operation removes or degrades it, label that later operation as UpdateError or DeletionError depending on whether it updates the unit or explicitly deletes it.
7. For retrieval candidates, verify that the memory store contains the required information before retrieval, but the retrieval pipeline fails to include it in the final retrieved context.
8. For response candidates, verify that the final retrieved context contains all necessary evidence and the answer model still answers incorrectly.

Return exactly one error attribution using the required structured schema:
- error_type: one of ExtractionError, UpdateError, DeletionError, RetrievalError, ResponseError.
- op_id: the operation identifier of the earliest decisive faulty operation.
- reason: a concise explanation grounded in trace evidence. The reason must state why the chosen operation is locally faulty, why correcting it would rescue the answer, and why earlier candidate operations are not the decisive fault when relevant.

Error type definitions:
- ExtractionError: The critical information is never captured into any memory unit during memory construction, and thus never enters the memory store.
- UpdateError: A memory unit initially contains the critical information, but a subsequent update removes or degrades it.
- DeletionError: A memory unit containing critical information is explicitly removed.
- RetrievalError: The memory store contains the required information, but retrieval fails to include it in the final retrieved context.
- ResponseError: The retrieved context contains all necessary evidence, but the final LLM still answers incorrectly.
"""


TASK_PROMPT_TEMPLATES = {
    "operation_block_search": """You are performing failure attribution for a memory-based question-answering system.

The execution trace is available through trace inspection tools. First inspect the relevant evidence before deciding on the earliest decisive fault.

{instructions}

Question:
{question}

Golden answer:
{golden_answer}

Model's wrong answer:
{prediction}
""",
    "long_context": """You are performing failure attribution for a memory-based question-answering system.

The execution trace is provided below. Note that the provided source-evidence list contains multiple evidence items, some of which may be irrelevant to the question. During failure attribution, you should independently assess the validity and relevance of each item.

{instructions}

Question:
{question}

Golden answer:
{golden_answer}

Model's wrong answer:
{prediction}

Source evidence:
{source_evidence_text}

Execution trace:
{execution_trace}
""",
    "with_source_evidence_initial_nodes": """You are performing failure attribution for a memory-based question-answering system.

The execution graph has already been initialized with the query node and the source-evidence message nodes in the exploration frontier.

{instructions}

Question:
{question}

Golden answer:
{golden_answer}

Model's wrong answer:
{prediction}

Query node:
{query_node_text}

Source evidence:
{source_evidence_text}
""",
    "with_pseudo_source_evidence_initial_nodes": """You are performing failure attribution for a memory-based question-answering system.

The execution graph has already been initialized with the query node and a set of retrieved starting message nodes in the exploration frontier.

These initialized message nodes may not be the correct source evidence, so you must verify them and determine the true source evidence through graph inspection.

{instructions}

Question:
{question}

Golden answer:
{golden_answer}

Model's wrong answer:
{prediction}

Query node:
{query_node_text}

Source evidence:
{source_evidence_text}
""",
    "without_initial_nodes": """You are performing failure attribution for a memory-based question-answering system.

The execution graph is available through graph-trace tools, but no task-specific starting nodes have been provided. First initialize and explore the graph.

{instructions}

Question:
{question}

Golden answer:
{golden_answer}

Model's wrong answer:
{prediction}

Query node:
{query_node_text}
""",
}


class ErrorAttributionPrediction(BaseModel):
    """Structured output for one error attribution prediction."""

    error_type: Literal[
        "ExtractionError",
        "UpdateError",
        "DeletionError",
        "RetrievalError",
        "ResponseError",
    ] = Field(
        description="The predicted memory-system error type."
    )
    op_id: str = Field(
        description="The operation identifier for the earliest decisive error.",
    )
    reason: str = Field(
        description="A concise graph-grounded reason for the error attribution.",
    )


class MemTraceConfig(AgentBaseConfig):
    """The configuration for MemTrace."""

    exploration_strategy: Literal[
        "graph_search",
        "operation_block_search",
        "long_context",
    ] = Field(
        default="graph_search",
        description="The strategy used to explore the execution graph.",
    )
    context_window: int = Field(
        default=272_000,
        description="The working context window size for the graph trace agent.",
        ge=1,
    )
    max_context_limit: int = Field(
        default=1_000_000,
        description=(
            "The maximum context length allowed when preparing messages for "
            "memory compression in the agent. It prevents the messages to be "
            "compressed from exceeding the model's context window, which would "
            "otherwise make the compression request itself overflow the context."
        ),
        ge=1,
    )
    full_log_token_limit: int = Field(
        default=600_000,
        description=(
            "Token budget for the flattened execution trace. "
            "It is truncated from the beginning so the latest operations can be kept. "
            "Note that this parameter is only used for the long-context baseline."
        ),
        ge=1,
    )
    keep_recent: int = Field(
        default=3,
        description=(
            "The number of most recent messages to keep uncompressed "
            "during the working context compression."
        ),
        ge=0,
    )
    max_trace_nodes: int = Field(
        default=16,
        description="Maximum number of graph trace nodes the notebook may explore.",
        ge=1,
    )
    max_iters: int = Field(
        default=200,
        description="Maximum number of reasoning iterations per attribution case.",
        ge=1,
    )
    embedding_model_name: str = Field(
        default="text-embedding-3-small",
        description="The OpenAI-compatible embedding model name used for retrieval.",
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="The dimensionality of the embedding vectors.",
        ge=1,
    )
    embedding_base_url: str | None = Field(
        default=None,
        description=(
            "The base URL for the OpenAI-compatible embedding endpoint. "
            "If not provided, the embedding client falls back to a base URL "
            "used by the agent backbone model."
        ),
    )
    embedding_api_key: str | None = Field(
        default=None,
        description=(
            "The API key for the OpenAI-compatible embedding endpoint. "
            "If not provided, the embedding client falls back to an API key "
            "used by the agent backbone model."
        ),
    )
    embedding_batch_size: int = Field(
        default=16,
        description="The batch size for the embedding endpoint.",
        ge=1,
    )
    retrieval_type: Literal["sparse", "dense", "hybrid"] = Field(
        default="hybrid",
        description=(
            "The retrieval strategy used to select pseudo source-evidence "
            "starting points when real source evidence is not provided."
        ),
    )
    starting_point_query_type: Literal[
        "query_with_golden_answer",
        "query_only",
        "query_with_prediction",
    ] = Field(
        default="query_with_golden_answer",
        description=(
            "The case fields used to construct the retrieval query for pseudo "
            "source-evidence starting points."
        ),
    )
    num_starting_points: int | None = Field(
        default=None,
        description=(
            "The number of retrieved pseudo source-evidence starting points. "
            "If not provided, it defaults to half of the maximum size of the "
            "to-explore list."
        ),
        ge=1,
    )
    candidate_multiplier: int = Field(
        default=2,
        description=(
            "It is only used for hybrid retrieval. The larger it is, the more "
            "candidates the sparse and dense retrievers each collect before they "
            "are merged, which gives the fusion a larger pool to choose the final "
            "results from."
        ),
        ge=1,
    )
    cache_dir: str | None = Field(
        default=None,
        description=(
            "The directory used to cache per-case attribution results. When it "
            "is provided, each attributed case is saved as one JSON file and "
            "reused on later runs to avoid recomputation. If it is not provided, "
            "caching is disabled."
        ),
    )


def _format_query_node(query_node: RuntimeVariable[Any]) -> str:
    """Format the failed query node for the attribution prompt.

    Args:
        query_node (`RuntimeVariable[Any]`):
            The failed query node.

    Returns:
        `str`:
            Human-readable query node text.
    """
    return (
        f"Full Node Identifier: {query_node.full_node_id}\n"
        f"Value: {query_node.value}"
    )


def _format_source_evidence(nodes: list[RuntimeVariable[Any]]) -> str:
    """Format source-evidence nodes for the attribution prompt.

    Args:
        nodes (`list[RuntimeVariable[Any]]`):
            Source-evidence nodes.

    Returns:
        `str`:
            Human-readable source-evidence text.
    """
    if not nodes:
        return "There are no relevant source-evidence nodes."

    chunks = []
    for index, node in enumerate(nodes, start=1):
        chunks.append(
            f"### Source Evidence {index}\n"
            f"Full Node Identifier: {node.full_node_id}\n"
            f"Creation Timestamp In Execution Graph: {node.created_at}\n"
            f"Value: {node.value}"
        )
    return "\n\n".join(chunks)


def _tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight lexical (BM25) retrieval.

    Args:
        text (`str`):
            The input text.

    Returns:
        `list[str]`:
            A token list containing English word tokens.
    """
    lowered = text.lower()
    return re.findall(r"[a-z0-9_]+", lowered)


def _rrf_fuse(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF).

    Args:
        ranked_lists (`list[list[str]]`):
            Multiple ranked lists of document identifiers.
        rrf_k (`int`, defaults to `60`):
            The RRF smoothing constant used in RRF.

    Returns:
        `list[tuple[str, float]]`:
            A list of tuples containing the document identifier and the 
            corresponding score.    
    """
    scores = {}
    best_rank = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            if doc_id not in best_rank or rank < best_rank[doc_id]:
                best_rank[doc_id] = rank
    return sorted(
        scores.items(),
        key=lambda item: (-item[1], best_rank[item[0]], item[0]),
    )


class DocumentRetriever(SimpleKnowledge):
    """A general-purpose document retriever over a fixed document collection."""

    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-small",
        embedding_dimensions: int = 1536,
        embedding_base_url: str | None = None,
        embedding_api_key: str | None = None,
        embedding_batch_size: int = 16,
        retrieval_type: Literal["sparse", "dense", "hybrid"] = "hybrid",
        candidate_multiplier: int = 2,
    ) -> None:
        """Initialize the retrieval module and set up the configured indexes.

        Args:
            embedding_model_name (`str`, defaults to `"text-embedding-3-small"`):
                The OpenAI-compatible embedding model name. It will be ignored 
                if the retrieval type is ``"sparse"``.
            embedding_dimensions (`int`, defaults to `1536`):
                The dimensionality of the embedding vectors. It will be ignored 
                if the retrieval type is ``"sparse"``.
            embedding_base_url (`str | None`, optional):
                The base URL for the OpenAI-compatible embedding endpoint. If
                not provided, it falls back to ``OPENAI_BASE_URL`` or ``OPENAI_API_BASE`` 
                environment variables. It will be ignored if the retrieval type is ``"sparse"``.
            embedding_api_key (`str | None`, optional):
                The API key for the OpenAI-compatible embedding endpoint. If not
                provided, it falls back to the ``OPENAI_API_KEY`` environment
                variable. It will be ignored if the retrieval type is ``"sparse"``.
            embedding_batch_size (`int`, defaults to `16`):
                The number of documents embedded per request when building the
                dense index. It will be ignored if the retrieval type is ``"sparse"``.
            retrieval_type (`Literal["sparse", "dense", "hybrid"]`, defaults to `"hybrid"`):
                The retrieval strategy used to build the index.
            candidate_multiplier (`int`, defaults to `2`):
                It is only used for hybrid retrieval. Before fusing the results,
                the sparse and dense retrievers each fetch this many times as
                many candidates as the number of documents finally returned, so
                that the fusion has a larger pool to work with.
        """
        if retrieval_type not in ("sparse", "dense", "hybrid"):
            raise ValueError(
                "`retrieval_type` must be one of 'sparse', 'dense', or 'hybrid'."
            )
        if candidate_multiplier < 1:
            raise ValueError("`candidate_multiplier` must be greater than or equal to 1.")
        if embedding_batch_size < 1:
            raise ValueError("`embedding_batch_size` must be greater than or equal to 1.")

        self._candidate_multiplier = candidate_multiplier
        self._embedding_batch_size = embedding_batch_size

        embedding_model = embedding_store = None
        if retrieval_type in ("dense", "hybrid"):
            api_key = embedding_api_key or os.environ.get("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "An embedding API key is required for dense retrieval. "
                    "Provide `embedding_api_key` or set `OPENAI_API_KEY`."
                )
            base_url = (
                embedding_base_url
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
            )
            client_kwargs = {}
            if base_url is not None:
                client_kwargs["base_url"] = base_url
            embedding_model = OpenAITextEmbedding(
                api_key=api_key,
                model_name=embedding_model_name,
                dimensions=embedding_dimensions,
                **client_kwargs,
            )
            embedding_store = QdrantStore(
                location=":memory:",
                collection_name="documents",
                dimensions=embedding_dimensions,
            )
        super().__init__(
            embedding_store=embedding_store,
            embedding_model=embedding_model,
        )

        # The canonical document store keyed by the document identifier, used to
        # return the original documents.
        self._documents = {}
        # The BM25 index is rebuilt from this corpus on every change.
        self._bm25_corpus = (
            [] if retrieval_type in ("sparse", "hybrid") else None
        )
        self._bm25 = None

    @property
    def retrieval_type(self) -> Literal["sparse", "dense", "hybrid"]:
        """Return the effective retrieval type.

        Returns:
            `Literal["sparse", "dense", "hybrid"]`:
                The effective retrieval type.
        """
        has_sparse = self._bm25_corpus is not None
        has_dense = self.embedding_store is not None
        if has_sparse and has_dense:
            return "hybrid"
        if has_dense:
            return "dense"
        return "sparse"

    async def add_documents(self, documents: list[Document], **kwargs: Any) -> None:
        """Add documents to the configured indexes.

        Args:
            documents (`list[Document]`):
                The documents to add. Each document's identity is read from
                its metadata.
            **kwargs (`Any`):
                Additional keyword arguments forwarded to the embedding model.
        """
        if not documents:
            return

        valid_documents = []
        for doc in documents:
            text = doc.metadata.content["text"]
            if not text or not text.strip():
                warnings.warn(
                    f"Document '{doc.metadata.doc_id}' has empty content and is skipped.",
                    UserWarning,
                )
                continue
            valid_documents.append(doc)
        if not valid_documents:
            return

        for doc in valid_documents:
            self._documents[doc.metadata.doc_id] = doc

        if self._bm25_corpus is not None:
            self._bm25_corpus.extend(
                (doc.metadata.doc_id, _tokenize(doc.metadata.content["text"]))
                for doc in valid_documents
            )
            self._bm25 = BM25Okapi([tokens for _, tokens in self._bm25_corpus])

        if self.embedding_store is not None:
            for start in range(0, len(valid_documents), self._embedding_batch_size):
                batch = valid_documents[start:start + self._embedding_batch_size]
                res_embeddings = await self.embedding_model(
                    [doc.metadata.content for doc in batch],
                    **kwargs,
                )
                for doc, embedding in zip(batch, res_embeddings.embeddings):
                    doc.embedding = embedding
            await self.embedding_store.add(valid_documents)

    def _sparse_scores(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Retrieve candidates via sparse retrieval.

        Args:
            query (`str`):
                The retrieval query.
            limit (`int`, defaults to `10`):
                The maximum number of candidates to return.
            score_threshold (`float | None`, optional):
                The score threshold used to filter the candidates.

        Returns:
            `list[tuple[str, float]]`:
                A list of tuples containing the document identifier and the 
                corresponding score.
        """
        if self._bm25 is None:
            return []
        doc_ids = [doc_id for doc_id, _ in self._bm25_corpus]
        query_tokens = _tokenize(query)
        if not query_tokens:
            scores = [0.0] * len(doc_ids)
        else:
            scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(
            ((doc_id, float(score)) for doc_id, score in zip(doc_ids, scores)),
            key=lambda item: -item[1],
        )
        results = []
        for doc_id, score in ranked:
            if score_threshold is not None and score < score_threshold:
                continue
            results.append((doc_id, score))
            if len(results) >= limit:
                break
        return results

    async def _dense_scores(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Retrieve candidates via dense retrieval.

        Args:
            query (`str`):
                The retrieval query.
            limit (`int`, defaults to `10`):
                The maximum number of candidates to return.
            score_threshold (`float | None`, optional):
                The embedding score threshold passed to the vector store.
            **kwargs (`Any`):
                Additional keyword arguments forwarded to the embedding model.

        Returns:
            `list[tuple[str, float]]`:
                A list of tuples containing the document identifier and the 
                corresponding score.
        """
        res_embedding = await self.embedding_model(
            [TextBlock(type="text", text=query)],
            **kwargs,
        )
        res = await self.embedding_store.search(
            res_embedding.embeddings[0],
            limit=limit,
            score_threshold=score_threshold,
        )
        return [(doc.metadata.doc_id, float(doc.score)) for doc in res]

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve the top-K documents for the query.

        Args:
            query (`str`):
                The retrieval query.
            limit (`int`, defaults to `10`):
                The number of documents to return.
            score_threshold (`float | None`, optional):
                The score threshold used to filter the results.
            **kwargs (`Any`):
                Additional keyword arguments forwarded to the embedding model.

        Returns:
            `list[Document]`:
                The retrieved documents carrying the corresponding retrieval score.
        """
        retrieval_type = self.retrieval_type

        if retrieval_type == "sparse":
            ranked = self._sparse_scores(
                query,
                limit=limit,
                score_threshold=score_threshold,
            )
        elif retrieval_type == "dense":
            ranked = await self._dense_scores(
                query,
                limit=limit,
                score_threshold=score_threshold,
                **kwargs,
            )
        else:
            # Hybrid retrieval: oversample each retriever without a per-retriever
            # threshold, fuse with RRF, then filter the fused score by threshold.
            candidate_limit = limit * self._candidate_multiplier
            sparse_ranked = self._sparse_scores(query, limit=candidate_limit)
            dense_ranked = await self._dense_scores(
                query,
                limit=candidate_limit,
                **kwargs,
            )
            fused = _rrf_fuse(
                [
                    [doc_id for doc_id, _ in sparse_ranked],
                    [doc_id for doc_id, _ in dense_ranked],
                ]
            )
            ranked = []
            for doc_id, score in fused:
                if score_threshold is not None and score < score_threshold:
                    continue
                ranked.append((doc_id, score))
                if len(ranked) >= limit:
                    break

        return [
            replace(self._documents[doc_id], score=score)
            for doc_id, score in ranked
        ]


class MemTraceRunner(AgentBaseRunner):
    """Runner that concurrently attributes failed memory-query cases."""

    def __init__(self, config: MemTraceConfig | None = None) -> None:
        """Initialize the MemTrace runner.

        Args:
            config (`MemTraceConfig | None`, optional):
                The MemTrace runner configuration. If not provided, 
                default configuration is used.
        """
        super().__init__(config or MemTraceConfig())

    def _cache_file_path(self, identifier: JsonValue) -> str | None:
        """Resolve the cache file path for one identifier.

        Args:
            identifier (`JsonValue`):
                The cache identifier.

        Returns:
            `str | None`:
                If caching is enabled, it is the cache file path.
        """
        if self.config.cache_dir is None:
            return None
        os.makedirs(self.config.cache_dir, exist_ok=True)
        json_str = json.dumps(identifier, sort_keys=True, ensure_ascii=False)
        filename = hashlib.sha256(json_str.encode("utf-8")).hexdigest() + ".json"
        return os.path.join(self.config.cache_dir, filename)

    def _load_cached_case(self, identifier: JsonValue) -> JsonValue:
        """Load a cached attribution case if it exists.

        Args:
            identifier (`JsonValue`):
                The cache identifier.

        Returns:
            `JsonValue`:
                If the case is cached, it is the cached payload with the 
                error attribution result and the elapsed seconds.
        """
        path = self._cache_file_path(identifier)
        if path is None or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _store_cached_case(self, identifier: JsonValue, payload: JsonValue) -> None:
        """Store one attributed case to the cache.

        Args:
            identifier (`JsonValue`):
                The cache identifier.
            payload (`JsonValue`):
                The payload with the error attribution result and the elapsed seconds.
        """
        path = self._cache_file_path(identifier)
        if path is None:
            return
        with open(path, "w", encoding="utf-8") as file:
            json.dump(
                payload, 
                file, 
                ensure_ascii=False, 
                indent=4
            )

    def _build_agent(
        self,
        notebook: GraphTraceNotebook,
        api_key: str,
        base_url: str,
    ) -> GraphTraceAgent:
        """Build a graph-trace attribution agent.

        Args:
            notebook (`GraphTraceNotebook`):
                The initialized graph trace notebook.
            api_key (`str`):
                OpenAI-compatible API key.
            base_url (`str`):
                OpenAI-compatible base URL.

        Returns:
            `GraphTraceAgent`:
                Configured attribution agent.
        """
        cfg = self.config
        client_kwargs = {"base_url": base_url}
        model = OpenAIChatModel(
            model_name=cfg.model,
            api_key=api_key,
            stream=cfg.stream,
            client_kwargs=client_kwargs,
            generate_kwargs={"temperature": cfg.temperature},
        )
        return GraphTraceAgent(
            name="memtrace",
            sys_prompt=(
                "You are a careful execution-graph failure attribution agent. "
                "Use the graph trace tools to inspect evidence before answering."
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            graph_trace_notebook=notebook,
            max_iters=cfg.max_iters,
            max_context_limit=cfg.max_context_limit,
            compression_config=GraphTraceAgent.CompressionConfig(
                enable=True,
                agent_token_counter=OpenAITokenCounter(cfg.model),
                trigger_threshold=cfg.context_window,
                keep_recent=cfg.keep_recent,
            ),
        )

    def _build_obs_agent(
        self,
        notebook: CCTraceNotebook,
        api_key: str,
        base_url: str,
    ) -> ObsTraceAttributionAgent:
        """Build a flat operation-log attribution agent."""
        cfg = self.config
        model = OpenAIChatModel(
            model_name=cfg.model,
            api_key=api_key,
            stream=cfg.stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": cfg.temperature},
        )
        return ObsTraceAttributionAgent(
            name="memtrace",
            sys_prompt=(
                "You are a careful execution-trace failure attribution agent. "
                "Use the trace inspection tools to inspect evidence before answering."
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            cc_trace_notebook=notebook,
            max_iters=cfg.max_iters,
            compression_config=ObsTraceAttributionAgent.CompressionConfig(
                enable=True,
                agent_token_counter=OpenAITokenCounter(cfg.model),
                trigger_threshold=cfg.context_window,
                keep_recent=cfg.keep_recent,
            ),
        )

    def _build_long_context_agent(
        self,
        api_key: str,
        base_url: str,
    ) -> ReActAgent:
        """Build a long-context attribution agent without trace tools.

        Args:
            api_key (`str`):
                OpenAI-compatible API key.
            base_url (`str`):
                OpenAI-compatible base URL.

        Returns:
            `ReActAgent`:
                Configured long-context attribution agent.
        """
        cfg = self.config
        model = OpenAIChatModel(
            model_name=cfg.model,
            api_key=api_key,
            stream=cfg.stream,
            client_kwargs={"base_url": base_url},
            generate_kwargs={"temperature": cfg.temperature},
        )
        return ReActAgent(
            name="memtrace",
            sys_prompt=(
                "You are a careful execution-trace failure attribution agent. "
                "Inspect the complete execution trace before answering."
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            max_iters=cfg.max_iters,
        )

    def _build_task_prompt(
        self,
        case: FailedQueryCase,
        query_node: RuntimeVariable[Any],
        source_evidence_nodes: list[RuntimeVariable[Any]] | None = None,
        is_pseudo_source_evidence: bool = False,
        memory_system: str | None = None, 
        custom_prior_knowledge: str | None = None,
        execution_trace: str | None = None,
    ) -> str:
        """Build the attribution task prompt for one failed case.

        Args:
            case (`FailedQueryCase`):
                The failed query case.
            query_node (`RuntimeVariable[Any]`):
                The failed query node.
            source_evidence_nodes (`list[RuntimeVariable[Any]] | None`, optional):
                The source-evidence nodes. If it is not provided, it uses 
                the prompt requiring the agent to initialize the graph.
            is_pseudo_source_evidence (`bool`, defaults to `False`):
                Whether the source-evidence nodes are pseudo. For example, these 
                source evidence nodes are obtained by the agent or retriever 
                rather than provided by the dataset.
            memory_system (`str | None`, optional):
                The memory system currently being diagnosed. If it is provied, 
                the relevant prior knowledge will be included in the prompt.
                If `custom_prior_knowledge` is provided, it will be ignored.  
            custom_prior_knowledge (`str | None`, optional):
                Custom prior knowledge. If it is provided, it will be included in 
                the prompt.
            execution_trace (`str | None`, optional):
                Flattened execution trace included by the long-context
                attribution strategy.

        Returns:
            `str`:
                The attribution task prompt.
        """
        if self.config.exploration_strategy == "operation_block_search":
            return TASK_PROMPT_TEMPLATES["operation_block_search"].format(
                instructions=OBS_ATTRIBUTION_INSTRUCTIONS,
                question=case.query,
                golden_answer=case.golden_answer,
                prediction=case.prediction,
            )

        if self.config.exploration_strategy == "long_context":
            if execution_trace is None:
                raise ValueError(
                    "`execution_trace` is required for the long-context strategy."
                )
            return TASK_PROMPT_TEMPLATES["long_context"].format(
                instructions=OBS_ATTRIBUTION_INSTRUCTIONS,
                question=case.query,
                golden_answer=case.golden_answer,
                prediction=case.prediction,
                source_evidence_text=_format_source_evidence(
                    source_evidence_nodes or []
                ),
                execution_trace=execution_trace,
            )

        instructions = ATTRIBUTION_INSTRUCTIONS
        if custom_prior_knowledge is not None:           
            instructions = f"{instructions}\n\n{custom_prior_knowledge}"
        elif memory_system is not None:
            prior_knowledge = SYSTEM_PRIOR_INSTRUCTIONS.get(memory_system)
            if prior_knowledge is None:
                warnings.warn(
                    f"No MemTrace prior knowledge is available for memory system "
                    f"'{memory_system}'.",
                    UserWarning,
                )
            else:
                instructions = f"{instructions}\n\n{prior_knowledge}"

        if source_evidence_nodes is None:
            template_key = "without_initial_nodes"
            source_evidence_text = ""
        elif is_pseudo_source_evidence:
            template_key = "with_pseudo_source_evidence_initial_nodes"
            source_evidence_text = _format_source_evidence(source_evidence_nodes)
        else:
            template_key = "with_source_evidence_initial_nodes"
            source_evidence_text = _format_source_evidence(source_evidence_nodes)

        return TASK_PROMPT_TEMPLATES[template_key].format(
            instructions=instructions,
            question=case.query,
            golden_answer=case.golden_answer,
            prediction=case.prediction,
            query_node_text=_format_query_node(query_node),
            source_evidence_text=source_evidence_text,
        )

    async def _retrieve_pseudo_source_evidence(
        self,
        graph: ExecNetwork,
        case: FailedQueryCase,
        case_index: int,
    ) -> list[RuntimeVariable[Any]]:
        """Retrieve pseudo source-evidence starting points for one failed case. 

        Args:
            graph (`ExecNetwork`):
                The execution graph containing the failed query.
            case (`FailedQueryCase`):
                The failed query case.
            case_index (`int`):
                Index of the case in the batch, used to pick API credentials.

        Returns:
            `list[RuntimeVariable[Any]]`:
                The retrieved pseudo source-evidence message nodes.
        """
        cfg = self.config
        message_nodes = graph.search_variables(category="message")
        documents = [
            Document(
                metadata=DocMetadata(
                    content=TextBlock(type="text", text=node.value),
                    doc_id=node.full_node_id,
                    chunk_id=0,
                    total_chunks=1,
                ),
                id=node.full_node_id,
            )
            for node in message_nodes
        ]

        api_key, base_url = self._api_pool.credential_for(case_index)
        retriever = DocumentRetriever(
            embedding_model_name=cfg.embedding_model_name,
            embedding_dimensions=cfg.embedding_dimensions,
            embedding_base_url=cfg.embedding_base_url or base_url,
            embedding_api_key=cfg.embedding_api_key or api_key,
            embedding_batch_size=cfg.embedding_batch_size,
            retrieval_type=cfg.retrieval_type,
            candidate_multiplier=cfg.candidate_multiplier,
        )
        await retriever.add_documents(documents)

        num_starting_points = (
            cfg.num_starting_points
            if cfg.num_starting_points is not None
            else cfg.max_trace_nodes // 2
        )
        if cfg.starting_point_query_type == "query_only":
            query = case.query
        elif cfg.starting_point_query_type == "query_with_prediction":
            query = f"{case.query}\n{case.prediction}"
        else:
            query = f"{case.query}\n{case.golden_answer}"
        retrieved = await retriever.retrieve(query, limit=num_starting_points)
        return [graph.get_variable(doc.metadata.doc_id) for doc in retrieved]

    async def arun(
        self,
        failed_cases: list[FailedQueryCase],
        graphs: ExecNetwork | list[ExecNetwork],
        batch_size: int | None = None,
        use_source_evidence_nodes: bool = False,
        use_pseudo_source_evidence: bool = True,
        memory_system: str | None = None,
        custom_prior_knowledge: str | None = None,
    ) -> tuple[list[ErrorAttributionPrediction], dict[str, float | int]]:
        """Run MemTrace attribution asynchronously.

        When a cache directory is configured, each case is loaded from the cache
        if available and otherwise computed and then saved. The returned cost
        statistics only cover the cases actually executed in this run. Cases
        served from the cache are excluded.

        Args:
            failed_cases (`list[FailedQueryCase]`):
                Failed query cases to attribute.
            graphs (`ExecNetwork | list[ExecNetwork]`):
                One shared execution graph or one graph per failed case.
            batch_size (`int | None`, optional):
                Maximum number of attribution agents to run concurrently.
            use_source_evidence_nodes (`bool`, defaults to `False`):
                Whether the real source-evidence nodes should seed graph exploration. 
                When it is enabled, `use_pseudo_source_evidence` is ignored.
            use_pseudo_source_evidence (`bool`, defaults to `True`):
                Whether to retrieve pseudo source-evidence starting points when
                real source evidence is not used. It only takes effect when
                `use_source_evidence_nodes` is `False`.
            memory_system (`str | None`, optional):
                Name of the memory system being diagnosed. If no custom prior
                knowledge is provided, matching built-in prior knowledge will be
                appended to the failure attribution instructions. If `custom_prior_knowledge` 
                is provided, it will be ignored.
            custom_prior_knowledge (`str | None`, optional):
                Custom prior knowledge appended to the failure attribution instructions.

        Returns:
            `tuple[list[ErrorAttributionPrediction], dict[str, float | int]]`:
                Attribution predictions and aggregate cost statistics.
        """
        if not failed_cases:
            raise ValueError("No failed cases need to be attributed.")

        api_pool = self._api_pool
        batch_size = batch_size or 1
        if batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if batch_size > api_pool.size:
            warnings.warn(
                "`batch_size` is greater than the available API credential slots.",
                UserWarning,
            )
        effective_batch_size = batch_size 

        if isinstance(graphs, ExecNetwork):
            case_graphs = [graphs] * len(failed_cases)
        elif len(graphs) != len(failed_cases):
            raise ValueError(
                "`graphs` must be a single `ExecNetwork`, "
                "or a list with the same length as `failed_cases`."
            )
        else:
            case_graphs = list(graphs)

        semaphore = asyncio.Semaphore(effective_batch_size)

        async def run_single_case(
            case_index: int,
            case: FailedQueryCase,
            graph: ExecNetwork,
        ) -> tuple[ErrorAttributionPrediction, float, bool]:
            """Run MemTrace attribution for one failed query case.

            Args:
                case_index (`int`):
                    Index of the case in the batch.
                case (`FailedQueryCase`):
                    Failed query case to attribute.
                graph (`ExecNetwork`):
                    Execution graph containing the failed query.

            Returns:
                `tuple[ErrorAttributionPrediction, float, bool]`:
                    The structured attribution prediction, the elapsed seconds,
                    and whether the case was executed in this run or served from 
                    the cache.
            """
            cache_identifier = {
                "graph_id": graph.graph_id,
                # The full case content (excluding the annotation metadata) is
                # used so that two cases sharing the same query node identifier but
                # differing in prediction, golden answer, or source evidence do
                # not collide in the cache.
                "case": case.model_dump(exclude={"metadata"}),
                # The configuration fields that change the attribution result and therefore
                # take part in the cache key. Infrastructure fields such as credentials, the
                # studio URL, the batch size, and the cache directory itself are excluded.
                "config": {
                    field: getattr(self.config, field)
                    for field in (
                        "model",
                        "temperature",
                        "exploration_strategy",
                        "context_window",
                        "max_context_limit",
                        "keep_recent",
                        "max_trace_nodes",
                        "full_log_token_limit", 
                        "max_iters",
                        "embedding_model_name",
                        "embedding_dimensions",
                        "retrieval_type",
                        "starting_point_query_type",
                        "num_starting_points",
                        "candidate_multiplier",
                    )
                },
                "arun_args": {
                    "use_source_evidence_nodes": use_source_evidence_nodes,
                    "use_pseudo_source_evidence": use_pseudo_source_evidence,
                    "memory_system": memory_system,
                    "custom_prior_knowledge": custom_prior_knowledge,
                },
            }
            
            cached = self._load_cached_case(cache_identifier)
            if cached is not None:
                return (
                    ErrorAttributionPrediction.model_validate(
                        cached["error_attribution_result"]
                    ),
                    cached["time"],
                    False,
                )

            async with semaphore:
                started = time.perf_counter() 
                query_node = graph.get_variable(case.query_full_node_id)
                is_pseudo_source_evidence = False
                if use_source_evidence_nodes:
                    source_evidence_nodes = [
                        graph.get_variable(full_node_id)
                        for full_node_id in case.source_evidence_full_node_ids
                    ]
                elif use_pseudo_source_evidence:
                    source_evidence_nodes = await self._retrieve_pseudo_source_evidence(
                        graph=graph,
                        case=case,
                        case_index=case_index,
                    )
                    is_pseudo_source_evidence = True
                else:
                    source_evidence_nodes = None

                api_key, base_url = api_pool.credential_for(case_index)
                execution_trace = None
                if self.config.exploration_strategy == "operation_block_search":
                    notebook = CCTraceNotebook.from_graph(
                        graph,
                        source_evidence_nodes=source_evidence_nodes,
                    )
                    agent = self._build_obs_agent(
                        notebook=notebook,
                        api_key=api_key,
                        base_url=base_url,
                    )
                elif self.config.exploration_strategy == "long_context":
                    execution_trace = flatten_execution_graph(graph, token_limit=self.config.full_log_token_limit)
                    agent = self._build_long_context_agent(api_key=api_key, base_url=base_url)
                else:
                    notebook = GraphTraceNotebook(
                        graph,
                        max_trace_nodes=self.config.max_trace_nodes,
                        include_metadata=False,
                    )

                    if source_evidence_nodes is not None:
                        initial_full_node_ids = [query_node.full_node_id]
                        initial_full_node_ids.extend(
                            node.full_node_id for node in source_evidence_nodes
                        )
                        _ = await notebook.initialize_execution_graph(initial_full_node_ids)

                    agent = self._build_agent(
                        notebook=notebook,
                        api_key=api_key,
                        base_url=base_url,
                    )
                task_msg = Msg(
                    "user",
                    self._build_task_prompt(
                        case=case,
                        query_node=query_node,
                        source_evidence_nodes=source_evidence_nodes,
                        is_pseudo_source_evidence=is_pseudo_source_evidence,
                        memory_system=memory_system,
                        custom_prior_knowledge=custom_prior_knowledge,
                        execution_trace=execution_trace,
                    ),
                    "user",
                )
                reply = await agent(
                    task_msg,
                    structured_model=ErrorAttributionPrediction,
                )
                prediction = ErrorAttributionPrediction.model_validate(reply.metadata)
                elapsed = time.perf_counter() - started
                self._store_cached_case(
                    cache_identifier,
                    {
                        "error_attribution_result": prediction.model_dump(),
                        "time": elapsed,
                    },
                )
                return prediction, elapsed, True

        studio = None
        if effective_batch_size == 1 and self.config.studio_url is not None:
            studio = StudioServer(
                url=self.config.studio_url,
                project=self.config.project,
            )
            studio.activate()

        ChatUsageTokenMonitor.reset()
        try:
            with agentscope_token_monitor():
                case_results = await asyncio.gather(
                    *[
                        run_single_case(
                            case_index=index,
                            case=case,
                            graph=case_graphs[index],
                        )
                        for index, case in enumerate(failed_cases)
                    ]
                )
        finally:
            if studio is not None:
                studio.deactivate()

        predictions = [
            prediction for prediction, _, _ in case_results
        ]

        # Estimate the costs of the attribution process. Only the cases actually
        # executed in this run (cache misses) are counted, so cases served from
        # the cache do not contribute to the cost.
        executed_case_times = [
            elapsed for _, elapsed, executed in case_results if executed
        ]
        num_executed = len(executed_case_times)
        token_cost = ChatUsageTokenMonitor.to_dict()
        n_events = token_cost["n"]
        avg_input = token_cost["avg_input_tokens"]
        avg_output = token_cost["avg_output_tokens"]

        costs = {
            "average_input_tokens": n_events / max(1, num_executed) * avg_input, 
            "average_output_tokens": n_events / max(1, num_executed) * avg_output,
            "average_minutes": sum(executed_case_times) / max(1, num_executed) / 60,
        }
        ChatUsageTokenMonitor.reset() 

        return predictions, costs

    def run(
        self,
        failed_cases: list[FailedQueryCase],
        graphs: ExecNetwork | list[ExecNetwork],
        batch_size: int | None = None,
        use_source_evidence_nodes: bool = False,
        use_pseudo_source_evidence: bool = True,
        memory_system: str | None = None,
        custom_prior_knowledge: str | None = None,
    ) -> tuple[list[ErrorAttributionPrediction], dict[str, float | int]]:
        """Run MemTrace attribution.

        Args:
            failed_cases (`list[FailedQueryCase]`):
                Failed query cases to attribute.
            graphs (`ExecNetwork | list[ExecNetwork]`):
                One shared execution graph or one graph per failed case.
            batch_size (`int | None`, optional):
                Maximum number of attribution agents to run concurrently.
            use_source_evidence_nodes (`bool`, defaults to `False`):
                Whether the real source-evidence nodes should seed graph exploration. 
                When it is enabled, `use_pseudo_source_evidence` is ignored.
            use_pseudo_source_evidence (`bool`, defaults to `True`):
                Whether to retrieve pseudo source-evidence starting points when
                real source evidence is not used. It only takes effect when
                `use_source_evidence_nodes` is `False`.
            memory_system (`str | None`, optional):
                Name of the memory system being diagnosed. If no custom prior
                knowledge is provided, matching built-in prior knowledge will be
                appended to the failure attribution instructions. If `custom_prior_knowledge` 
                is provided, it will be ignored.
            custom_prior_knowledge (`str | None`, optional):
                Custom prior knowledge appended to the attribution instructions.

        Returns:
            `tuple[list[ErrorAttributionPrediction], dict[str, float | int]]`:
                Attribution predictions and aggregate cost statistics.
        """
        try:
            # Check whether there is an active running event loop. 
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    failed_cases=failed_cases,
                    graphs=graphs,
                    batch_size=batch_size,
                    use_source_evidence_nodes=use_source_evidence_nodes,
                    use_pseudo_source_evidence=use_pseudo_source_evidence,
                    memory_system=memory_system,
                    custom_prior_knowledge=custom_prior_knowledge,
                )
            )

        raise RuntimeError(
            "`MemTraceRunner(...).run(...)` cannot be called from a running event loop. "
            "Use `await MemTraceRunner(...).arun(...)` instead."
        )
