import pickle
from pathlib import Path
import numpy as np
import jieba
from memory_layer.prompts import CURRENT_LANGUAGE
from online_memory.config import EmbeddingConfig
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple, 
    Union,
)


class InMemoryIndexManager:
    """
    In-memory index manager for BM25 and embedding-based retrieval.
    
    Features:
    - BM25 index for lexical matching (atomic fact preferred)
    - Embedding index for semantic matching (MaxSim strategy for atomic facts)
    - Incremental updates (add documents one by one)
    - Persistence support (save/load)
    
    MaxSim Strategy:
        Each atomic fact in event log gets its own embedding vector.
        During search, the maximum similarity across all atomic fact embeddings
        is used as the document score. This precisely matches specific facts
        and avoids semantic dilution from averaging.
    
    Usage:
        ```python
        manager = InMemoryIndexManager(embedding_config)
        
        # Add documents
        await manager.add_memcell(memcell)
        
        # Search
        bm25_results = manager.search_bm25(query, top_n=10)
        emb_results = await manager.search_embedding(query, top_n=10)
        ```
    """
    
    def __init__(
        self,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> None:
        """Initialize the index manager."""
        self.embedding_config = embedding_config or EmbeddingConfig()
        
        # Document storage
        self._documents: List[Dict[str, Any]] = []
        
        # BM25 components
        self._bm25 = None
        self._corpus: List[List[str]] = []
        self._stemmer = None
        self._stop_words: set = set()
        
        # Embedding storage (MaxSim strategy: each doc can have multiple embeddings)
        # Structure: List of {"atomic_facts": [emb1, emb2, ...], "subject": emb, "summary": emb, "episode": emb}
        self._embeddings: List[Dict[str, Union[List[np.ndarray], np.ndarray]]] = []
        
        # Ensure NLTK data
        self._ensure_nltk_data()

        # Initialize vectorize service
        self._init_vectorize_service()
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is available."""
        # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage2_index_building.py#L21
        if CURRENT_LANGUAGE == "zh":
            return

        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import PorterStemmer
            # Download required data
            for resource in ["punkt", "punkt_tab", "stopwords"]:
                try:
                    nltk.data.find(
                        f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
                    )
                except LookupError:
                    nltk.download(resource, quiet=True)
            
            self._stemmer = PorterStemmer()
            self._stop_words = set(stopwords.words("english"))
        except Exception as e:
            print(f"Warning: NLTK initialization failed: {e}")
            self._stemmer = None
            self._stop_words = set()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage2_index_building.py#L97
        if not text:
            return []

        if CURRENT_LANGUAGE == "zh":
            return list(jieba.cut(text))
        
        try:
            from nltk.tokenize import word_tokenize
            
            tokens = word_tokenize(text.lower())
            
            if self._stemmer:
                processed = [
                    self._stemmer.stem(t)
                    for t in tokens
                    if t.isalpha() and len(t) >= 2 and t not in self._stop_words
                ]
                return processed
            else:
                return [t for t in tokens if t.isalpha() and len(t) >= 2]
        except Exception:
            # Fallback to simple tokenization
            return [w.lower() for w in text.split() if len(w) >= 2]
    
    def _build_searchable_text(self, doc: Dict[str, Any]) -> str:
        """Build searchable text from document with weighted fields."""
        # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage2_index_building.py#L53
        parts = []
        
        # Prefer event_log's atomic_fact (if exists)
        event_log = doc.get("event_log")
        if event_log and isinstance(event_log, dict):
            atomic_facts = event_log.get("atomic_fact", [])
            if isinstance(atomic_facts, list) and atomic_facts:
                for fact in atomic_facts:
                    if isinstance(fact, dict) and "fact" in fact:
                        # New format: {"fact": "...", "embedding": [...]}
                        if fact["fact"]:
                            parts.append(str(fact["fact"]))
                    elif isinstance(fact, str) and fact:
                        # Old format: pure string
                        parts.append(str(fact))
                if parts:
                    return " ".join(parts)
        
        # Fall back to original fields (maintain backward compatibility)
        # Title has highest weight (repeat 3 times)
        subject = doc.get("subject", "")
        if subject:
            parts.extend([str(subject)] * 3)
        
        # Summary (repeat 2 times)
        summary = doc.get("summary", "")
        if summary:
            parts.extend([str(summary)] * 2)
        
        # Content
        episode = doc.get("episode", "")
        if episode:
            parts.append(str(episode))
        
        return " ".join(p for p in parts if p)
    
    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from corpus."""
        if not self._corpus:
            self._bm25 = None
            return
        
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus)
        except ImportError:
            print("Warning: rank_bm25 not installed. BM25 search will be unavailable.")
            self._bm25 = None
    
    def _init_vectorize_service(self) -> None:
        """Initialize the vectorize service from EmbeddingConfig."""        
        from agentic_layer.vectorize_service import (
            HybridVectorizeService, 
            HybridVectorizeConfig, 
        )
        # Create `HybridVectorizeConfig` from input parameters.
        # We don't use fallback here.
        vectorize_config = HybridVectorizeConfig(
            primary_provider=self.embedding_config.provider,
            primary_api_key=self.embedding_config.api_key or "",
            primary_base_url=self.embedding_config.base_url or "",
            model=self.embedding_config.model or "",
            dimensions=self.embedding_config.embedding_dims,
            enable_fallback=False, 
        )
        
        self._vectorize_service = HybridVectorizeService(config=vectorize_config)
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        if not text or not text.strip():
            raise ValueError("Text is empty or whitespace.")
        
        vec = await self._vectorize_service.get_embedding(text)
        return np.array(vec, dtype=np.float32) 
    
    async def add_memcell(self, memcell: Any) -> None:
        """Add a MemCell to the index."""
        # Convert to dict if needed
        if hasattr(memcell, "to_dict"):
            doc = memcell.to_dict()
        else:
            doc = dict(memcell)
        
        # Store document
        self._documents.append(doc)
        
        # Build searchable text and tokenize for BM25
        searchable_text = self._build_searchable_text(doc)
        tokens = self._tokenize(searchable_text)
        self._corpus.append(tokens)
        
        # Rebuild BM25 index
        self._rebuild_bm25()
        
        # Build embeddings 
        doc_embeddings: Dict[str, Union[List[np.ndarray], np.ndarray]] = {}
        
        # Check for event_log atomic_facts
        # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage2_index_building.py#L209
        event_log = doc.get("event_log")
        if event_log and isinstance(event_log, dict):
            atomic_facts = event_log.get("atomic_fact", [])
            if isinstance(atomic_facts, list) and atomic_facts:
                # Get embedding for each atomic fact (single inference)
                valid_embeddings = []
                for fact in atomic_facts:
                    if isinstance(fact, dict) and "fact" in fact:
                        fact_text = fact.get("fact", "")
                    elif isinstance(fact, str):
                        fact_text = fact
                    else:
                        fact_text = ""
                    
                    if fact_text and fact_text.strip():
                        emb = await self._get_embedding(fact_text)
                        valid_embeddings.append(emb)
                
                if valid_embeddings:
                    doc_embeddings["atomic_facts"] = valid_embeddings
        
        # If no atomic facts, fall back to original fields
        if "atomic_facts" not in doc_embeddings:
            for field in ["subject", "summary", "episode"]:
                text = doc.get(field)
                if text and str(text).strip():
                    emb = await self._get_embedding(str(text))
                    doc_embeddings[field] = emb
        
        self._embeddings.append(doc_embeddings)
    
    def search_bm25(
        self,
        query: str,
        top_n: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search using BM25.

        See https://github.com/EverMind-AI/EverMemOS/blob/main/evaluation/src/adapters/evermemos/stage2_index_building.py#L115. 
        
        Parameters
        ----------
        query : str
            Search query.
        top_n : int
            Number of results to return.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples.
        """
        if not self._bm25 or not self._documents:
            print("Warning: BM25 index or documents are not available.")
            return []
        
        tokens = self._tokenize(query)
        if not tokens:
            print("Warning: Query is empty after tokenization.") 
            print(query)
            return []
        
        scores = self._bm25.get_scores(tokens)
        
        # Get top-n indices
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append((self._documents[idx], float(scores[idx])))
        
        return results
    
    def _compute_maxsim_score(
        self, 
        query_emb: np.ndarray, 
        atomic_fact_embs: List[np.ndarray],
    ) -> float:
        """
        Compute maximum similarity between query and multiple atomic fact embeddings.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L95. 
        
        Parameters
        ----------
        query_emb : np.ndarray
            Query embedding vector (1D numpy array).
        atomic_fact_embs : List[np.ndarray]
            List of atomic_fact embedding vectors.
        
        Returns
        -------
        float
            Maximum similarity score (range [-1, 1], typically [0, 1]).
        """
        if not atomic_fact_embs:
            return 0.0
        
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return 0.0
        
        # Optimization: use matrix operations instead of loops (2-3x speedup)
        try:
            # Convert list to matrix: shape = (n_facts, embedding_dim)
            fact_matrix = np.array(atomic_fact_embs)
            
            # Batch compute norms for all facts
            fact_norms = np.linalg.norm(fact_matrix, axis=1)
            
            # Filter zero vectors
            valid_mask = fact_norms > 0
            if not np.any(valid_mask):
                return 0.0
            
            # Vectorized computation of all similarities
            # sims = (fact_matrix @ query_emb) / (query_norm * fact_norms)
            dot_products = np.dot(fact_matrix[valid_mask], query_emb)
            sims = dot_products / (query_norm * fact_norms[valid_mask])
            
            # Return maximum similarity
            return float(np.max(sims))
        
        except Exception:
            # Fall back to loop method (compatibility guarantee)
            similarities = []
            for fact_emb in atomic_fact_embs:
                fact_norm = np.linalg.norm(fact_emb)
                if fact_norm == 0:
                    continue
                sim = np.dot(query_emb, fact_emb) / (query_norm * fact_norm)
                similarities.append(sim)
            return max(similarities) if similarities else 0.0
    
    async def search_embedding(
        self,
        query: str,
        top_n: int = 10,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Execute embedding retrieval using MaxSim strategy.

        See https://github.com/EverMind-AI/EverMemOS/blob/main/evaluation/src/adapters/evermemos/stage2_index_building.py#L171.
    
        Parameters
        ----------
        query : str
            Search query.
        top_n : int
            Number of results to return.
        query_embedding : np.ndarray, optional
            Pre-computed query embedding.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples.
        """
        if not self._documents or not self._embeddings:
            print("Warning: Documents or embeddings are not available.")
            return []
        
        # Get query embedding
        if query_embedding is None:
            query_embedding = await self._get_embedding(query)
        
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        # Compute MaxSim scores for each document
        scores = []
        for doc_emb in self._embeddings:
            if not doc_emb:
                scores.append(0.0)
                continue
            
            max_sim = 0.0
            
            # Prefer atomic_facts (MaxSim strategy with matrix optimization)
            if "atomic_facts" in doc_emb and doc_emb["atomic_facts"]:
                max_sim = self._compute_maxsim_score(
                    query_embedding, 
                    doc_emb["atomic_facts"],
                )
            else:
                # Fall back to traditional fields (also use MaxSim: take maximum)
                field_scores = []
                for field in ["subject", "summary", "episode"]:
                    if field in doc_emb and doc_emb[field]:
                        emb = doc_emb[field]
                        emb_norm = np.linalg.norm(emb)
                        if emb_norm > 0:
                            sim = float(np.dot(query_embedding, emb) / (query_norm * emb_norm))
                            field_scores.append(sim)
                
                if field_scores:
                    max_sim = max(field_scores)
            
            scores.append(max_sim)
        
        # Get top-n indices
        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        results = []
        for idx in top_indices:
            results.append((self._documents[idx], float(scores[idx])))
        
        return results
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all indexed documents."""
        return self._documents.copy()
    
    def __len__(self) -> int:
        """Return number of indexed documents."""
        return len(self._documents)
    
    async def save(self, save_path: Path) -> None:
        """Save index to disk as a single pkl file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "documents": self._documents,
            "corpus": self._corpus,
            "embeddings": self._embeddings,
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
    
    async def load(self, save_path: Path) -> bool:
        """Load index from disk."""
        save_path = Path(save_path)
        
        if not save_path.exists():
            return False
        
        try:
            with open(save_path, "rb") as f:
                state = pickle.load(f)
            
            self._documents = state["documents"]
            self._corpus = state["corpus"]
            self._embeddings = state["embeddings"]
            self._rebuild_bm25()
            
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False