import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class HybridVectorStore:
    """
    Production Hybrid Search Engine combining FAISS (Dense Vector) 
    and BM25 (Sparse Keyword) using Reciprocal Rank Fusion (RRF).
    """
    def __init__(
        self, 
        processed_dir: str = "data/processed", 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.processed_dir = Path(processed_dir)
        self.model_name = model_name
        self._model = None
        
        self.chunks: List[Dict[str, Any]] = []
        self.faiss_index: faiss.IndexFlatL2 = None
        self.bm25_index: BM25Okapi = None
        self.is_ready = False

        self.load_index()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def load_index(self) -> bool:
        chunks_path = self.processed_dir / "all_chunks.json"
        embeddings_path = self.processed_dir / "embeddings.npy"

        if not chunks_path.exists() or not embeddings_path.exists():
            logging.warning("Chunks or embeddings missing. Index not loaded.")
            self.is_ready = False
            return False

        # 1. Load Chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 2. Load FAISS Vector Index
        embeddings = np.load(embeddings_path).astype("float32")
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)

        # 3. Load BM25 Keyword Index
        tokenized_corpus = [chunk["content"].lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        self.is_ready = True
        logging.info(f"⚡ Hybrid Store initialized with {len(self.chunks)} chunks.")
        return True

    def search_hybrid(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        if not self.is_ready:
            logging.error("Index is not initialized!")
            return []

        # --- A. FAISS Vector Search ---
        query_vec = self.model.encode([query]).astype("float32")
        _, faiss_indices = self.faiss_index.search(query_vec, top_k * 2)
        faiss_hits = [int(idx) for idx in faiss_indices[0] if idx != -1]

        # --- B. BM25 Keyword Search ---
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_hits = [int(idx) for idx in np.argsort(bm25_scores)[::-1][:top_k * 2]]

        # --- C. Reciprocal Rank Fusion (RRF) ---
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(faiss_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        for rank, idx in enumerate(bm25_hits):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort indices by composite RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            chunk = self.chunks[idx].copy()
            chunk["rrf_score"] = round(rrf_scores[idx], 4)
            results.append(chunk)

        return results