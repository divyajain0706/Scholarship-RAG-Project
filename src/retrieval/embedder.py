import os
import json
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ProductionChunkEmbedder:
    """
    Lightweight, production-grade vector embedding generator optimized for low RAM footprint.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.processed_dir = Path("data/processed")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader to ensure model is only loaded into memory when active."""
        if self._model is None:
            logging.info(f"🧠 Loading Lightweight Vector Model ({self.model_name})...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate_embeddings(self, chunks_file: str = "all_chunks.json") -> bool:
        chunks_path = self.processed_dir / chunks_file
        if not chunks_path.exists():
            logging.error(f"Missing {chunks_path}. Run Phase 1 chunker first!")
            return False

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            logging.warning("all_chunks.json is empty!")
            return False

        logging.info(f"🚀 Generating vector embeddings for {len(chunks)} chunk(s)...")
        texts = [f"{chunk.get('section_title', '')}: {chunk['content']}" for chunk in chunks]
        
        # Batch encode with lightweight footprint
        embeddings = self.model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )

        # Save embeddings array
        embeddings_path = self.processed_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings.astype("float32"))

        logging.info(f"✅ Successfully saved embeddings array shape {embeddings.shape} to {embeddings_path}")
        return True

if __name__ == "__main__":
    embedder = ProductionChunkEmbedder()
    embedder.generate_embeddings()