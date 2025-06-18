from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Any

class VectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def build_index(self, chunks: List[Any]):
        """Build FAISS index from chunks"""
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        return self.index
    
    def search(self, query: str, chunks: List[Any], k: int = 5) -> List[Any]:
        """Search for relevant chunks"""
        # Build index if needed
        if self.index is None or self.chunks != chunks:
            self.build_index(chunks)
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Return relevant chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        return relevant_chunks
    
    def get_similarity_scores(self, query: str, chunks: List[Any]) -> List[float]:
        """Get similarity scores for all chunks"""
        if self.index is None or self.chunks != chunks:
            self.build_index(chunks)
        
        query_embedding = self.encoder.encode([query])
        distances, _ = self.index.search(query_embedding, len(chunks))
        
        # Convert distances to similarity scores (0-1)
        similarities = 1 / (1 + distances[0])
        return similarities.tolist()