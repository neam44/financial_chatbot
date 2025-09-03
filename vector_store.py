from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple

class FinancialVectorStore:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        
    def add_documents(self, documents: List[str]):
        if not documents:
            return
            
        new_embeddings = self.encoder.encode(documents, convert_to_tensor=False)
        self.embeddings.extend(new_embeddings)
        self.documents.extend(documents)
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)
        
        # Simple similarity search
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding[0], doc_embedding)
            similarities.append((self.documents[i], float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]