import os
import chromadb
from typing import List, Optional, Dict, Any

from chromadb import QueryResult


class ChromaVectorStore:
    def __init__(self, persist_dir: str = './chroma_db'):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name='properties',
            metadata={"hnsw:space": "cosine"}
        )

    @classmethod
    def persist_dir(cls, d: str):
        return cls(persist_dir=d)

    def upsert(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], documents: List[str]):
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embedding: List[float], k: int = 5, where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Query the vector store for similar items.
        
        Args:
            query_embedding: The embedding vector to query with
            k: Number of results to return
            where: Optional filter dictionary for metadata
            
        Returns:
            Dictionary containing query results with 'ids', 'documents', 'metadatas', and 'distances'
        """
        try:
            # Convert the where clause to Chroma's format if needed
            filter_dict = None
            if where:
                filter_dict = where  # Chroma expects the filter as-is
                
            # Perform the query with only valid include parameters
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Add IDs separately since they're not in the include list
            if results.get('documents'):
                results['ids'] = [[str(i) for i in range(len(results['documents'][0]))]]
            
            return results
            
        except Exception as e:
            error_msg = f"Error querying vector store: {str(e)}"
            raise ValueError(error_msg) from e