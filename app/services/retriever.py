"""
Real Estate RAG - Document Retriever Module

This module provides functionality for retrieving relevant property documents based on semantic search queries.
It interfaces with the ChromaDB vector store to perform efficient similarity search.

Key Components:
- Vector store connection and management
- Query processing and embedding generation
- Document retrieval with filtering capabilities

Example Usage:
    from app.services.retriever import Retriever, retrieve
    
    # Using the Retriever class
    retriever = Retriever()
    results = retriever.search("2-bedroom apartment in New Zaid")
    
    # Or using the standalone function
    results = retrieve("2-bedroom apartment in New Zaid",
                      filters={"city": "New Zaid", "bedrooms": 2})
"""
# app/services/retriever.py
from typing import Dict, Any, List, Optional
from app.services.embeddings import get_embedding
from app.services.vector_store import ChromaVectorStore
from app.services.query_parser import parse_filters
import os, logging

vs = ChromaVectorStore.persist_dir(os.getenv('CHROMA_PERSIST_DIR', './chroma_db'))
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store=None):
        self.vector_store = vector_store or vs

    def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, session_history: Optional[List[Dict[str,str]]] = None) -> List[Dict[str, Any]]:
        merged_filters = {}
        if filters:
            merged_filters.update(filters)
        parsed = parse_filters(query)
        merged_filters.update(parsed)
        return retrieve(query, k=k, filters=merged_filters or None, session_history=session_history)

def _match_filter(meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters:
        return True
    for key, requirement in filters.items():
        if key not in meta:
            # allow matching on city/location contained in metadata strings
            if key == 'location':
                loc = meta.get('location', '') or meta.get('city', '')
                if requirement and requirement.lower() not in (loc or '').lower():
                    return False
                else:
                    continue
            return False
        val = meta.get(key)
        # range filter
        if isinstance(requirement, dict):
            if '$gte' in requirement and (val is None or float(val) < float(requirement['$gte'])):
                return False
            if '$lte' in requirement and (val is None or float(val) > float(requirement['$lte'])):
                return False
            if '$eq' in requirement and (val is None or str(val).lower() != str(requirement['$eq']).lower()):
                return False
        else:
            # direct contains/equality
            if val is None:
                return False
            if isinstance(val, str):
                if str(requirement).lower() not in val.lower():
                    return False
            else:
                try:
                    if float(val) != float(requirement):
                        return False
                except:
                    if str(val).lower() != str(requirement).lower():
                        return False
    return True

def retrieve(query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None, session_history: Optional[List[Dict[str,str]]] = None) -> List[Dict[str, Any]]:
    try:
        q_emb = get_embedding(query)

        # request more candidates, we will filter and rerank in Python
        multiplier = 3
        res = vs.query(q_emb, k=k * multiplier if k*multiplier > k else k, where=None)

        hits = []
        docs = res.get('documents', [[]])[0]
        metas = res.get('metadatas', [[]])[0]
        dists = res.get('distances', [[]])[0]
        ids = res.get('ids', [[]])[0] if res.get('ids') else [str(i) for i in range(len(docs))]

        # convert distance -> similarity score
        for doc, meta, dist, id_ in zip(docs, metas, dists, ids):
            try:
                dist = float(dist)
            except:
                dist = 1.0
            similarity = 1.0 / (1.0 + dist)  # higher is closer
            hits.append({
                'id': id_,
                'score': similarity,
                'distance': dist,
                'document': doc,
                'metadata': meta or {}
            })

        # Post-filtering by metadata (safe)
        if filters:
            hits = [h for h in hits if _match_filter(h['metadata'], filters)]

        # Boost by recent conversation context (if provided)
        context_boost_text = ''
        if session_history:
            # pick last user message
            for m in reversed(session_history):
                if m.get('role') == 'user':
                    context_boost_text = (m.get('content') or '').lower()
                    break
        if context_boost_text:
            for h in hits:
                # small boost if last query tokens appear in title or document
                if context_boost_text and (context_boost_text in (h['document'] or '').lower() or context_boost_text in str(h['metadata'].get('title','')).lower()):
                    h['score'] += 0.05

        # Final ranking: score desc
        hits.sort(key=lambda x: x['score'], reverse=True)

        # Keep top-k
        top = hits[:k]

        return top

    except Exception as e:
        logging.error(f"Error in retrieve: {str(e)}", exc_info=True)
        return []
