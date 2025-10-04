# app/services/conversation_store.py
import shelve
import uuid
import time
import os
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

MAX_HISTORY_SIZE = 100
MAX_STORED_RESULTS = 20


class ConversationStore:
    def __init__(self, backend: str = "shelve", path: str = "./conversation_history.db"):
        self.backend = backend
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) if os.path.dirname(path) else '.', exist_ok=True)

    def _get_shelve(self):
        return shelve.open(self.path, writeback=True)

    def start_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._get_shelve() as db:
            db[session_id] = {
                'messages': [],
                'last_search_results': [],
                'last_search_query': None,
                'created_at': time.time()
            }
        logger.info(f"Started session: {session_id}")
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        with self._get_shelve() as db:
            if session_id not in db:
                db[session_id] = {
                    'messages': [],
                    'last_search_results': [],
                    'last_search_query': None,
                    'created_at': time.time()
                }

            if isinstance(db[session_id], list):
                db[session_id] = {
                    'messages': db[session_id],
                    'last_search_results': [],
                    'last_search_query': None,
                    'created_at': time.time()
                }

            db[session_id]['messages'].append({"role": role, "content": content})

            if len(db[session_id]['messages']) > MAX_HISTORY_SIZE:
                db[session_id]['messages'] = db[session_id]['messages'][-MAX_HISTORY_SIZE:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._get_shelve() as db:
            session_data = db.get(session_id, {})

            if isinstance(session_data, list):
                return session_data

            return session_data.get('messages', [])

    def get_recent(self, session_id: str, n: int = 6) -> List[Dict[str, str]]:
        history = self.get_history(session_id)
        return history[-n:]

    def get_last_user_message(self, session_id: str) -> str:
        history = self.get_history(session_id)
        for m in reversed(history):
            if m.get('role') == 'user':
                return m.get('content', '')
        return ''

    def store_search_results(self, session_id: str, results: List[Dict[str, Any]], query: Optional[str] = None):
        if not results or len(results) == 0:
            logger.info(f"Skipping empty results for {session_id}")
            return

        with self._get_shelve() as db:
            if session_id not in db:
                db[session_id] = {
                    'messages': [],
                    'last_search_results': [],
                    'last_search_query': None,
                    'created_at': time.time()
                }

            if isinstance(db[session_id], list):
                db[session_id] = {
                    'messages': db[session_id],
                    'last_search_results': [],
                    'last_search_query': None,
                    'created_at': time.time()
                }

            db[session_id]['last_search_results'] = results[:MAX_STORED_RESULTS]
            db[session_id]['last_search_query'] = query

            logger.info(f"Stored {len(results[:MAX_STORED_RESULTS])} results for {session_id}")

    def get_last_search_results(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_shelve() as db:
            session_data = db.get(session_id, {})

            if isinstance(session_data, list):
                return []

            results = session_data.get('last_search_results', [])
            logger.info(f"Retrieved {len(results)} stored results for {session_id}")
            return results

    def get_last_search_query(self, session_id: str) -> Optional[str]:
        with self._get_shelve() as db:
            session_data = db.get(session_id, {})

            if isinstance(session_data, dict):
                return session_data.get('last_search_query')

            return None