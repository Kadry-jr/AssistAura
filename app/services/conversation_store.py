# app/services/conversation_store.py
import shelve
import uuid
from typing import List, Dict, Optional, Any


class ConversationStore:
    def __init__(self, backend: str = "shelve", path: str = "./conversation_history.db"):
        self.backend = backend
        self.path = path

    def _get_shelve(self):
        return shelve.open(self.path, writeback=True)

    def start_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._get_shelve() as db:
            db[session_id] = {
                'messages': [],
                'last_search_results': []  # Store last search results
            }
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        with self._get_shelve() as db:
            if session_id not in db:
                db[session_id] = {'messages': [], 'last_search_results': []}

            # Handle old format (just a list) by converting to new format
            if isinstance(db[session_id], list):
                db[session_id] = {
                    'messages': db[session_id],
                    'last_search_results': []
                }

            db[session_id]['messages'].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._get_shelve() as db:
            session_data = db.get(session_id, {})

            # Handle old format
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

    def store_search_results(self, session_id: str, results: List[Dict[str, Any]]):
        """Store the last search results for potential comparison"""
        with self._get_shelve() as db:
            if session_id not in db:
                db[session_id] = {'messages': [], 'last_search_results': []}

            # Handle old format
            if isinstance(db[session_id], list):
                db[session_id] = {
                    'messages': db[session_id],
                    'last_search_results': []
                }

            db[session_id]['last_search_results'] = results

    def get_last_search_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve the last search results for comparison"""
        with self._get_shelve() as db:
            session_data = db.get(session_id, {})

            # Handle old format
            if isinstance(session_data, list):
                return []

            return session_data.get('last_search_results', [])