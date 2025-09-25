# app/services/conversation_store.py
import shelve
import uuid
from typing import List, Dict, Optional

class ConversationStore:
    def __init__(self, backend: str = "shelve", path: str = "./conversation_history.db"):
        self.backend = backend
        self.path = path

    def _get_shelve(self):
        return shelve.open(self.path, writeback=True)

    def start_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self._get_shelve() as db:
            db[session_id] = []
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        with self._get_shelve() as db:
            if session_id not in db:
                db[session_id] = []
            db[session_id].append({"role": role, "content": content})

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._get_shelve() as db:
            return db.get(session_id, [])

    def get_recent(self, session_id: str, n: int = 6) -> List[Dict[str, str]]:
        history = self.get_history(session_id)
        return history[-n:]

    def get_last_user_message(self, session_id: str) -> str:
        history = self.get_history(session_id)
        for m in reversed(history):
            if m.get('role') == 'user':
                return m.get('content', '')
        return ''
