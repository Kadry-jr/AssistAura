# app/services/conversation_store.py
import shelve
import uuid
import time
import os
import threading
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta


class ConversationStoreError(Exception):
    """Base exception for ConversationStore errors"""
    pass


class ConversationStore:
    def __init__(self, backend: str = "shelve", path: str = "./conversation_history.db"):
        """Initialize the conversation store.
        
        Args:
            backend: Storage backend (currently only 'shelve' is supported)
            path: Path to the storage file
            
        Raises:
            ConversationStoreError: If initialization fails
        """
        self.backend = backend
        self.path = path
        self._lock = threading.RLock()  # Thread safety lock
        
        # Ensure the directory exists
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        except OSError as e:
            raise ConversationStoreError(f"Failed to create storage directory: {e}")

    def _get_shelve(self):
        """Get a thread-safe shelve instance.
        
        Returns:
            shelve.DbfilenameShelf: An open shelve instance
            
        Raises:
            ConversationStoreError: If the shelve cannot be opened
        """
        try:
            # Using writeback=False for better performance and thread safety
            return shelve.open(self.path, writeback=False)
        except Exception as e:
            raise ConversationStoreError(f"Failed to open shelve database: {e}")

    def _validate_session_id(self, session_id: str) -> None:
        """Validate session_id format.
        
        Args:
            session_id: The session ID to validate
            
        Raises:
            ValueError: If session_id is invalid
        """
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise ValueError("session_id must be a valid UUID")

    def start_session(self) -> str:
        """Start a new conversation session.
        
        Returns:
            str: The new session ID
            
        Raises:
            ConversationStoreError: If the session cannot be created
        """
        with self._lock:
            try:
                session_id = str(uuid.uuid4())
                with self._get_shelve() as db:
                    db[session_id] = {
                        'messages': [],
                        'last_search_results': [],
                        'created_at': time.time(),
                        'last_accessed': time.time()
                    }
                return session_id
            except Exception as e:
                raise ConversationStoreError(f"Failed to start session: {e}")

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation.
        
        Args:
            session_id: The session ID
            role: The role of the message sender (e.g., 'user', 'assistant')
            content: The message content
            
        Raises:
            ValueError: If inputs are invalid
            ConversationStoreError: If the message cannot be added
        """
        self._validate_session_id(session_id)
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")

        with self._lock:
            try:
                with self._get_shelve() as db:
                    if session_id not in db:
                        db[session_id] = {
                            'messages': [],
                            'last_search_results': [],
                            'created_at': time.time()
                        }

                    # Handle old format (just a list) by converting to new format
                    if isinstance(db[session_id], list):
                        db[session_id] = {
                            'messages': db[session_id],
                            'last_search_results': [],
                            'created_at': time.time()
                        }

                    db[session_id]['messages'].append({
                        "role": role,
                        "content": content,
                        "timestamp": time.time()
                    })
                    db[session_id]['last_accessed'] = time.time()
            except Exception as e:
                raise ConversationStoreError(f"Failed to add message: {e}")

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List[Dict[str, Any]]: The conversation history
            
        Raises:
            ValueError: If session_id is invalid
            ConversationStoreError: If the history cannot be retrieved
        """
        self._validate_session_id(session_id)
        
        with self._lock:
            try:
                with self._get_shelve() as db:
                    session_data = db.get(session_id, {})
                    
                    # Update last accessed time
                    if session_data and isinstance(session_data, dict):
                        session_data['last_accessed'] = time.time()
                        db[session_id] = session_data
                    
                    # Handle old format
                    if isinstance(session_data, list):
                        return session_data
                        
                    return session_data.get('messages', [])
            except Exception as e:
                raise ConversationStoreError(f"Failed to get history: {e}")

    def get_recent(self, session_id: str, n: int = 6) -> List[Dict[str, Any]]:
        """Get the most recent messages from a session.
        
        Args:
            session_id: The session ID
            n: Number of recent messages to return
            
        Returns:
            List[Dict[str, Any]]: The most recent messages
        """
        history = self.get_history(session_id)
        return history[-n:]

    def get_last_user_message(self, session_id: str) -> str:
        """Get the last user message from a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            str: The last user message content, or empty string if not found
        """
        history = self.get_history(session_id)
        for m in reversed(history):
            if m.get('role') == 'user':
                return m.get('content', '')
        return ''

    def store_search_results(self, session_id: str, results: List[Dict[str, Any]]) -> None:
        """Store search results for a session.
        
        Args:
            session_id: The session ID
            results: The search results to store
            
        Raises:
            ValueError: If session_id is invalid or results is not a list
            ConversationStoreError: If results cannot be stored
        """
        self._validate_session_id(session_id)
        if not isinstance(results, list):
            raise ValueError("results must be a list")

        with self._lock:
            try:
                with self._get_shelve() as db:
                    if session_id not in db:
                        db[session_id] = {
                            'messages': [],
                            'last_search_results': [],
                            'created_at': time.time()
                        }

                    # Handle old format
                    if isinstance(db[session_id], list):
                        db[session_id] = {
                            'messages': db[session_id],
                            'last_search_results': [],
                            'created_at': time.time()
                        }

                    db[session_id]['last_search_results'] = results
                    db[session_id]['last_accessed'] = time.time()
            except Exception as e:
                raise ConversationStoreError(f"Failed to store search results: {e}")

    def get_last_search_results(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the last search results for a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            List[Dict[str, Any]]: The last search results, or empty list if none
            
        Raises:
            ValueError: If session_id is invalid
            ConversationStoreError: If results cannot be retrieved
        """
        self._validate_session_id(session_id)
        
        with self._lock:
            try:
                with self._get_shelve() as db:
                    session_data = db.get(session_id, {})
                    
                    # Update last accessed time
                    if session_data and isinstance(session_data, dict):
                        session_data['last_accessed'] = time.time()
                        db[session_id] = session_data
                    
                    # Handle old format
                    if isinstance(session_data, list):
                        return []
                        
                    return session_data.get('last_search_results', [])
            except Exception as e:
                raise ConversationStoreError(f"Failed to get search results: {e}")

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Remove sessions older than the specified number of days.
        
        Args:
            max_age_days: Maximum age in days before a session is considered old
            
        Returns:
            int: Number of sessions removed
            
        Raises:
            ConversationStoreError: If cleanup fails
        """
        if not isinstance(max_age_days, int) or max_age_days < 0:
            raise ValueError("max_age_days must be a non-negative integer")
            
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        with self._lock:
            try:
                with self._get_shelve() as db:
                    session_ids = list(db.keys())
                    for session_id in session_ids:
                        try:
                            session_data = db[session_id]
                            # Skip if it's in the old format or missing timestamps
                            if not isinstance(session_data, dict):
                                continue
                                
                            last_accessed = session_data.get('last_accessed')
                            created_at = session_data.get('created_at', 0)
                            
                            # Use the most recent timestamp available
                            last_activity = max(last_accessed or 0, created_at)
                            
                            if last_activity < cutoff_time:
                                del db[session_id]
                                removed_count += 1
                        except Exception as e:
                            # Log the error but continue with other sessions
                            print(f"Error cleaning up session {session_id}: {e}")
                            continue
                    
                    return removed_count
            except Exception as e:
                raise ConversationStoreError(f"Failed to clean up old sessions: {e}")

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get metadata about a session.
        
        Args:
            session_id: The session ID
            
        Returns:
            Dict[str, Any]: Session metadata including message count and timestamps
            
        Raises:
            ValueError: If session_id is invalid
            ConversationStoreError: If the session info cannot be retrieved
        """
        self._validate_session_id(session_id)
        
        with self._lock:
            try:
                with self._get_shelve() as db:
                    session_data = db.get(session_id, {})
                    
                    # Handle old format
                    if isinstance(session_data, list):
                        return {
                            'message_count': len(session_data),
                            'format': 'legacy',
                            'exists': True
                        }
                    
                    if not session_data:
                        return {'exists': False}
                        
                    return {
                        'exists': True,
                        'message_count': len(session_data.get('messages', [])),
                        'created_at': datetime.fromtimestamp(session_data.get('created_at', 0)).isoformat(),
                        'last_accessed': datetime.fromtimestamp(session_data.get('last_accessed', 0)).isoformat(),
                        'has_search_results': bool(session_data.get('last_search_results', [])),
                        'format': 'current'
                    }
            except Exception as e:
                raise ConversationStoreError(f"Failed to get session info: {e}")