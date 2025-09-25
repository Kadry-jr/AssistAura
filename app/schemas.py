from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class Hit(BaseModel):
    id: str
    score: float
    document: str
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    hits: List[Any]
    query_id: Optional[str] = None
    timestamp: Optional[str] = None
