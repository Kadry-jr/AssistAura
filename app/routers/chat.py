# app/routers/chat.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import LLMService
from app.services.conversation_store import ConversationStore
from app.services.retriever import Retriever
from app.services.query_parser import parse_filters
from app.schemas import Hit as HitModel, ChatResponse as ChatResponseModel
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()
llm_service = LLMService()
conversation_store = ConversationStore()
retriever = Retriever()

class ChatRequest(BaseModel):
    session_id: str = None
    query: str
    k: int = 5

@router.post("/chat", response_model=ChatResponseModel)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or conversation_store.start_session()

        # 1) store user message immediately so history includes it
        conversation_store.add_message(session_id, "user", request.query)

        # 2) retrieve recent history to pass to LLM & retriever
        history = conversation_store.get_recent(session_id, n=6)

        # 3) parse filters from the query
        filters = parse_filters(request.query)

        # 4) retrieve docs (pass session_history for context boosting)
        docs = retriever.search(request.query, k=request.k, filters=filters, session_history=history)

        # 5) generate answer (LLMService will use history)
        response = llm_service.generate_response(request.query, docs, history)

        # normalize response
        if isinstance(response, dict):
            answer = str(response.get('answer', ''))
            hits = response.get('hits', docs)
            cards = response.get('cards', [])
        else:
            answer = str(response)
            hits = docs
            cards = []

        # 6) store assistant reply
        conversation_store.add_message(session_id, "assistant", answer)

        # 7) convert hits into schema-friendly list if you want strict typing
        out_hits = []
        for h in hits:
            try:
                out_hits.append(HitModel(
                    id=str(h.get('id')),
                    score=float(h.get('score', 0.0)),
                    document=h.get('document', ''),
                    metadata=h.get('metadata', {})
                ))
            except Exception:
                # fallback: keep raw
                out_hits.append(h)

        return ChatResponseModel(
            session_id=session_id,
            answer=answer,
            hits=out_hits,
            query_id=str(hash(f"{session_id}{request.query}")),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.exception("Error in chat endpoint")
        return ChatResponseModel(
            session_id=request.session_id or "error",
            answer="Sorry, I encountered an error processing your request. Please try again.",
            hits=[],
            query_id=str(hash(f"error{datetime.utcnow().isoformat()}")),
            timestamp=datetime.utcnow().isoformat()
        )
