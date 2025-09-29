# app/routers/chat.py - UPDATED VERSION
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm import LLMService
from app.services.conversation_store import ConversationStore
from app.services.retriever import Retriever
from app.services.query_parser import parse_filters, test_parsing  # Import test function
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


# DEBUG ENDPOINT - Remove in production
@router.get("/debug/test-parsing")
async def test_parsing_endpoint():
    """Debug endpoint to test query parsing"""
    test_queries = [
        "3 bedroom villa in New Cairo",
        "show me a 2 bedroom apartment",
        "4 bed house under 2M EGP",
        "villa between 1.5M and 3M",
        "apartment under 500k",
        "over 1M EGP villa",
        "3BR 2BA house",
        "compare property 1 and 2",
        "what's better between these"
    ]

    results = {}
    for query in test_queries:
        try:
            parsed = parse_filters(query)
            results[query] = {
                "parsed_filters": parsed,
                "status": "success"
            }
        except Exception as e:
            results[query] = {
                "error": str(e),
                "status": "error"
            }

    return {
        "test_results": results,
        "groq_model": llm_service.provider,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/chat", response_model=ChatResponseModel)
async def chat_endpoint(request: ChatRequest):
    try:
        session_id = request.session_id or conversation_store.start_session()

        # Debug logging
        logger.info(f"Processing query: '{request.query}' for session: {session_id}")

        # 1) Store user message immediately
        conversation_store.add_message(session_id, "user", request.query)

        # 2) Retrieve recent history
        history = conversation_store.get_recent(session_id, n=6)

        # 3) Check if this is a comparison query
        from app.services.llm import is_comparison_query
        is_comparing = is_comparison_query(request.query)

        if is_comparing:
            # For comparison, use the LAST search results, not new search
            logger.info("Comparison query detected - using last search results")
            docs = conversation_store.get_last_search_results(session_id)

            if not docs or len(docs) < 2:
                # No previous search or not enough properties
                response = {
                    "answer": "I don't have any previous search results to compare. Please search for properties first, then ask me to compare them.\n\nExample:\n1. 'Show me villas in New Cairo'\n2. 'Compare property 1 and 3'",
                    "hits": [],
                    "cards": [],
                    "insights": {}
                }
                conversation_store.add_message(session_id, "assistant", response['answer'])

                return ChatResponseModel(
                    session_id=session_id,
                    answer=response['answer'],
                    hits=[],
                    query_id=str(hash(f"{session_id}{request.query}{datetime.utcnow().timestamp()}")),
                    timestamp=datetime.utcnow().isoformat()
                )

            logger.info(f"Using {len(docs)} properties from last search for comparison")
        else:
            # Regular search - parse filters and retrieve new docs
            try:
                filters = parse_filters(request.query)
                logger.info(f"Parsed filters: {filters}")
            except Exception as e:
                logger.error(f"Filter parsing error: {e}")
                filters = {}

            # 4) Retrieve docs with error handling
            try:
                docs = retriever.search(request.query, k=request.k, filters=filters, session_history=history)
                logger.info(f"Retrieved {len(docs)} documents")

                # Store these results for potential future comparison
                conversation_store.store_search_results(session_id, docs)
                logger.info(f"Stored search results for session {session_id}")

            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                docs = []

        # 5) Generate response with enhanced error handling
        try:
            response = llm_service.generate_response(request.query, docs, history)
            logger.info(f"Generated response type: {type(response)}")
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            # Fallback response
            response = {
                "answer": "I apologize, but I encountered an error processing your request. Please try rephrasing your query or contact support if the issue persists.",
                "hits": docs,
                "cards": [],
                "insights": {}
            }

        # 6) Normalize response
        if isinstance(response, dict):
            answer = str(response.get('answer', ''))
            hits = response.get('hits', docs)
            cards = response.get('cards', [])
            insights = response.get('insights', {})
            comparison = response.get('comparison', None)
        else:
            answer = str(response)
            hits = docs
            cards = []
            insights = {}
            comparison = None

        # 7) Store assistant reply
        conversation_store.add_message(session_id, "assistant", answer)

        # 8) Convert hits to schema-friendly format
        out_hits = []
        for h in hits:
            try:
                # Handle both old and new hit formats
                if isinstance(h, dict) and 'metadata' in h:
                    out_hits.append(HitModel(
                        id=str(h.get('id', '')),
                        score=float(h.get('score', 0.0)),
                        document=str(h.get('document', '')),
                        metadata=h.get('metadata', {})
                    ))
                else:
                    # Fallback for unexpected formats
                    out_hits.append({
                        'id': str(h.get('id', 'unknown')),
                        'score': float(h.get('score', 0.0)),
                        'document': str(h.get('document', '')),
                        'metadata': h.get('metadata', {})
                    })
            except Exception as e:
                logger.warning(f"Hit conversion error: {e}")
                # Keep raw hit as fallback
                out_hits.append(h)

        # 9) Build final response
        final_response = ChatResponseModel(
            session_id=session_id,
            answer=answer,
            hits=out_hits,
            query_id=str(hash(f"{session_id}{request.query}{datetime.utcnow().timestamp()}")),
            timestamp=datetime.utcnow().isoformat()
        )

        # Add debug info if insights or comparison available
        if insights:
            logger.info(f"Response insights: {insights}")
        if comparison:
            logger.info(f"Comparison performed: {len(comparison.get('properties', []))} properties")

        return final_response

    except Exception as e:
        logger.exception("Critical error in chat endpoint")

        # Emergency fallback response
        emergency_response = ChatResponseModel(
            session_id=request.session_id or "error",
            answer="I'm experiencing technical difficulties. Please try again in a moment. If the problem persists, try:\n• Simplifying your query\n• Using basic terms like 'apartment', 'villa', 'New Cairo'\n• Avoiding special characters",
            hits=[],
            query_id=str(hash(f"error{datetime.utcnow().isoformat()}")),
            timestamp=datetime.utcnow().isoformat()
        )

        return emergency_response


# Health check for the chat service
@router.get("/chat/health")
async def chat_health():
    """Health check specifically for chat functionality"""
    try:
        # Test query parsing
        test_query = "3 bedroom villa"
        parsed = parse_filters(test_query)

        # Test LLM service initialization
        llm_ready = llm_service.provider is not None

        # Test conversation store
        test_session = conversation_store.start_session()

        return {
            "status": "healthy",
            "chat_service": "operational",
            "query_parser": "working" if parsed else "error",
            "llm_provider": llm_service.provider,
            "llm_ready": llm_ready,
            "conversation_store": "working",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }