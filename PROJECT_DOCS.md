# realestate-rag Project Documentation

## Table of Contents
- [Project Structure](#project-structure)
- [Key Files](#key-files)

## Project Structure

```
.
└── app/
    └── api/
    └── core/
    └── models/
    └── routers/
    └── services/
└── chroma_db/
    └── a0645b62-2375-4234-82c5-91461ff31b5a/
└── data/
└── scripts/
├── .env (809 bytes)
├── .env.example (649 bytes)
├── Dockerfile (1175 bytes)
├── PROJECT_DOCS.md (37156 bytes)
├── README.md (4609 bytes)
    └── __init__.py (54 bytes)
        └── __init__.py (0 bytes)
        └── __init__.py (0 bytes)
        └── config.py (460 bytes)
        └── database.py (466 bytes)
        └── __init__.py (0 bytes)
        └── chat.py (9469 bytes)
        └── dp.py (216 bytes)
    └── schemas.py (349 bytes)
        └── __init__.py (0 bytes)
        └── comparison.py (7965 bytes)
        └── conversation_store.py (3038 bytes)
        └── db_tasks.py (286 bytes)
        └── embeddings.py (1112 bytes)
        └── llm.py (19902 bytes)
        └── query_parser.py (6632 bytes)
        └── real_estate_keywords.py (3526 bytes)
        └── retriever.py (5641 bytes)
        └── vector_store.py (2286 bytes)
    └── test_fixes.py (5890 bytes)
        └── data_level0.bin (3432448 bytes)
        └── header.bin (100 bytes)
        └── index_metadata.pickle (55266 bytes)
        └── length.bin (8192 bytes)
        └── link_lists.bin (18188 bytes)
    └── chroma.sqlite3 (11751424 bytes)
├── conversation_history.db (204800 bytes)
    └── properties_cleaned.csv (8125269 bytes)
├── main.py (1705 bytes)
├── requirements.txt (239 bytes)
    └── document_project.py (7133 bytes)
    └── gradio_ui.py (14954 bytes)
    └── ingest_to_vectorstore.py (2333 bytes)
```

## Key Files

### `PROJECT_DOCS.md`

```md
# realestate-rag Project Documentation

## Table of Contents
- [Project Structure](#project-structure)
- [Key Files](#key-files)

## Project Structure

```
.gradio/
.idea/
    inspectionProfiles/
app/
    api/
    core/
    models/
    routers/
    services/
chroma_db/
    a0645b62-2375-4234-82c5-91461ff31b5a/
data/
scripts/
.env (809 bytes)
.env.example (649 bytes)
    certificate.pem (1970 bytes)
        profiles_settings.xml (174 bytes)
    misc.xml (318 bytes)
    modules.xml (287 bytes)
    realestate-rag.iml (474 bytes)
    vcs.xml (185 bytes)
    workspace.xml (7697 bytes)
Dockerfile (1175 bytes)
PROJECT_DOCS.md (21355 bytes)
README.md (4609 bytes)
    __init__.py (54 bytes)
        __init__.py (0 bytes)
        __init__.py (0 bytes)
        config.py (460 bytes)
        database.py (466 bytes)
        __init__.py (0 bytes)
        chat.py (9469 bytes)
        dp.py (216 bytes)
    schemas.py (349 bytes)
        __init__.py (0 bytes)
        comparison.py (7965 bytes)
        conversation_store.py (3038 bytes)
        db_tasks.py (286 bytes)
        embeddings.py (1112 bytes)
        llm.py (19902 bytes)
        query_parser.py (6632 bytes)
        real_estate_keywords.py (3526 bytes)
        retriever.py (5641 bytes)
        vector_store.py (2286 bytes)
    test_fixes.py (5890 bytes)
        data_level0.bin (3432448 bytes)
        header.bin (100 bytes)
        index_metadata.pickle (55266 bytes)
        length.bin (8192 bytes)
        link_lists.bin (18188 bytes)
    chroma.sqlite3 (11751424 bytes)
conversation_history.db (204800 bytes)
    properties_cleaned.csv (8125269 bytes)
main.py (1705 bytes)
requirements.txt (239 bytes)
    document_project.py (4350 bytes)
    gradio_ui.py (14954 bytes)
    ingest_to_vectorstore.py (2333 bytes)
```

## Key Files

### `PROJECT_DOCS.md`

```md
# realestate-rag Project Documentation

## Table of Contents
- [Project Structure](#project-structure)
- [Key Files](#key-files)

## Project Structure

```
.idea/
    inspection
... [content truncated]
```

---

### `README.md`

```md
# Assist-Aura - Real Estate RAG Assistant

> Intelligent Property Search and Recommendation System

A FastAPI-based Retrieval-Augmented Generation (RAG) system for real estate property search and recommendation. This application allows users to query property listings using natural language and receive relevant property recommendations based on semantic similarity.

## 🚀 Key Features

- **Natural Language Search**: Find properties using everyday language
- **Conversational AI**: Maintains context across multi-turn conversations
- **Smart Filtering**: Automatically extracts filters from natural language queries
- **Property Comparison**: Compare multiple properties side by side
- **Vector Similarity Search**: Powered by ChromaDB for efficient semantic search
- **Flexible LLM Integration**: Supports multiple LLM providers (OpenAI, Groq, etc.)

## 🛠 Tech Stack

- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **Embeddings**: Local or cloud-based sentence transformers
- **LLM Integration**: OpenAI, Groq, and other compatible providers
- **Session Management**: SQLite-based conversation store

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- [Optional] Docker for containerized deployment

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kadry-jr/AssistAura
   cd AssistAura
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # OR
   source .venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   copy .env.example .env
   ```
   Update the `.env` file with your API keys and settings.
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.


### Configuration

Configure the following environment variables 
... [content truncated]
```

---

### `app\core\config.py`

```py
import os
from typing import Optional

class Settings:
    PROJECT_NAME: str = "RealEstate-RAG"
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./chroma")
    DATA_PATH: str = os.getenv("DATA_PATH", "./data/properties.csv")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    HISTORY_BACKEND: str = os.getenv("HISTORY_BACKEND", "shelve")
    DATABASE_URL: Optional[str] = os.getenv("DB_URL")

settings = Settings()
```

---

### `app\core\database.py`

```py
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create database engine using the URL from settings
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Simple function to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

```

---

### `app\routers\chat.py`

```py
# app/routers/chat.py
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
        logger.info(f"Processing query: '{request.query}' for session: {session_id}"
... [content truncated]
```

---

### `app\routers\dp.py`

```py
from fastapi import APIRouter
from app.services.db_tasks import list_tables

router = APIRouter(prefix="/db", tags=["Database"])

@router.get("/tables")
def get_tables():
    return {"tables": list_tables()}

```

---

### `app\schemas.py`

```py
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

```

---

### `app\services\comparison.py`

```py
# app/services/comparison.py - NEW FILE
from typing import List, Dict, Any, Optional
import statistics


class PropertyComparison:

    @staticmethod
    def compare_properties(properties: List[Dict[str, Any]], comparison_type: str = "side_by_side") -> Dict[str, Any]:
        """
        Compare multiple properties and generate insights
        """
        if len(properties) < 2:
            return {"error": "Need at least 2 properties to compare"}

        comparison_result = {
            "properties": properties,
            "comparison_type": comparison_type,
            "insights": PropertyComparison._generate_insights(properties),
            "recommendations": PropertyComparison._generate_recommendations(properties)
        }

        return comparison_result

    @staticmethod
    def _generate_insights(properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison insights"""
        insights = {}

        # Extract numeric values safely
        prices = [PropertyComparison._safe_float(p.get('metadata', {}).get('price_egp')) for p in properties]
        areas = [PropertyComparison._safe_float(p.get('metadata', {}).get('area_m2')) for p in properties]
        beds = [PropertyComparison._safe_int(p.get('metadata', {}).get('beds')) for p in properties]

        # Filter out None values
        valid_prices = [p for p in prices if p is not None]
        valid_areas = [a for a in areas if a is not None]
        valid_beds = [b for b in beds if b is not None]

        # Price analysis
        if valid_prices:
            insights['price'] = {
                'cheapest_index': prices.index(min(valid_prices)),
                'most_expensive_index': prices.index(max(valid_prices)),
                'average_price': statistics.mean(valid_prices),
                'price_range': max(valid_prices) - min(valid_prices)
            }

        # Area analysis
        if valid_areas:
            insights['area'] = {
                'smallest_index': area
... [content truncated]
```

---

### `app\services\conversation_store.py`

```py
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

    def store_search_results(self, sessio
... [content truncated]
```

---

### `app\services\db_tasks.py`

```py
from sqlalchemy import text
from app.core.database import engine

def list_tables():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
        return [row[0] for row in result]

```

---

### `app\services\embeddings.py`

```py
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

EMBED_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local')

if EMBED_PROVIDER == 'local':
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = os.getenv('HF_LOCAL_MODEL', 'all-MiniLM-L6-v2')
    model = SentenceTransformer(MODEL_NAME)
else:
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')


def get_embedding(text: str) -> List[float]:
    if EMBED_PROVIDER == 'local':
        v = model.encode([text], convert_to_numpy=True)[0]
        return v.tolist()
    else:
        resp = openai.Embedding.create(model=os.getenv('OPENAI_EMBEDDING_MODEL'), input=text)
        return resp['data'][0]['embedding']


def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    if EMBED_PROVIDER == 'local':
        arr = model.encode(texts, convert_to_numpy=True)
        return [a.tolist() for a in arr]
    else:
        resp = openai.Embedding.create(model=os.getenv('OPENAI_EMBEDDING_MODEL'), input=texts)
        return [d['embedding'] for d in resp['data']]
```

---

### `app\services\llm.py`

```py
# app/services/llm.py - ENHANCED VERSION
import os
import logging
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from .real_estate_keywords import REAL_ESTATE_KEYWORDS
from .comparison import PropertyComparison, format_comparison_response

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').lower()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama3-8b-8192')  # Default model
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

SYSTEM_PROMPT = """You are an expert Egyptian real estate assistant helping users find properties in Egypt.

AVAILABLE PROPERTY ATTRIBUTES:
- Location (city/area)
- Property type (apartment, villa, townhouse, office, etc.) 
- Number of bedrooms and bathrooms
- Area in square meters
- Price in EGP
- Price per square meter
- Payment plan details (if available)
- Project/Compound name

CAPABILITIES: 
1. Property Search and Filtering
2. Property Information and Details
3. Property Comparison - When comparing properties, provide:
   - Clear identification of best value (lowest price per sqm)
   - Spaciousness comparison
   - Location advantages/disadvantages
   - Recommendation based on typical buyer needs
4. Market Insights and Price Analysis

COMPARISON GUIDELINES:
- Be concise - max 150 words for comparisons
- Start with "Comparing X properties:" 
- Highlight the SINGLE best property for value
- Mention key differentiators only
- End with one clear recommendation

GENERAL GUIDELINES:
1. Be concise and factual based on available property data
2. If exact matches aren't found, suggest similar options
3. For price comparisons, always reference price per square meter
4. When showing multiple properties, use structured format
5. Be transparent about any limitations in the data

RESPONSE FORMAT (Regular Queries):
1. Brief summary of foun
... [content truncated]
```

---

### `app\services\query_parser.py`

```py
# app/services/query_parser.py - IMPROVED VERSION
import re
from typing import Dict, Any, Optional, Tuple
from .real_estate_keywords import REAL_ESTATE_KEYWORDS

_num_re = re.compile(r'(\d+(?:[.,]\d+)?)')
_range_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(\d+(?:[.,]\d+)?)')

# IMPROVED BEDROOM DETECTION - Multiple patterns
beds_patterns = [
    re.compile(r'(\d+)\s*(?:bedrooms?|beds?|br\b)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:bedroom|bed)', re.IGNORECASE),  # "3 bedroom villa"
    re.compile(r'(\d+)\s*(?:b\b)', re.IGNORECASE),  # "3b villa"
    re.compile(r'(\d+)\s*(?:room)', re.IGNORECASE),  # "3 room apartment"
]

# IMPROVED BATHROOM DETECTION
baths_patterns = [
    re.compile(r'(\d+)\s*(?:bathrooms?|baths?|ba\b)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:bathroom|bath)', re.IGNORECASE),
    re.compile(r'(\d+)(?:\.\d+)?\s*(?:bath)', re.IGNORECASE),  # "2.5 bath"
]

sqm_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:sqm|m2|sq m|square meters|square metres|sq\.m)', re.IGNORECASE)

# IMPROVED PRICE PATTERNS
price_patterns = [
    re.compile(r'(under|below|less than|up to|upto|<=|<)\s*([0-9.,kKmM]+)', re.IGNORECASE),
    re.compile(r'(over|above|more than|>=|>)\s*([0-9.,kKmM]+)', re.IGNORECASE),
    re.compile(r'between\s+([0-9.,kKmM]+)\s+and\s+([0-9.,kKmM]+)', re.IGNORECASE),
    re.compile(r'from\s+([0-9.,kKmM]+)\s+to\s+([0-9.,kKmM]+)', re.IGNORECASE),
    re.compile(r'([0-9.,kKmM]+)\s*(?:-|to|–)\s*([0-9.,kKmM]+)', re.IGNORECASE),
]


def _to_number(token: str) -> Optional[float]:
    if not token:
        return None
    t = token.replace(',', '').lower().strip()
    mult = 1
    if t.endswith('m'):
        mult = 1_000_000
        t = t[:-1]
    elif t.endswith('k'):
        mult = 1_000
        t = t[:-1]
    try:
        return float(t) * mult
    except:
        return None


def parse_bedrooms(text: str) -> Optional[int]:
    """Enhanced bedroom detection with multiple patterns"""
    text = text.lower()
    for pattern in beds_patterns:
        
... [content truncated]
```

---

### `app\services\real_estate_keywords.py`

```py
REAL_ESTATE_KEYWORDS = [
    # Property Types
    "apartment", "villa", "house", "condo", "condominium", "townhouse",
    "penthouse", "studio", "duplex", "triplex", "mansion", "cottage",
    "bungalow", "loft", "flat", "unit", "property", "home", "residence",
    "building", "tower", "complex", "development", "estate",

    # Transaction Types
    "rent", "buy", "sell", "lease", "purchase", "sale", "rental",
    "selling", "buying", "renting", "leasing", "for rent", "for sale",
    "available", "listing", "deal", "offer", "investment",

    # Property Features & Specifications
    "square meters", "sqm", "square feet", "sq ft", "area", "space",
    "bedrooms", "beds", "rooms", "bathrooms", "baths", "kitchen",
    "living room", "dining room", "balcony", "terrace", "garden",
    "garage", "parking", "pool", "swimming pool", "gym", "elevator",
    "furnished", "unfurnished", "air conditioning", "heating",
    "floor", "ground floor", "top floor", "view", "sea view", "city view",

    # Financial Terms
    "price", "cost", "budget", "installments", "payment", "deposit",
    "down payment", "monthly", "cash", "financing", "mortgage",
    "loan", "commission", "broker fee", "maintenance", "utilities",
    "expensive", "cheap", "affordable", "luxury", "premium",

    # Location-Related (Egypt-specific and General)
    "new cairo", "sheikh zayed", "6th october", "maadi", "zamalek",
    "heliopolis", "nasr city", "downtown", "giza", "alexandria",
    "compound", "gated community", "residential", "commercial",
    "location", "neighborhood", "district", "area", "zone",
    "metro", "transportation", "mall", "school", "hospital",
    "near", "close to", "walking distance", "minutes away",

    # Real Estate Professionals & Services
    "real estate", "realtor", "agent", "broker", "developer",
    "property manager", "landlord", "tenant", "owner", "seller",
    "buyer", "client", "viewing", "inspection", "tour", "visit",

    # Property Condition & Features
    "new", "old", 
... [content truncated]
```

---

### `app\services\retriever.py`

```py
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
           
... [content truncated]
```

---

### `app\services\vector_store.py`

```py
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
                results['ids'] = [[str(i) for i in range(le
... [content truncated]
```

---

### `app\test_fixes.py`

```py
# test_fixes.py - Run this to verify all fixes work
import os


def test_bedroom_parsing():
    """Test the improved bedroom detection"""
    print("🛏️  Testing Bedroom Detection")
    print("=" * 40)

    from app.services.query_parser import parse_filters

    test_cases = [
        "3 bedroom villa in New Cairo",
        "show me a 2 bedroom apartment",
        "4 bed house under 2M EGP",
        "5 room villa with garden",
        "3BR 2BA townhouse",
        "2 bed flat in Maadi",
        "studio apartment"  # Should not detect bedrooms
    ]

    for query in test_cases:
        try:
            result = parse_filters(query)
            beds = result.get('beds', 'NOT DETECTED')
            status = "✅" if beds != 'NOT DETECTED' else "❌"
            print(f"{status} '{query}' -> beds: {beds}")
        except Exception as e:
            print(f"❌ '{query}' -> ERROR: {e}")

    print()


def test_price_range_parsing():
    """Test improved price range detection"""
    print("💰 Testing Price Range Detection")
    print("=" * 40)

    from app.services.query_parser import parse_filters

    test_cases = [
        "villa under 2M EGP",
        "apartment between 500k and 1.5M",
        "house from 1M to 3M EGP",
        "property over 2M",
        "budget up to 800k",
        "villa 1.5M - 2.5M EGP",
        "apartment below 600k"
    ]

    for query in test_cases:
        try:
            result = parse_filters(query)
            price = result.get('price_egp', 'NOT DETECTED')
            status = "✅" if price != 'NOT DETECTED' else "❌"
            print(f"{status} '{query}' -> price: {price}")
        except Exception as e:
            print(f"❌ '{query}' -> ERROR: {e}")

    print()


def test_comparison_detection():
    """Test comparison query detection"""
    print("🔄 Testing Comparison Detection")
    print("=" * 40)

    from app.services.llm import is_comparison_query

    test_cases = [
        ("compare these properties", True),
        ("property 1 vs p
... [content truncated]
```

---

### `main.py`

```py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers.chat import router as chat_router
from dotenv import load_dotenv
import os
import logging
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="RealEstate RAG API",
    description="API for Real Estate Chatbot with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Include routers
app.include_router(chat_router, prefix="/api")

@app.get('/health', tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": app.version,
        "environment": os.getenv("ENV", "development")
    }

if __name__ == '__main__':
    uvicorn.run(
        'main:app',
        host=os.getenv('HOST', '0.0.0.0'),
        port=int(os.getenv('PORT', 8000)),
        reload=os.getenv('ENV') == 'development'
    )
```

---

### `requirements.txt`

```txt
sqlalchemy
gradio
requests
pydantic
# requirements.txt - Updated for free options
fastapi
uvicorn[standard]
python-dotenv
pandas
sentence-transformers
chromadb
openai  # optional - only if using OpenAI
groq  # for free Groq API
```

---

### `scripts\document_project.py`

```py
import os
import pathlib
from typing import List, Dict, Optional
import markdown
from dataclasses import dataclass

@dataclass
class FileInfo:
    path: str
    is_dir: bool
    size: int
    content: Optional[str] = None
    description: str = ""

def get_project_structure(root_dir: str, ignore_dirs: List[str] = None) -> Dict[str, FileInfo]:
    """
    Recursively get the project structure and file information.
    """
    if ignore_dirs is None:
        ignore_dirs = ['.git', '__pycache__', '.pytest_cache', '.venv', 'venv', '.gradio', '.idea']

    project_structure = {}
    root_path = pathlib.Path(root_dir)
    
    # First add all directories to maintain hierarchy
    for dirpath, dirnames, _ in os.walk(root_dir):
        # Remove ignored directories from dirnames to prevent os.walk from traversing them
        dirnames[:] = [d for d in dirnames if not any(ignore in d for ignore in ignore_dirs)]
        
        relative_path = str(pathlib.Path(dirpath).relative_to(root_dir))
        if relative_path == '.':
            relative_path = ''
            
        # Add current directory if it's not the root
        if relative_path and relative_path not in project_structure:
            project_structure[relative_path] = FileInfo(
                path=relative_path,
                is_dir=True,
                size=0
            )
            
        # Add all subdirectories
        for dirname in dirnames:
            dir_rel_path = os.path.join(relative_path, dirname) if relative_path else dirname
            project_structure[dir_rel_path] = FileInfo(
                path=dir_rel_path,
                is_dir=True,
                size=0
            )
    
    # Then process all files
    for item in root_path.rglob('*'):
        if any(ignore in str(item) for ignore in ignore_dirs):
            continue
            
        if item.is_file():
            relative_path = str(item.relative_to(root_dir))
            size = item.stat().st_size
            
       
... [content truncated]
```

---

### `scripts\gradio_ui.py`

```py
import requests
import gradio as gr
import uuid

# ========================
# CONFIG
# ========================
BACKEND_URL = "http://127.0.0.1:8000/api/chat"


# ========================
# HELPER
# ========================
def chat_with_backend(message, session_id, k=5):
    payload = {"session_id": session_id, "query": message, "k": k}
    try:
        response = requests.post(BACKEND_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        answer = data.get("answer", "No answer from backend.")
        hits = data.get("hits", [])
        session_id = data.get("session_id", session_id)
        return answer, session_id, hits
    except Exception as e:
        return f"⚠️ Error contacting backend: {str(e)}", session_id, []


# ========================
# CHAT LOGIC (patched)
# ========================
def user_message(user_message, history, session_id, k):
    bot_message, session_id, hits = chat_with_backend(user_message, session_id, k)

    if history is None:
        history = []

    # Detect format (tuple or dict)
    uses_tuple_style = False
    if len(history) > 0 and isinstance(history[0], (list, tuple)):
        uses_tuple_style = True

    if uses_tuple_style:
        history.append((user_message, bot_message))
    else:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_message})

    # Build property cards
    cards = []
    for hit in hits:
        meta = hit.get("metadata", {}) or {}

        title = meta.get("title") or "🏠 Property"
        project = meta.get("project")
        location = meta.get("location") or meta.get("city") or "Unknown"
        beds = meta.get("beds")
        baths = meta.get("baths")
        area = meta.get("area_m2") or meta.get("area") or None

        price_raw = (
                meta.get("price_egp")
                or meta.get("price")
                or meta.get("price_formatted")
        )
        if price_raw:

... [content truncated]
```

---

### `scripts\ingest_to_vectorstore.py`

```py
import os
import sys
import shutil
import hashlib
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from app.services.embeddings import get_embedding_batch
from app.services.vector_store import ChromaVectorStore

load_dotenv()

CSV = os.getenv("PROPERTIES_CSV", "data/properties_cleaned.csv")
BATCH = int(os.getenv("INGEST_BATCH_SIZE", 128))
RESET_DB = os.getenv("RESET_DB", "false").lower() == "true"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

print(f"Loading CSV: {CSV}")
df = pd.read_csv(CSV)

# Drop duplicates (keep first occurrence)
df = df.drop_duplicates()

# Ensure there is an 'id' column (use numeric incremental IDs)
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))

# Build embedding text (richer context)
df["embedding_text"] = (
    "Title: " + df["title"].astype(str) + ", "
    "Location: " + df["location"].astype(str) + ", "
    "Beds: " + df["beds"].astype(str) + ", "
    "Baths: " + df["baths"].astype(str) + ", "
    "Area: " + df["area_m2"].astype(str) + " sqm, "
    "Price: " + df["price_egp"].astype(str) + " EGP, "
    "Payment Plan: " + df["payment_plan"].astype(str) + ". "
    "Details: " + df["details"].astype(str)
)

# Prepare fields
texts = df["embedding_text"].astype(str).fillna("").tolist()
ids = df["id"].astype(str).tolist()

# Metadata for filtering
metadatas = df[
    ["url", "title", "location", "property_type", "price_egp", "area_m2", "beds", "baths", "project", "city", "country"]
].to_dict(orient="records")

# Reset DB if requested
if RESET_DB and os.path.exists(CHROMA_DIR):
    print("RESET_DB is true → clearing Chroma database...")
    shutil.rmtree(CHROMA_DIR)

vs = ChromaVectorStore.persist_dir(CHROMA_DIR)

# Upsert in batches
for i in range(0, len(texts), BATCH):
    batch_texts = texts[i : i + BATCH]
    batch_ids = ids[i : i + BATCH]
    batch_
... [content truncated]
```

---

