# realestate-rag Project Documentation

## Table of Contents
- [Project Structure](#project-structure)
- [Key Files](#key-files)

## Project Structure

```
.idea/
    inspectionProfiles/
app/
    api/
    core/
    models/
    routers/
    services/
chroma_db/
    21a0dba6-a988-4a44-a320-6e44fb20982a/
data/
scripts/
.env (644 bytes)
.env.example (606 bytes)
        profiles_settings.xml (174 bytes)
    misc.xml (318 bytes)
    modules.xml (287 bytes)
    realestate-rag.iml (413 bytes)
    workspace.xml (2388 bytes)
Dockerfile (156 bytes)
PROJECT_DOCS.md (16360 bytes)
README.md (4621 bytes)
    __init__.py (54 bytes)
        __init__.py (0 bytes)
        __init__.py (0 bytes)
        config.py (0 bytes)
        __init__.py (0 bytes)
        chat.py (5133 bytes)
    schemas.py (236 bytes)
        __init__.py (0 bytes)
        embeddings.py (1112 bytes)
        llm.py (7796 bytes)
        retriever.py (5447 bytes)
        vector_store.py (2286 bytes)
        data_level0.bin (32608256 bytes)
        header.bin (100 bytes)
        index_metadata.pickle (561672 bytes)
        length.bin (77824 bytes)
        link_lists.bin (170780 bytes)
    chroma.sqlite3 (68792320 bytes)
    properties_cleaned.csv (8125269 bytes)
main.py (1729 bytes)
requirements.txt (199 bytes)
    document_project.py (4350 bytes)
    ingest_to_vectorstore.py (1282 bytes)
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
    inspectionProfiles/
app/
    api/
    core/
    models/
    routers/
    services/
chroma_db/
    21a0dba6-a988-4a44-a320-6e44fb20982a/
data/
scripts/
.env (640 bytes)
.env.example (606 bytes)
        profiles_settings.xml (174 bytes)
    misc.xml (318 bytes)
    modules.xml (287 bytes)
    realestate-rag.iml (413 bytes)
    workspace.xml (2388 bytes)
Dockerfile (156 bytes)
PROJECT_DOCS.md (10826 bytes)
README.md (4621 bytes)
    __init__.py (54 bytes)
        __init__.py (0 bytes)
        __init__.py (0 bytes)
        config.py (0 bytes)
        __init__.py (0 bytes)
        chat.py (672 bytes)
    schemas.py (236 bytes)
        __init__.py (0 bytes)
        embeddings.py (1112 bytes)
        llm.py (5624 bytes)
        retriever.py (787 bytes)
        vector_store.py (1079 bytes)
        data_level0.bin (15446016 bytes)
        header.bin (100 bytes)
        index_metadata.pickle (255998 bytes)
        length.bin (36864 bytes)
        link_lists.bin (78548 bytes)
    chroma.sqlite3 (35364864 bytes)
    properties_cleaned.csv (8125269 bytes)
main.py (442 bytes)
requirements.txt (199 bytes)
    document_project.py (4350 bytes)
    ingest_to_vectorstore.py (1282 bytes)
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
    inspectionProfiles/
app/
    api/
    core/
    models/
    routers/
    services/
data/
scripts/
.env (0 bytes)
.env.example (304 bytes)
        profiles_settings.xml (174 bytes)
    misc.xml (318 bytes)
    modules.xml (287 bytes)
    realestate-rag.iml (413 bytes)
    workspace.xml (2297 bytes)
Dockerfile (194 bytes)
README.md (399 bytes)
        __init__.py (0 bytes)
        __init__.py (0 bytes)
        config.py (0 bytes)
... [content truncated]
```

---

### `README.md`

```md
# RealEstate RAG

FastAPI-based Retrieval-Augmented Generation (RAG) backend for real estate properties with **FREE** AI options.

## Features

- 🆓 **100% Free Options**: Choose between local-only or Groq API
- 🏠 **Real Estate Search**: Semantic search through property listings
- ⚡ **Fast**: Local embeddings + Groq for lightning-fast responses
- 💰 **Cost Control**: Multiple pricing tiers from free to paid

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <your-repo>
cd realestate-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure (Choose Your Option)

Copy `.env.example` to `.env` and choose:

**Option A: Completely FREE**
```env
LLM_PROVIDER=local
EMBEDDING_PROVIDER=local
```

**Option B: FREE with Better AI (Groq)**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_free_groq_key
EMBEDDING_PROVIDER=local
```

**Option C: Paid (OpenAI)**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_key
EMBEDDING_PROVIDER=local
```

### 3. Get Free Groq API Key (Recommended)

1. Visit: https://console.groq.com/
2. Sign up (no credit card needed)
3. Get your free API key
4. 6,000 requests/minute free tier!

### 4. Prepare Data

Make sure your CSV has these columns:
- `details` (property description)
- `id` (unique identifier)
- Other metadata (beds, baths, price_egp, location, etc.)

### 5. Ingest Data

```bash
python scripts/ingest_to_vectorstore.py
```

### 6. Run Server

```bash
python main.py
```

### 7. Test

```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "2 bedroom apartment under 8 million EGP", "k": 5}'
```

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Search properties

### Chat Request
```json
{
    "query": "3 bedroom villa in New Cairo",
    "k": 5
}
```

### Chat Response
```json
{
    "answer": "Here ar
... [content truncated]
```

---

### `app\core\config.py`

```py

```

---

### `app\routers\chat.py`

```py
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from app.services.retriever import retrieve
from app.services.llm import answer_with_context

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)

class PropertyHit(BaseModel):
    """Represents a property search result hit"""
    id: str = Field(..., description="Unique identifier for the property")
    score: float = Field(..., description="Relevance score of the result")
    document: str = Field(..., description="Property details or description")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the property"
    )
    source: Optional[HttpUrl] = Field(None, description="Source URL of the property")
    last_updated: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="When this property was last updated"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        json_schema_extra = {
            "example": {
                "id": "prop_123",
                "score": 0.95,
                "document": "Beautiful 3-bedroom apartment in downtown...",
                "metadata": {
                    "bedrooms": 3,
                    "bathrooms": 2,
                    "price": 750000,
                    "location": "Downtown"
                }
            }
        }

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's query about properties"
    )
    k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of results to return (1-20)"
    )
    user_id: Optional[str] = Field(
        None,
        description="Optional user ID for pers
... [content truncated]
```

---

### `app\schemas.py`

```py
from pydantic import BaseModel
from typing import Any, Dict


class Hit(BaseModel):
    id: str
    score: float
    document: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    hits: list
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
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'local').lower()  # 'local', 'groq', or 'openai'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

# Log the loaded configuration
logger.info(f"LLM Provider: {LLM_PROVIDER}")
if LLM_PROVIDER == 'groq':
    logger.info(f"Using Groq model: {GROQ_MODEL}")
elif LLM_PROVIDER == 'openai':
    logger.info(f"Using OpenAI model: {OPENAI_CHAT_MODEL}")

SYSTEM_PROMPT = """You are an assistant that helps users find real estate properties. Use the retrieved property details and the user's query to answer concisely. Cite matching property ids."""


def answer_with_context_local(query: str, hits: list) -> str:
    """Local/rule-based response - completely free"""
    if not hits:
        return "I couldn't find any properties matching your criteria. Please try adjusting your search parameters."

    # Extract key info from hits
    matching_properties = []
    for h in hits:
        meta = h.get('metadata', {})
        if not meta:
            continue
            
        # Skip if score is too low (less than 0.3)
        if h.get('score', 0) < 0.3:
            continue
            
        prop_info = {
            'id': h.get('id', 'N/A'),
            'beds': meta.get('beds'),
            'baths': meta.get('baths'),
            'area': meta.get('area'),
            'price': meta.get('price_egp'),
            'location': meta.get('location') or f"{meta.get('city', '')}, {meta.get('country', '')}",
            'type': meta.get('property_type', 'property')
        }
        matching_properties.append(prop_info)

    # If no properties meet the score threshold
    if not matching_properties:
        # Fi
... [content truncated]
```

---

### `app\services\retriever.py`

```py
from app.services.embeddings import get_embedding
from app.services.vector_store import ChromaVectorStore
import os
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger(__name__)

vs = ChromaVectorStore.persist_dir(os.getenv('CHROMA_PERSIST_DIR', './chroma_db'))

# Field mappings between user-friendly names and database fields
FIELD_MAPPINGS = {
    # Price related
    'price': 'price_egp',
    'min_price': ('price_egp', '$gte'),
    'max_price': ('price_egp', '$lte'),
    
    # Size related
    'area': 'area_m2',
    'min_area': ('area_m2', '$gte'),
    'max_area': ('area_m2', '$lte'),
    
    # Property details
    'bedrooms': 'beds',
    'min_bedrooms': ('beds', '$gte'),
    'max_bedrooms': ('beds', '$lte'),
    'bathrooms': 'baths',
    'min_bathrooms': ('baths', '$gte'),
    'max_bathrooms': ('baths', '$lte'),
    
    # Location and type
    'location': 'location',
    'city': 'city',
    'type': 'property_type',
    'property_type': 'property_type',
}

def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and validate filter parameters.
    Converts user-friendly field names to database field names and handles special cases.
    """
    if not filters:
        return {}
        
    normalized = {}
    
    for key, value in filters.items():
        if value is None:
            continue
            
        # Handle special filter types
        if key in FIELD_MAPPINGS:
            field_info = FIELD_MAPPINGS[key]
            
            # Handle range filters (min/max)
            if isinstance(field_info, tuple):
                field, operator = field_info
                if field not in normalized:
                    normalized[field] = {}
                normalized[field][operator] = float(value) if isinstance(value, (int, float, str)) and str(value).replace('.', '').isdigit() else value
            # Handle direct mappings
            else:
     
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

### `main.py`

```py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.routers.chat import router as chat_router
from dotenv import load_dotenv
import os
import logging
from typing import Any
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
        ignore_dirs = ['.git', '__pycache__', '.pytest_cache', '.venv', 'venv']

    project_structure = {}
    root_path = pathlib.Path(root_dir)

    for item in root_path.rglob('*'):
        if any(ignore in str(item) for ignore in ignore_dirs):
            continue

        relative_path = str(item.relative_to(root_dir))
        is_dir = item.is_dir()
        size = sum(f.stat().st_size for f in item.glob('**/*') if f.is_file()) if is_dir else item.stat().st_size

        file_info = FileInfo(
            path=relative_path,
            is_dir=is_dir,
            size=size
        )

        # Read content of relevant files
        if not is_dir and should_include_file(relative_path):
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    file_info.content = f.read()
            except (UnicodeDecodeError, PermissionError):
                file_info.content = "[Binary or unreadable file]"

        project_structure[relative_path] = file_info

    return project_structure

def should_include_file(file_path: str) -> bool:
    """Determine if a file should have its content included in the documentation."""
    include_extensions = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.sh'}
    exclude_files = {'__init__.py'}

    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_path)

    if file_name in exclude_files:
        return False
    return ext.lower() in include_extensions

def generate_markdown_docs(project_structure: Dict[str, FileInfo], output
... [content truncated]
```

---

### `scripts\ingest_to_vectorstore.py`

```py
import os
import sys
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

print(f"Loading CSV: {CSV}")
df = pd.read_csv(CSV)

# Ensure there is an id column
if 'id' not in df.columns:
    df.insert(0, 'id', range(1, len(df) + 1))

texts = df['details'].astype(str).fillna("").tolist()
ids = df['id'].astype(str).tolist()

vs = ChromaVectorStore.persist_dir(os.getenv('CHROMA_PERSIST_DIR', './chroma_db'))

# Upsert in batches
for i in range(0, len(texts), BATCH):
    batch_texts = texts[i:i+BATCH]
    batch_ids = ids[i:i+BATCH]
    embeddings = get_embedding_batch(batch_texts)
    metadatas = [df.iloc[j].to_dict() for j in range(i, min(i+BATCH, len(texts)))]
    documents = batch_texts
    vs.upsert(batch_ids, embeddings, metadatas, documents)
    print(f"Upserted batch {i//BATCH + 1} ({len(batch_texts)} items)")

print("Ingestion complete.")
```

---

