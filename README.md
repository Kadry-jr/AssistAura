# Assist-Aura

> Part of the BautAura Real Estate Platform

A FastAPI-based Retrieval-Augmented Generation (RAG) system for real estate property search and recommendation. This application allows users to query property listings using natural language and receive relevant property recommendations based on semantic similarity, serving as the intelligent assistant component of the BautAura platform.

## Features

- **Semantic Property Search**: Find properties using natural language queries
- **Conversational Interface**: Maintains conversation context for more relevant results
- **Advanced Filtering**: Automatically extracts and applies filters from natural language
- **Vector Similarity Search**: Leverages ChromaDB for efficient semantic search
- **Multi-turn Conversations**: Maintains conversation history for context-aware responses
- **FastAPI Backend**: High-performance API with async support

## Tech Stack

- **Backend**: FastAPI
- **Vector Database**: ChromaDB
- **Embeddings**: Local or cloud-based embeddings
- **LLM Integration**: Support for multiple LLM providers
- **Session Management**: In-memory conversation store

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- ChromaDB

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd assistaura
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit the `.env` file with your configuration.

### Configuration

Configure the following environment variables in your `.env` file:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Vector Store Configuration
CHROMA_PERSIST_DIR=./chroma_db

# LLM Configuration (choose one)
# Option 1: Local LLM
LLM_PROVIDER=local

# Option 2: Groq
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_groq_api_key

# Option 3: OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your_openai_api_key
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Chat Endpoint

`POST /api/chat`

Send a chat message and get property recommendations.

**Request Body:**
```json
{
  "session_id": "optional-session-id",
  "query": "Find me a 2-bedroom apartment in New Zaid",
  "k": 5
}
```

**Response:**
```json
{
  "session_id": "generated-session-id",
  "response": "Here are some 2-bedroom apartments in New Zaid...",
  "hits": [
    {
      "id": "property-123",
      "score": 0.92,
      "metadata": {
        "title": "Luxury 2-Bed Apartment",
        "price": "$250,000",
        "bedrooms": 2,
        "location": "New Zaid"
      }
    }
  ]
}
```

### Health Check

`GET /health`

Check if the API is running.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2023-10-01T12:00:00Z"
}
```

## Development

### Project Structure

```
assistaura/
├── app/
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   │   ├── llm.py         # LLM service
│   │   ├── retriever.py   # Document retrieval
│   │   └── vector_store.py# Vector store interface
│   ├── schemas.py         # Pydantic models
│   └── routers/           # API routers
├── data/                  # Property data
├── scripts/               # Utility scripts
├── .env.example           # Example environment variables
├── main.py                # Application entry point
└── requirements.txt       # Python dependencies
```

### Adding New Properties

To add new properties to the vector store:

1. Place your property data in CSV format in the `data/` directory
2. Run the ingestion script:
   ```bash
   python scripts/ingest_to_vectorstore.py
   ```

## License

This project is part of the BautAura platform and is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.