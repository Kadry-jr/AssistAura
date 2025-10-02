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

## 🔍 Recommendation System

The application includes a powerful recommendation system that helps users discover relevant properties based on different criteria. The system provides three types of recommendations:

1. **Location-based Similar Properties**: Finds properties in the same location with similar pricing
2. **Price-based Similar Properties**: Finds properties with similar pricing regardless of location
3. **User-based Recommendations**: Provides personalized recommendations based on user preferences and behavior

### 📍 Location-based Similar Properties

Finds properties in the same location with similar pricing (±30% of the target property's price).

**Endpoint**: `GET /recommendations/similar/location/{property_id}`

**Parameters**:
- `property_id` (required): ID of the property to find similar ones for
- `limit` (optional, default=5): Maximum number of similar properties to return

**Example Request**:
```http
GET /recommendations/similar/location/123?limit=5
```

### 💰 Price-based Similar Properties

Finds properties with similar pricing (±30% of the target property's price) regardless of location.

**Endpoint**: `GET /recommendations/similar/anywhere/{property_id}`

**Parameters**:
- `property_id` (required): ID of the property to find similar ones for
- `limit` (optional, default=5): Maximum number of similar properties to return

**Example Request**:
```http
GET /recommendations/similar/anywhere/123?limit=5
```

### 👤 User-based Recommendations

Provides personalized property recommendations based on the user's preferences and behavior.

**Endpoint**: `GET /recommendations/user/{user_id}`

**Parameters**:
- `user_id` (required): ID of the user to get recommendations for
- `limit` (optional, default=5): Maximum number of recommendations to return

**Example Request**:
```http
GET /recommendations/user/456?limit=5
```

### Response Format

All recommendation endpoints return an array of property objects with the following structure:

```json
[
  {
    "property_details_id": 123,
    "address": "123 Main St, City",
    "area": 120.5,
    "description": "Beautiful property with great views...",
    "price": 250000,
    "title": "Luxury Apartment in Prime Location",
    "type": "Apartment",
    "purpose": "For Sale",
    "property_status": "Available",
    "image_url": "https://example.com/image1.jpg"
  },
  ...
]
```

## 📚 Project Structure

```
AssistAura/
├── app/                      # Main application package
│   ├── api/                  # API endpoints
│   ├── core/                 # Core configurations
│   ├── models/               # Data models
│   ├── routers/              # API routers
│   ├── services/             # Business logic
│   │   ├── comparison.py     # Property comparison
│   │   ├── llm.py           # LLM integration
│   │   ├── retriever.py     # Document retrieval
│   │   └── ...              # Other services
│   └── schemas.py           # Pydantic models
├── data/                    # Property data
├── scripts/                 # Utility scripts
├── .env.example             # Environment template
├── main.py                 # Application entry point
└── requirements.txt         # Dependencies
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
