# app/services/llm.py
import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import pandas as pd
from .real_estate_keywords import REAL_ESTATE_KEYWORDS

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq').lower()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = os.getenv('GROQ_MODEL')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

SYSTEM_PROMPT = """
You are an expert Egyptian real estate assistant helping users find properties in Egypt.
 AVAILABLE PROPERTY ATTRIBUTES:
  - Location (city/area)
   - Property type (apartment, villa, townhouse, office, etc.) 
   - Number of bedrooms - Number of bathrooms 
   - Area in square meters 
   - Price in EGP - Price per square meter
- Payment plan details (if available) 
- Project/Compound name CAPABILITIES: 
1. Property Search:
 - Filter by location (e.g., New Cairo, 6th of October, Sheikh Zayed) 
 - Filter by property type 
 - Filter by number of bedrooms/bathrooms
  - Filter by price range 
  - Filter by area/size 
  - Sort by price, area, or price per square meter
   2. Property Information: 
   - Provide detailed property specifications 
   - Calculate price per square meter
   - Explain payment plans 
   - Compare similar properties 
   3. Market Insights: 
   - Price ranges in different areas 
   - Average prices by property type 
   - Price per square meter comparisons 
   GUIDELINES: 
   1. Be concise and factual based on available property data 
   2. If exact matches aren't found, suggest similar options 
   3. For price comparisons, always show price per square meter 
   4. When showing multiple properties, include key details in a structured format 
   5. Be transparent about any limitations in the data 
   6. For payment plans, clearly explain the terms and total cost 
   7. Always mention the source of the property listing
    LIMITATIONS: 
    - Cannot provide legal or financial advice 
    - Cannot guarantee availability of listed properties
     - Prices and availability may change without notice 
     - Always verify property details before making decisions 
     RESPONSE FORMAT: 
     1. Start with a brief summary of found properties 
     2. List key details for each property (price, size, location, etc.) 
     3. Include relevant calculations (price per sqm, monthly payments if applicable) 
     4. End with next steps or suggestions"""

# small chit-chat detector
SMALL_TALK = {"hello","hi","hey","how are you","thanks","thank you","good morning","good evening","bye"}

def is_chit_chat(query: str) -> bool:
    q = query.lower().strip()
    return any(tok in q for tok in SMALL_TALK)

def is_real_estate_query(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in REAL_ESTATE_KEYWORDS)

class LLMService:
    def __init__(self, provider: str = None):
        self.provider = provider.lower() if provider else LLM_PROVIDER
        self.system_prompt = SYSTEM_PROMPT

    def format_property_card(self, hit: dict) -> Dict[str, Any]:
        meta = hit.get('metadata', {})
        # normalize fields
        price = meta.get('price_egp') or meta.get('price') or None
        try:
            price_n = float(price) if price is not None else None
        except:
            price_n = None
        area = meta.get('area_m2') or meta.get('area') or None
        try:
            area_n = float(area) if area is not None else None
        except:
            area_n = None
        price_per_m2 = None
        if price_n and area_n:
            price_per_m2 = price_n / area_n if area_n > 0 else None

        return {
            "id": hit.get('id'),
            "title": meta.get('title') or meta.get('property_type') or "Property",
            "type": meta.get('property_type'),
            "beds": meta.get('beds'),
            "baths": meta.get('baths'),
            "area_m2": area_n,
            "price_egp": price_n,
            "price_per_m2": price_per_m2,
            "location": meta.get('location') or meta.get('city'),
            "url": meta.get('url')
        }

    def _build_prompt_with_context(self, query: str, cards: List[dict], history: List[dict]) -> List[dict]:
        messages = [{"role": "system", "content": self.system_prompt}]
        # attach recent history turns (last 6)
        if history:
            for h in history[-6:]:
                messages.append({"role": h.get("role"), "content": h.get("content")})
        # build summary of cards
        if cards:
            summary_lines = []
            for i, c in enumerate(cards[:5], 1):
                summary_lines.append(f"{i}. {c['title']} - {c.get('location','')} - {c.get('beds','?')}b/{c.get('baths','?')}b - {c.get('area_m2','?')} sqm - {c.get('price_egp','?')} EGP")
            props_text = "\n".join(summary_lines)
        else:
            props_text = "No properties found matching criteria."

        user_content = f"User query: {query}\n\nRetrieved (top) properties:\n{props_text}\n\nPlease answer the user's question concisely, reference the best matching properties (by number), include price per sqm when possible, and end with a follow-up question."
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate_response(self, query: str, hits: List[Dict[str, Any]] = None, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        if hits is None:
            hits = []
        if history is None:
            history = []

        # Chit-chat / short talk handling
        if is_chit_chat(query) and not is_real_estate_query(query):
            reply = "Hi! I'm AssistAura. Your real estate assistant. You can ask about properties, prices, or neighborhoods in Egypt — for example: '3 bedroom apartment in New Cairo under 5M EGP'."
            return {"answer": reply, "hits": [], "cards": [], "insights": {}}

        # Domain guard
        if not is_real_estate_query(query):
            return {"answer": "AssistAura can only help with Egyptian real estate questions. Try asking about apartments, villas, or areas like New Cairo.", "hits": [], "cards": [], "insights": {}}

        # Build structured cards and simple insights
        cards = [self.format_property_card(h) for h in hits]
        # compute simple stats
        price_per_m2_list = [c['price_per_m2'] for c in cards if c.get('price_per_m2')]
        avg_ppm = sum(price_per_m2_list) / len(price_per_m2_list) if price_per_m2_list else None
        insights = {"avg_price_per_m2": avg_ppm, "num_properties": len(cards)}

        # Build messages for LLM provider
        messages = self._build_prompt_with_context(query, cards, history)

        # Choose provider
        if self.provider == 'groq':
            resp = answer_with_context_groq(messages, hits)
        elif self.provider == 'openai':
            resp = answer_with_context_openai(messages, hits)
        else:
            resp = answer_with_context_local(query, hits, cards, insights)

        # Ensure dict output
        if not isinstance(resp, dict):
            resp = {"answer": str(resp), "hits": hits, "cards": cards, "insights": insights}

        # attach stats
        resp.setdefault('cards', cards)
        resp.setdefault('insights', insights)
        return resp


def answer_with_context_local(query: str, hits: list, cards: list, insights: dict) -> dict:
    # Rule-based nice formatting when LLM is not used
    if not hits:
        return {"answer": "No matching properties found. Try broadening your search or changing location/price.", "hits": [], "cards": [], "insights": {}}

    # pick top 3
    top = cards[:3]
    lines = []
    for i, c in enumerate(top, 1):
        line = f"{i}. {c.get('title','Property')} — {c.get('location','Unknown')} — {int(c.get('area_m2')) if c.get('area_m2') else '?'} sqm — {int(c.get('price_egp')) if c.get('price_egp') else '?'} EGP"
        if c.get('price_per_m2'):
            line += f" — {int(c['price_per_m2']):,} EGP/sqm"
        lines.append(line)

    summary = f"I found {len(cards)} matching properties. Top results:\n\n" + "\n".join(lines)
    if insights.get('avg_price_per_m2'):
        summary += f"\n\nAverage price per sqm among results: {int(insights['avg_price_per_m2']):,} EGP/sqm"
    summary += "\n\nWould you like more details on any of these (e.g. 'more info on 1')?"

    # convert hits to processed hits shape similar to earlier version
    processed_hits = []
    for h in hits:
        meta = h.get('metadata', {}) or {}
        processed_hits.append({
            'id': h.get('id'),
            'score': h.get('score'),
            'document': h.get('document'),
            'metadata': meta
        })

    return {"answer": summary, "hits": processed_hits, "cards": top, "insights": insights}


def answer_with_context_groq(messages: List[dict], hits: list) -> dict:
    try:
        from groq import Groq
        if not GROQ_API_KEY:
            logger.warning("No Groq API key. Falling back to local.")
            return answer_with_context_local(messages[-1]['content'] if messages else "", hits, [], {})
        client = Groq(api_key=GROQ_API_KEY)
        logger.debug("Sending to Groq...")
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.1
        )
        answer_text = response.choices[0].message.content
        return {"answer": answer_text, "hits": hits}
    except Exception as e:
        logger.exception("Groq error, falling back to local")
        return answer_with_context_local(messages[-1]['content'] if messages else "", hits, [], {})


def answer_with_context_openai(messages: List[dict], hits: list) -> dict:
    try:
        from openai import OpenAI
        if not OPENAI_KEY:
            logger.warning("No OpenAI key. Falling back to local.")
            return answer_with_context_local(messages[-1]['content'] if messages else "", hits, [], {})
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.1
        )
        return {"answer": response.choices[0].message.content, "hits": hits}
    except Exception as e:
        logger.exception("OpenAI error, falling back to local")
        return answer_with_context_local(messages[-1]['content'] if messages else "", hits, [], {})
