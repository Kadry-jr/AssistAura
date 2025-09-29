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
1. Brief summary of found properties
2. List key details (price, size, location)
3. Include price per sqm calculations
4. End with helpful next step

RESPONSE FORMAT (Comparisons):
1. "Comparing X properties:"
2. Best value identification
3. Key differences (2-3 points max)
4. Clear recommendation (1 sentence)"""

# Enhanced keyword detection
SMALL_TALK = {"hello", "hi", "hey", "how are you", "thanks", "thank you", "good morning", "good evening", "bye",
              "greetings"}
COMPARISON_KEYWORDS = {"compare", "comparison", "vs", "versus", "difference", "differences", "better", "best",
                       "which one", "between"}


def is_chit_chat(query: str) -> bool:
    q = query.lower().strip()
    return any(tok in q for tok in SMALL_TALK) and len(q.split()) <= 4


def is_real_estate_query(query: str) -> bool:
    q = (query or "").lower()
    return any(word in q for word in REAL_ESTATE_KEYWORDS)


def is_comparison_query(query: str) -> bool:
    """Detect if user wants to compare properties"""
    q = (query or "").lower()
    return any(word in q for word in COMPARISON_KEYWORDS)


def extract_property_references(query: str, num_properties: int) -> List[int]:
    """Extract property numbers from queries like 'compare 1 and 3' or 'property 2 vs 4'"""
    query = query.lower()
    # Find number patterns
    numbers = re.findall(r'\b(\d+)\b', query)
    valid_numbers = []
    for num_str in numbers:
        try:
            num = int(num_str)
            if 1 <= num <= num_properties:  # Valid property reference
                valid_numbers.append(num - 1)  # Convert to 0-based index
        except ValueError:
            continue
    return valid_numbers


class LLMService:
    def __init__(self, provider: str = None):
        self.provider = provider.lower() if provider else LLM_PROVIDER
        self.system_prompt = SYSTEM_PROMPT
        self.last_search_results = []  # Store last search for comparisons

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
        if price_n and area_n and area_n > 0:
            price_per_m2 = price_n / area_n

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

        # Check if this is a comparison query
        if is_comparison_query(query) and cards:
            # Build comparison-focused prompt
            comparison_text = self._build_comparison_prompt(query, cards)
            user_content = f"User query: {query}\n\n{comparison_text}\n\nPlease provide a detailed comparison analysis with recommendations."
        else:
            # Regular search prompt
            if cards:
                summary_lines = []
                for i, c in enumerate(cards[:5], 1):
                    line = f"{i}. {c['title']} - {c.get('location', '')} - {c.get('beds', '?')}b/{c.get('baths', '?')}b - {c.get('area_m2', '?')} sqm - {c.get('price_egp', '?')} EGP"
                    if c.get('price_per_m2'):
                        line += f" - {int(c['price_per_m2']):,} EGP/sqm"
                    summary_lines.append(line)
                props_text = "\n".join(summary_lines)
            else:
                props_text = "No properties found matching criteria."

            user_content = f"User query: {query}\n\nRetrieved properties:\n{props_text}\n\nPlease answer the user's question concisely, reference properties by number, include price per sqm when possible, and end with a follow-up question."

        messages.append({"role": "user", "content": user_content})
        return messages

    def _build_comparison_prompt(self, query: str, cards: List[dict]) -> str:
        """Build a comparison-focused prompt"""
        # Try to extract specific property references
        property_indices = extract_property_references(query, len(cards))

        if len(property_indices) >= 2:
            # User specified which properties to compare
            comparison_cards = [cards[i] for i in property_indices]
        elif len(cards) >= 2:
            # Compare all available properties (up to 4)
            comparison_cards = cards[:4]
        else:
            return f"Only {len(cards)} property available - cannot compare. Please search for more properties first."

        # Format properties for comparison with better structure
        comparison_lines = ["Here are the properties to compare:\n"]
        for i, card in enumerate(comparison_cards, 1):
            price = int(card.get('price_egp', 0)) if card.get('price_egp') else 'N/A'
            area = int(card.get('area_m2', 0)) if card.get('area_m2') else 'N/A'
            price_per_sqm = int(card.get('price_per_m2', 0)) if card.get('price_per_m2') else 'N/A'

            line = f"\n🏠 Property {i}: {card['title']}"
            line += f"\n   📍 Location: {card.get('location', 'N/A')}"
            line += f"\n   🏗️ Type: {card.get('type', 'N/A')}"
            line += f"\n   🛏️ Bedrooms: {card.get('beds', 'N/A')} | 🚿 Bathrooms: {card.get('baths', 'N/A')}"
            line += f"\n   📐 Area: {area} sqm"
            line += f"\n   💰 Price: {price:,} EGP" if isinstance(price, int) else f"\n   💰 Price: {price} EGP"
            line += f"\n   📊 Price/sqm: {price_per_sqm:,} EGP/sqm" if isinstance(price_per_sqm,
                                                                                 int) else f"\n   📊 Price/sqm: {price_per_sqm}"
            comparison_lines.append(line)

        comparison_lines.append("\n\nProvide a brief comparison highlighting:")
        comparison_lines.append("1. Which offers best value (lowest price per sqm)")
        comparison_lines.append("2. Which is most spacious")
        comparison_lines.append("3. Key differences in location/amenities")
        comparison_lines.append("4. Your recommendation based on typical buyer priorities")

        return "\n".join(comparison_lines)

    def generate_response(self, query: str, hits: List[Dict[str, Any]] = None, history: List[Dict[str, str]] = None) -> \
    Dict[str, Any]:
        if hits is None:
            hits = []
        if history is None:
            history = []

        # Store results for future comparisons
        self.last_search_results = hits

        # Chit-chat / short talk handling
        if is_chit_chat(query) and not is_real_estate_query(query):
            reply = "Hi! I'm AssistAura, your Egyptian real estate assistant. I can help you find properties, compare options, and analyze prices. Try asking: '3-bedroom Apartment in New Cairo under 9M EGP' or 'compare these properties'."
            return {"answer": reply, "hits": [], "cards": [], "insights": {}}

        # Domain guard
        if not is_real_estate_query(query) and not is_comparison_query(query):
            return {
                "answer": "AssistAura specializes in Egyptian real estate. I can help you find properties, compare options, or get market insights. Try asking about apartments, villas, or specific areas.",
                "hits": [], "cards": [], "insights": {}}

        # Build structured cards and insights
        cards = [self.format_property_card(h) for h in hits]

        # Handle comparison queries specially
        if is_comparison_query(query):
            if len(cards) < 2:
                return {
                    "answer": "I need at least 2 properties to compare. Please search for properties first, then ask me to compare them.",
                    "hits": hits,
                    "cards": cards,
                    "insights": {}
                }

            # Use comparison service
            comparison_result = PropertyComparison.compare_properties(hits)

            # Build messages with comparison data for LLM
            messages = self._build_prompt_with_context(query, cards, history)

            # Get LLM response (it will have the comparison data in the prompt)
            if self.provider == 'groq' and GROQ_API_KEY:
                llm_response = answer_with_context_groq(messages, hits)
                answer_text = llm_response.get('answer', '')
            else:
                # Use local formatting for comparison
                answer_text = format_comparison_response(comparison_result)

            return {
                "answer": answer_text,  # Only ONE response
                "hits": hits,
                "cards": cards,
                "insights": comparison_result.get('insights', {}),
                "comparison": comparison_result
            }

        # Regular search handling
        price_per_m2_list = [c['price_per_m2'] for c in cards if c.get('price_per_m2')]
        avg_ppm = sum(price_per_m2_list) / len(price_per_m2_list) if price_per_m2_list else None
        insights = {"avg_price_per_m2": avg_ppm, "num_properties": len(cards)}

        # Build messages for LLM provider
        messages = self._build_prompt_with_context(query, cards, history)

        # Choose provider (Groq only as requested)
        if self.provider == 'groq' and GROQ_API_KEY:
            resp = answer_with_context_groq(messages, hits)
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
    """Enhanced local response with better formatting"""
    if not hits:
        return {
            "answer": "No matching properties found. Try:\n• Broadening your search criteria\n• Changing location or price range\n• Using different property type (villa, apartment, etc.)",
            "hits": [],
            "cards": [],
            "insights": {}
        }

    # Check if comparison requested
    if is_comparison_query(query):
        if len(cards) >= 2:
            comparison_result = PropertyComparison.compare_properties(hits)
            return {
                "answer": format_comparison_response(comparison_result),
                "hits": hits,
                "cards": cards,
                "insights": comparison_result.get('insights', {}),
                "comparison": comparison_result
            }
        else:
            return {
                "answer": "I need at least 2 properties to compare. Please search for more properties first.",
                "hits": hits,
                "cards": cards,
                "insights": insights
            }

    # Regular property listing
    top = cards[:3]
    lines = ["🏠 **Found Properties:**\n"]

    for i, c in enumerate(top, 1):
        title = c.get('title', f"Property {i}")
        location = c.get('location', 'Unknown')
        beds = c.get('beds', '?')
        baths = c.get('baths', '?')
        area = int(c.get('area_m2', 0)) if c.get('area_m2') else '?'
        price = int(c.get('price_egp', 0)) if c.get('price_egp') else '?'

        line = f"**{i}. {title}**"
        line += f"\n📍 {location} | 🛏️ {beds}BR 🚿 {baths}BA | 📐 {area} sqm"
        line += f"\n💰 {price:,} EGP" if isinstance(price, int) else f"\n💰 {price} EGP"

        if c.get('price_per_m2'):
            price_per_sqm = int(c['price_per_m2'])
            line += f" | 📊 {price_per_sqm:,} EGP/sqm"

        lines.append(line)
        lines.append("")  # Empty line between properties

    # Add insights
    if insights.get('avg_price_per_m2'):
        lines.append(f"📈 **Average price per sqm:** {int(insights['avg_price_per_m2']):,} EGP/sqm")

    if len(cards) > 3:
        lines.append(f"\n*Showing top 3 of {len(cards)} matching properties*")

    lines.append(f"\n💡 **Next steps:** Ask for compare between the results or More related properties")

    summary = "\n".join(lines)

    # Convert hits to processed format
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
    """Enhanced Groq integration with better error handling"""
    try:
        from groq import Groq
        if not GROQ_API_KEY:
            logger.warning("No Groq API key found. Falling back to local response.")
            return answer_with_context_local(
                messages[-1]['content'] if messages else "",
                hits,
                [LLMService().format_property_card(h) for h in hits],
                {}
            )

        client = Groq(api_key=GROQ_API_KEY)
        logger.debug(f"Sending request to Groq with model: {GROQ_MODEL}")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=600,  # Increased for comparison responses
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.1
        )

        answer_text = response.choices[0].message.content

        # Clean up response if needed
        if answer_text:
            answer_text = answer_text.strip()

        return {
            "answer": answer_text,
            "hits": hits,
            "cards": [LLMService().format_property_card(h) for h in hits]
        }

    except Exception as e:
        logger.exception(f"Groq API error: {str(e)}. Falling back to local response.")
        return answer_with_context_local(
            messages[-1]['content'] if messages else "",
            hits,
            [LLMService().format_property_card(h) for h in hits],
            {}
        )


def answer_with_context_openai(messages: List[dict], hits: list) -> dict:
    """OpenAI integration - kept for compatibility"""
    try:
        from openai import OpenAI
        if not OPENAI_KEY:
            logger.warning("No OpenAI key. Falling back to local.")
            return answer_with_context_local(
                messages[-1]['content'] if messages else "",
                hits,
                [],
                {}
            )
        client = OpenAI(api_key=OPENAI_KEY)
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            max_tokens=600,
            temperature=0.1
        )
        return {"answer": response.choices[0].message.content, "hits": hits}
    except Exception as e:
        logger.exception("OpenAI error, falling back to local")
        return answer_with_context_local(
            messages[-1]['content'] if messages else "",
            hits,
            [],
            {}
        )


# Test functions for debugging
def test_bedroom_detection():
    """Test the bedroom detection improvements"""
    test_cases = [
        "3 bedroom villa",
        "show me a 2 bedroom apartment",
        "4 bed house",
        "5 room villa",
        "studio apartment",  # Should not match
        "3BR villa",
        "2 bed townhouse"
    ]

    from .query_parser import parse_filters
    for case in test_cases:
        result = parse_filters(case)
        print(f"'{case}' -> beds: {result.get('beds', 'NOT DETECTED')}")


def test_comparison_detection():
    """Test comparison query detection"""
    test_cases = [
        "compare these properties",
        "what's better between property 1 and 2",
        "property 1 vs property 3",
        "show me differences",
        "which one is better",
        "regular villa search"  # Should not match
    ]

    for case in test_cases:
        result = is_comparison_query(case)
        print(f"'{case}' -> comparison: {result}")


if __name__ == "__main__":
    test_bedroom_detection()
    print("\n" + "=" * 50 + "\n")
    test_comparison_detection()