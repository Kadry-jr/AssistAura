# app/services/query_parser.py
import re
from typing import Dict, Any, Optional, Tuple
from .real_estate_keywords import REAL_ESTATE_KEYWORDS

_num_re = re.compile(r'(\d+(?:[.,]\d+)?)')
_range_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(\d+(?:[.,]\d+)?)')

# BEDROOM DETECTION - Multiple patterns
beds_patterns = [
    re.compile(r'(\d+)\s*(?:bedrooms?|beds?|br\b)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:bedroom|bed)', re.IGNORECASE),  # "3 bedroom villa"
    re.compile(r'(\d+)\s*(?:b\b)', re.IGNORECASE),  # "3b villa"
    re.compile(r'(\d+)\s*(?:room)', re.IGNORECASE),  # "3 room apartment"
]

# BATHROOM DETECTION
baths_patterns = [
    re.compile(r'(\d+)\s*(?:bathrooms?|baths?|ba\b)', re.IGNORECASE),
    re.compile(r'(\d+)\s*(?:bathroom|bath)', re.IGNORECASE),
    re.compile(r'(\d+)(?:\.\d+)?\s*(?:bath)', re.IGNORECASE),
]

sqm_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:sqm|m2|sq m|square meters|square metres|sq\.m)', re.IGNORECASE)

# PRICE PATTERNS
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
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    return None


def parse_bathrooms(text: str) -> Optional[float]:
    """Enhanced bathroom detection with decimal support"""
    text = text.lower()
    for pattern in baths_patterns:
        match = pattern.search(text)
        if match:
            try:
                return float(match.group(1))
            except:
                continue
    return None


def parse_price_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    """IMPROVED price range parsing with better patterns"""
    text = text.lower()

    # Pattern 1: between X and Y
    for pattern in price_patterns:
        match = pattern.search(text)
        if not match:
            continue

        if 'between' in match.group(0):
            return _to_number(match.group(1)), _to_number(match.group(2))
        elif any(word in match.group(1) for word in ['under', 'below', 'less than', 'up to', 'upto']):
            return None, _to_number(match.group(2))
        elif any(word in match.group(1) for word in ['over', 'above', 'more than']):
            return _to_number(match.group(2)), None
        elif 'from' in match.group(0) or 'to' in match.group(0) or '-' in match.group(0):
            return _to_number(match.group(1)), _to_number(match.group(2))

    # Fallback: single number with context
    single_price = re.search(r'([0-9.,kKmM]+)\s*(?:egp|e|le|egyptian pounds)?', text)
    if single_price:
        # Check context for upper/lower bound
        price_val = _to_number(single_price.group(1))
        price_context = text[max(0, single_price.start() - 20):single_price.end() + 10]

        if any(word in price_context for word in ['under', 'below', 'max', 'up to', 'budget']):
            return None, price_val
        elif any(word in price_context for word in ['over', 'above', 'min', 'starting']):
            return price_val, None

    return None, None


def find_location(text: str) -> Optional[str]:
    t = text.lower()
    # look for multiword locations from REAL_ESTATE_KEYWORDS first
    candidates = [kw for kw in REAL_ESTATE_KEYWORDS if ' ' in kw and kw in t]
    if candidates:
        # take the longest match first (most specific)
        return sorted(candidates, key=len, reverse=True)[0]
    # then single tokens
    words = set(t.split())
    for kw in REAL_ESTATE_KEYWORDS:
        if ' ' not in kw and kw in words:
            return kw
    return None


PROPERTY_TYPES = {
    'apartment', 'villa', 'house', 'condo', 'townhouse', 'studio', 'penthouse', 'duplex', 'mansion', 'loft', 'flat' , 'retail' , 'office' , 'warehouse' , 'commercial' , 'land' , 'other'
}


def find_property_type(text: str) -> Optional[str]:
    t = text.lower()
    for p in PROPERTY_TYPES:
        if p in t:
            return p
    return None


def parse_filters(text: str) -> Dict[str, Any]:
    """ filter parsing with better detection"""
    text = (text or "").lower()
    filters = {}

    #  bedrooms detection
    beds = parse_bedrooms(text)
    if beds:
        filters['beds'] = beds

    #  bathrooms detection
    baths = parse_bathrooms(text)
    if baths:
        filters['baths'] = baths

    # area
    m = sqm_re.search(text)
    if m:
        val = _to_number(m.group(1))
        if val:
            filters['area_m2'] = val

    #  price range
    pmin, pmax = parse_price_range(text)
    if pmin is not None or pmax is not None:
        pr = {}
        if pmin is not None:
            pr['$gte'] = pmin
        if pmax is not None:
            pr['$lte'] = pmax
        filters['price_egp'] = pr

    # location & property type
    location = find_location(text)
    if location:
        filters['location'] = location
    ptype = find_property_type(text)
    if ptype:
        filters['property_type'] = ptype

    return filters


# TEST FUNCTION
def test_parsing():
    """Test function to verify parsing works correctly"""
    test_queries = [
        "3 bedroom villa in New Cairo",
        "show me a 2 bedroom apartment",
        "4 bed house under 2M EGP",
        "villa between 1.5M and 3M",
        "apartment under 500k",
        "over 1M EGP villa",
        "3BR 2BA house"
    ]

    for query in test_queries:
        result = parse_filters(query)
        print(f"Query: '{query}' -> {result}")


if __name__ == "__main__":
    test_parsing()