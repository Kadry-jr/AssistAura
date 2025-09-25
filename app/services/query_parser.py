# app/services/query_parser.py
import re
from typing import Dict, Any, Optional, Tuple
from .real_estate_keywords import REAL_ESTATE_KEYWORDS

_num_re = re.compile(r'(\d+(?:[.,]\d+)?)')
_range_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:-|to|–)\s*(\d+(?:[.,]\d+)?)')
beds_re = re.compile(r'(\d+)\s*(?:bedrooms?|beds?|br\b)')
baths_re = re.compile(r'(\d+)\s*(?:bathrooms?|baths?|ba\b)')
sqm_re = re.compile(r'(\d+(?:[.,]\d+)?)\s*(?:sqm|m2|sq m|square meters|square metres|sq\.m)')
price_peek_re = re.compile(r'(under|below|less than|up to|upto|<=|<|between|from|to|and)\s*([0-9.,kKmM]+)')

def _to_number(token: str) -> Optional[float]:
    if not token:
        return None
    t = token.replace(',', '').lower().strip()
    mult = 1
    if t.endswith('m'):
        mult = 1_000_000
        t = t[:-1]
    if t.endswith('k'):
        mult = 1_000
        t = t[:-1]
    try:
        return float(t) * mult
    except:
        return None

def parse_price_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    text = text.lower()
    # between X and Y
    m = re.search(r'between\s+([0-9.,kKmM]+)\s+and\s+([0-9.,kKmM]+)', text)
    if m:
        return _to_number(m.group(1)), _to_number(m.group(2))
    # "under X", "below X", "up to X"
    m = re.search(r'(under|below|less than|up to|upto)\s+([0-9.,kKmM]+)', text)
    if m:
        return None, _to_number(m.group(2))
    # "from X to Y" or "X - Y"
    m = re.search(r'from\s+([0-9.,kKmM]+)\s+to\s+([0-9.,kKmM]+)', text)
    if m:
        return _to_number(m.group(1)), _to_number(m.group(2))
    m = _range_re.search(text)
    if m:
        return _to_number(m.group(1)), _to_number(m.group(2))
    # single number (treat as max price if 'under' near it wasn't found)
    m = re.search(r'([0-9.,kKmM]+)\s*(egp|e|le|egyptian pounds)?', text)
    if m:
        # Heuristic: if 'budget' or 'price' or 'for' nearby then it's a price
        if any(w in text for w in ['price', 'budget', 'egp', 'under', 'below', 'max', 'up to']):
            return None, _to_number(m.group(1))
    return None, None

def find_location(text: str) -> Optional[str]:
    t = text.lower()
    # look for multiword locations from REAL_ESTATE_KEYWORDS first
    candidates = [kw for kw in REAL_ESTATE_KEYWORDS if ' ' in kw and kw in t]
    if candidates:
        # take the first best match
        return candidates[0]
    # then single tokens
    words = set(t.split())
    for kw in REAL_ESTATE_KEYWORDS:
        if ' ' not in kw and kw in words:
            return kw
    return None

PROPERTY_TYPES = {
    'apartment','villa','house','condo','townhouse','studio','penthouse','duplex','mansion','loft','flat'
}

def find_property_type(text: str) -> Optional[str]:
    t = text.lower()
    for p in PROPERTY_TYPES:
        if p in t:
            return p
    return None

def parse_filters(text: str) -> Dict[str, Any]:
    text = (text or "").lower()
    filters = {}
    # bedrooms
    m = beds_re.search(text)
    if m:
        try:
            filters['beds'] = int(m.group(1))
        except:
            pass
    # bathrooms
    m = baths_re.search(text)
    if m:
        try:
            filters['baths'] = int(m.group(1))
        except:
            pass
    # area
    m = sqm_re.search(text)
    if m:
        val = _to_number(m.group(1))
        if val:
            filters['area_m2'] = val
    # price range
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
