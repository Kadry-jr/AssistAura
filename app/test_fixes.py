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
        ("property 1 vs property 2", True),
        ("what's better between them", True),
        ("show differences", True),
        ("which one is better", True),
        ("3 bedroom villa search", False),
        ("find apartments", False)
    ]

    for query, expected in test_cases:
        result = is_comparison_query(query)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{query}' -> comparison: {result} (expected: {expected})")

    print()


def test_groq_configuration():
    """Test Groq setup"""
    print("🤖 Testing Groq Configuration")
    print("=" * 40)

    try:
        from app.services.llm import LLMService, GROQ_API_KEY, GROQ_MODEL

        print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'not set')}")
        print(f"Groq Model: {GROQ_MODEL}")
        print(f"API Key: {'✅ Set' if GROQ_API_KEY else '❌ Missing'}")

        llm = LLMService()
        print(f"Service Provider: {llm.provider}")

        if GROQ_API_KEY:
            print("✅ Groq configuration looks good!")
        else:
            print("❌ GROQ_API_KEY not found in environment")
            print("   Add to .env file: GROQ_API_KEY=your_key_here")

    except Exception as e:
        print(f"❌ Configuration error: {e}")

    print()


def test_property_comparison():
    """Test property comparison functionality"""
    print("🏠 Testing Property Comparison")
    print("=" * 40)

    try:
        from app.services.comparison import PropertyComparison, format_comparison_response

        # Mock properties for testing
        test_properties = [
            {
                'id': 'prop1',
                'metadata': {
                    'title': 'Villa A',
                    'price_egp': 2000000,
                    'area_m2': 250,
                    'beds': 3,
                    'baths': 2,
                    'location': 'New Cairo'
                }
            },
            {
                'id': 'prop2',
                'metadata': {
                    'title': 'Villa B',
                    'price_egp': 2500000,
                    'area_m2': 300,
                    'beds': 4,
                    'baths': 3,
                    'location': 'Sheikh Zayed'
                }
            }
        ]

        comparison = PropertyComparison.compare_properties(test_properties)

        if 'insights' in comparison:
            print("✅ Comparison analysis working")
            print(f"   Best value index: {comparison['insights'].get('value', {}).get('best_value_index', 'N/A')}")
        else:
            print("❌ Comparison analysis failed")

        # Test formatting
        formatted = format_comparison_response(comparison)
        if len(formatted) > 100:  # Should be a substantial response
            print("✅ Comparison formatting working")
        else:
            print("❌ Comparison formatting issues")

    except Exception as e:
        print(f"❌ Comparison test error: {e}")

    print()


def run_all_tests():
    """Run all tests"""
    print("🧪 Running All Fix Tests")
    print("=" * 50)
    print()

    test_bedroom_parsing()
    test_price_range_parsing()
    test_comparison_detection()
    test_groq_configuration()
    test_property_comparison()

    print("🎉 Testing complete!")
    print("\nNext steps:")
    print("1. Make sure GROQ_API_KEY is set in your .env file")
    print("2. Test the /debug/test-parsing endpoint")
    print("3. Try these queries in your chat API:")
    print("   - '3 bedroom villa in New Cairo'")
    print("   - 'apartment under 1.5M EGP'")
    print("   - 'compare these properties' (after a search)")


if __name__ == "__main__":
    run_all_tests()