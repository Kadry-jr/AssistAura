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
                'smallest_index': areas.index(min(valid_areas)),
                'largest_index': areas.index(max(valid_areas)),
                'average_area': statistics.mean(valid_areas)
            }

        # Price per sqm analysis
        price_per_sqm = []
        for i, prop in enumerate(properties):
            price = PropertyComparison._safe_float(prop.get('metadata', {}).get('price_egp'))
            area = PropertyComparison._safe_float(prop.get('metadata', {}).get('area_m2'))
            if price and area and area > 0:
                price_per_sqm.append((i, price / area))

        if price_per_sqm:
            best_value = min(price_per_sqm, key=lambda x: x[1])
            worst_value = max(price_per_sqm, key=lambda x: x[1])
            insights['value'] = {
                'best_value_index': best_value[0],
                'best_value_per_sqm': best_value[1],
                'worst_value_index': worst_value[0],
                'worst_value_per_sqm': worst_value[1]
            }

        return insights

    @staticmethod
    def _generate_recommendations(properties: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendation text"""
        recommendations = []

        if len(properties) < 2:
            return recommendations

        insights = PropertyComparison._generate_insights(properties)

        # Price recommendations
        if 'price' in insights:
            cheapest_idx = insights['price']['cheapest_index']
            recommendations.append(f"Property {cheapest_idx + 1} is the most budget-friendly option")

        # Value recommendations
        if 'value' in insights:
            best_value_idx = insights['value']['best_value_index']
            recommendations.append(f"Property {best_value_idx + 1} offers the best value per square meter")

        # Area recommendations
        if 'area' in insights:
            largest_idx = insights['area']['largest_index']
            recommendations.append(f"Property {largest_idx + 1} provides the most living space")

        return recommendations

    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float"""
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert to int"""
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None


# Integration functions for LLM service
def format_comparison_response(comparison_result: Dict[str, Any]) -> str:
    """Format comparison result for LLM response"""
    if 'error' in comparison_result:
        return comparison_result['error']

    properties = comparison_result['properties']
    insights = comparison_result.get('insights', {})
    recommendations = comparison_result.get('recommendations', [])

    response_lines = ["🏠 Property Comparison\n"]

    # Property summary
    for i, prop in enumerate(properties):
        meta = prop.get('metadata', {})
        response_lines.append(f"{i + 1}. {meta.get('title', 'Property')}")
        response_lines.append(f"   📍 {meta.get('location', 'N/A')}")
        response_lines.append(f"   🛏️ {meta.get('beds', '?')}BR, 🚿 {meta.get('baths', '?')}BA")
        response_lines.append(f"   📐 {meta.get('area_m2', '?')} sqm")
        response_lines.append(f"   💰 {int(meta.get('price_egp', 0)):,} EGP")
        if meta.get('price_egp') and meta.get('area_m2'):
            price_per_sqm = float(meta['price_egp']) / float(meta['area_m2'])
            response_lines.append(f"   📊 {int(price_per_sqm):,} EGP/sqm")
        response_lines.append("")

    # Insights
    if insights:
        response_lines.append("📈 Key Insights:")

        if 'price' in insights:
            cheapest_idx = insights['price']['cheapest_index']
            expensive_idx = insights['price']['most_expensive_index']
            response_lines.append(f"💵 Cheapest: Property {cheapest_idx + 1}")
            response_lines.append(f"💸 Most expensive: Property {expensive_idx + 1}")

        if 'value' in insights:
            best_value_idx = insights['value']['best_value_index']
            best_value = insights['value']['best_value_per_sqm']
            response_lines.append(f"⭐ Best value: Property {best_value_idx + 1} ({int(best_value):,} EGP/sqm)")

        if 'area' in insights:
            largest_idx = insights['area']['largest_index']
            response_lines.append(f"📏 Largest: Property {largest_idx + 1}")

    # Recommendations
    if recommendations:
        response_lines.append("\n🎯 Recommendations:")
        for rec in recommendations:
            response_lines.append(f"• {rec}")

    return "\n".join(response_lines)


if __name__ == "__main__":
    test_comparison()