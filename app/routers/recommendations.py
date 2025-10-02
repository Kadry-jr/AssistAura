from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from sqlalchemy import text
import random

from ..core.database import get_db

router = APIRouter(
    prefix="/recommendations",
    tags=["Recommendations"],
    responses={404: {"description": "Not found"}},
)

def row_to_dict(row) -> Dict[str, Any]:
    """Convert SQLAlchemy row to dictionary"""
    return {column: getattr(row, column) for column in row._fields}

#done
@router.get("/similar/location/{property_id}", response_model=List[dict])
async def get_similar_properties_same_location(property_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """
    Get properties with the same address and similar price (±30%).
    
    - **property_id**: ID of the property to find similar ones for
    - **limit**: Maximum number of similar properties to return (default: 5)
    """
    try:
       # print(f"[DEBUG] Starting location-based recommendations for property {property_id}")
        
        # 1. Get target property details (excluding latitude/longitude)
        target_query = text("""
            SELECT 
                pd.property_details_id,
                pd.address,
                pd.area,
                pd.description,
                pd.price,
                pd.title,
                pd.type,
                pd.purpose,
                p.property_status
            FROM property_details pd
            JOIN properties p ON p.property_details_id = pd.property_details_id
            WHERE pd.property_details_id = :property_id
        """)
        
        target_result = db.execute(target_query, {"property_id": property_id}).fetchone()
        if not target_result:
            print(f"[ERROR] Property {property_id} not found")
            raise HTTPException(status_code=404, detail="Property not found")
        
        target = row_to_dict(target_result)
       # print(f"[DEBUG] Found target property with address: {target.get('address')}")
        
        if not target.get('address'):
            print("[WARNING] Target property has no address data")
            return []
            
        # 2. Calculate price range (±30% of target price)
        price_range_low = target['price'] * 0.7
        price_range_high = target['price'] * 1.3
        
      #  print(f"[DEBUG] Price range: {price_range_low} - {price_range_high}")
       # print(f"[DEBUG] Looking for properties with address: {target['address']}")

        # 3. Query for properties with same address and similar price (excluding latitude/longitude)
        similar_query = text("""
            SELECT 
                pd.property_details_id,
                pd.address,
                pd.area,
                pd.description,
                pd.price,
                pd.title,
                pd.type,
                pd.purpose,
                p.property_status,
                m.url as image_url
            FROM property_details pd
            JOIN properties p ON p.property_details_id = pd.property_details_id
            LEFT JOIN media m ON m.property_details_id = pd.property_details_id AND m.media_id = (
                SELECT MIN(media_id) 
                FROM media 
                WHERE property_details_id = pd.property_details_id
            )
            WHERE pd.property_details_id != :property_id
            AND LOWER(TRIM(pd.address)) = LOWER(TRIM(:target_address))
            AND pd.price BETWEEN :price_low AND :price_high
            ORDER BY ABS(pd.price - :target_price)
            LIMIT 800  -- Fetch up to 800 results to ensure we have enough after deduplication
        """)
        
        query_params = {
            "property_id": property_id,
            "price_low": price_range_low,
            "price_high": price_range_high,
            "target_price": target['price'],
            "target_address": target['address'].strip(),
        }
        
        #print(f"[DEBUG] Query params: {query_params}")
        
        # 4. Fetch all results and remove duplicates
        similar_results = db.execute(similar_query, query_params).fetchall()
       # print(f"[DEBUG] Found {len(similar_results)} similar properties before deduplication")
        
        # 5. Remove duplicates based on property details (excluding ID)
        unique_properties = {}
        for prop in similar_results:
            prop_dict = row_to_dict(prop)
            # Create a unique key based on property details (excluding ID)
            prop_key = (
                prop_dict.get('address', '').lower().strip(),
                prop_dict.get('price'),
                prop_dict.get('area'),
                prop_dict.get('type'),
                prop_dict.get('purpose')
            )
            
            # Only add if we haven't seen this combination before
            if prop_key not in unique_properties:
                unique_properties[prop_key] = prop_dict
        
        # 6. Convert to list, shuffle, and apply the limit
        unique_results = list(unique_properties.values())
        random.shuffle(unique_results)
        unique_results = unique_results[:limit]
       # print(f"[DEBUG] Returning {len(unique_results)} unique properties after deduplication and shuffling")
        
        return unique_results
        
    except HTTPException as he:
        print(f"[HTTPException] {str(he)}")
        raise
    except Exception as e:
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[ERROR] {error_details}")
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Error fetching similar properties",
                "error": str(e),
                "type": type(e).__name__
            }
        )

@router.get("/similar/anywhere/{property_id}", response_model=List[dict])
async def get_similar_properties_any_location(property_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """
    Get properties with similar price range (±30%) regardless of location.
    
    - **property_id**: ID of the property to find similar ones for
    - **limit**: Maximum number of similar properties to return (default: 5)
    """
    try:
       # print(f"[DEBUG] Starting price-based recommendations for property {property_id}")
        
        # 1. Get target property details
        target_query = text("""
            SELECT pd.*, p.property_status 
            FROM property_details pd
            JOIN properties p ON p.property_details_id = pd.property_details_id
            WHERE pd.property_details_id = :property_id
        """)
        
        target_result = db.execute(target_query, {"property_id": property_id}).fetchone()
        if not target_result:
            print(f"[ERROR] Property {property_id} not found")
            raise HTTPException(status_code=404, detail="Property not found")
        
        target = row_to_dict(target_result)
       # print(f"[DEBUG] Found target property with price: {target.get('price')}")
        
        # 2. Calculate price range (±10% of target price)
        price_range_low = target['price'] * 0.9
        price_range_high = target['price'] * 1.1
        
        #print(f"[DEBUG] Price range: {price_range_low} - {price_range_high}")

        # 3. Query for similar properties (price only)
        similar_query = text("""
            SELECT 
                pd.*, 
                p.property_status, 
                m.url as image_url
            FROM property_details pd
            JOIN properties p ON p.property_details_id = pd.property_details_id
            LEFT JOIN media m ON m.property_details_id = pd.property_details_id AND m.media_id = (
                SELECT MIN(media_id) 
                FROM media 
                WHERE property_details_id = pd.property_details_id
            )
            WHERE pd.property_details_id != :property_id
            AND pd.price BETWEEN :price_low AND :price_high
            AND pd.type = :property_type
            AND pd.purpose = :purpose
            ORDER BY ABS(pd.price - :target_price)
            LIMIT 800  -- Fetch up to 800 results to ensure we have enough after deduplication
        """)
        
        query_params = {
            "property_id": property_id,
            "price_low": price_range_low,
            "price_high": price_range_high,
            "target_price": target['price'],
            "property_type": target['type'],
            "purpose": target['purpose']
        }
        
        #print(f"[DEBUG] Query params: {query_params}")
        
        # 4. Fetch all results and remove duplicates
        similar_results = db.execute(similar_query, query_params).fetchall()
       # print(f"[DEBUG] Found {len(similar_results)} similar properties before deduplication")
        
        # 5. Remove duplicates based on property details (excluding ID)
        unique_properties = {}
        for prop in similar_results:
            prop_dict = row_to_dict(prop)
            # Create a unique key based on property details (excluding ID)
            prop_key = (
                prop_dict.get('address', '').lower().strip(),
                prop_dict.get('price'),
                prop_dict.get('area'),
                prop_dict.get('type'),
                prop_dict.get('purpose')
            )
            
            # Only add if we haven't seen this combination before
            if prop_key not in unique_properties:
                unique_properties[prop_key] = prop_dict
        
        # 6. Convert to list, shuffle, and apply the limit
        unique_results = list(unique_properties.values())
        random.shuffle(unique_results)
        unique_results = unique_results[:limit]
       # print(f"[DEBUG] Returning {len(unique_results)} unique properties after deduplication and shuffling")
        
        return unique_results
        
    except HTTPException as he:
        print(f"[HTTPException] {str(he)}")
        raise
    except Exception as e:
        import traceback
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        print(f"[ERROR] {error_details}")
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Error fetching similar properties",
                "error": str(e),
                "type": type(e).__name__
            }
        )

@router.get("/user/{user_id}", response_model=List[dict])
async def get_user_based_recommendations(user_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """
    Get property recommendations for a user based on similar users' preferences.
    
    - **user_id**: ID of the user to get recommendations for
    - **limit**: Maximum number of recommendations to return (default: 5)
    """
    try:
        # Check if user exists
        user_check = db.execute(
            text("SELECT user_id FROM users WHERE user_id = :user_id"),
            {"user_id": user_id}
        ).fetchone()
        
        if not user_check:
            raise HTTPException(status_code=404, detail="User not found")

        # Get properties this user has favorited
        user_favorites = db.execute(
            text("SELECT property_id FROM favorites WHERE customer_id = :user_id"),
            {"user_id": user_id}
        ).fetchall()
        
        if not user_favorites:
            return []

        user_fav_ids = [f[0] for f in user_favorites]

        # Find other users who favorited the same properties
        other_user_favs = db.execute(
            text("""
                SELECT DISTINCT customer_id 
                FROM favorites 
                WHERE property_id = ANY(:fav_ids) 
                AND customer_id != :user_id
            """),
            {"fav_ids": user_fav_ids, "user_id": user_id}
        ).fetchall()
        
        if not other_user_favs:
            return []
            
        other_user_ids = [f[0] for f in other_user_favs]

        # Get properties these other users liked, excluding already favorited ones
        recommended_query = text("""
            SELECT DISTINCT pd.*, p.property_status, m.url as image_url
            FROM property_details pd
            JOIN properties p ON p.property_details_id = pd.property_details_id
            JOIN favorites f ON f.property_id = pd.property_details_id
            LEFT JOIN media m ON m.property_details_id = pd.property_details_id AND m.media_id = (
                SELECT MIN(media_id) 
                FROM media 
                WHERE property_details_id = pd.property_details_id
            )
            WHERE f.customer_id = ANY(:user_ids)
            AND pd.property_details_id != ALL(:fav_ids)
            LIMIT :limit
        """)
        
        recommended_results = db.execute(
            recommended_query,
            {
                "user_ids": other_user_ids,
                "fav_ids": user_fav_ids,
                "limit": limit
            }
        ).fetchall()

        # Convert to list of dicts and shuffle
        recommended_properties = [row_to_dict(prop) for prop in recommended_results]
        random.shuffle(recommended_properties)
        
        return recommended_properties
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
