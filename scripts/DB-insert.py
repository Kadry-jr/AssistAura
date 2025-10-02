import os
import re
import random
import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('db_operations.log'),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

# Database connection
DB_URL = os.getenv('DB_URL')
if not DB_URL:
    raise ValueError("DB_URL environment variable is not set. Please check your .env file.")

engine = create_engine(DB_URL)

# Property type mapping from CSV to database enum values
PROPERTY_TYPE_MAPPING = {
    'apartment': 'APARTMENT',
    'villa': 'VILLA',
    'house': 'HOUSE',
    'condo': 'APARTMENT',
    'townhouse': 'TOWNHOUSE',
    'studio': 'STUDIO',
    'penthouse': 'PENTHOUSE',
    'duplex': 'DUPLEX',
    'mansion': 'MANSION',
    'loft': 'LOFT',
    'flat': 'APARTMENT',
    'office': 'OFFICE',
    'land': 'LAND',
    'warehouse': 'WAREHOUSE',
    'commercial': 'COMMERCIAL',
    'retail': 'RETAIL',
    'Retail': 'RETAIL'
}


def clean_text(text):
    """Clean and standardize text fields"""
    if pd.isna(text):
        return ""
    return str(text).strip()


def parse_price(price_str):
    """Parse price string and convert to float"""
    if pd.isna(price_str):
        return None

    try:
        # Remove any non-digit characters except decimal point
        price_str = re.sub(r'[^\d.]', '', str(price_str))
        return float(price_str)
    except (ValueError, TypeError):
        return None


def parse_area(area_str):
    """Parse area string and convert to float"""
    if pd.isna(area_str):
        return None
    try:
        # Extract numbers and decimal point
        area_str = re.sub(r'[^\d.]', '', str(area_str))
        return float(area_str)
    except (ValueError, TypeError):
        return None


def generate_egyptian_phone_number():
    """Generate a random valid Egyptian phone number"""
    mobile_prefixes = ['10', '11', '12', '15']
    prefix = random.choice(mobile_prefixes)
    number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
    return f"+20{prefix}{number}"


def generate_egypt_coordinates():
    """Generate random latitude and longitude within Egypt's boundaries"""
    latitude = random.uniform(22.0, 31.5)
    longitude = random.uniform(25.0, 35.0)
    return latitude, longitude


def generate_profile_pic_url(first_name, last_name):
    """Generate a profile picture URL with initials"""
    initials = f"{first_name[0]}{last_name[0] if last_name else first_name[0]}".upper()
    colors = ['1f8a70', '00425a', '2e4f4f', '0e8388', '3a4f7a']
    bg_color = random.choice(colors)
    return f"https://ui-avatars.com/api/?name={initials}&background={bg_color}&color=fff&size=256"


def create_user(conn, project_name):
    """Create a new user for a project"""
    try:
        # Generate a username from project name
        base_username = re.sub(r'[^a-zA-Z0-9]', '', project_name).lower()[:20]
        username = base_username
        counter = 1

        # Ensure username is unique
        while True:
            result = conn.execute(
                text("SELECT COUNT(*) FROM users WHERE username = :username"),
                {"username": username}
            ).scalar()
            if result == 0:
                break
            username = f"{base_username}{counter}"
            counter += 1

        # Generate user data
        first_name = project_name.split()[0] if project_name else "Property"
        last_name = ' '.join(project_name.split()[1:]) if project_name and len(project_name.split()) > 1 else "Owner"
        email = f"{username}@example.com"
        phone = generate_egyptian_phone_number()
        profile_pic = generate_profile_pic_url(first_name, last_name)

        # Create user
        user_result = conn.execute(
            text("""
                INSERT INTO users (
                    email, username, password, first_name, last_name, 
                    role, created_at, updated_at, phone, profile_picture_url
                ) VALUES (
                    :email, :username, :password, :first_name, :last_name,
                    'PROVIDER', NOW(), NOW(), :phone, :profile_picture_url
                )
                RETURNING user_id;
            """),
            {
                "email": email,
                "username": username,
                "password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone,
                "profile_picture_url": profile_pic
            }
        )
        return user_result.scalar()
    except Exception as e:
        logging.error(f"Error creating user: {str(e)}")
        raise


def create_provider(conn, user_id, project_name, location):
    """Create a provider record for a user"""
    try:
        conn.execute(
            text("""
                INSERT INTO providers (
                    user_id, company_name, company_address, provider_status
                ) VALUES (
                    :user_id, :company_name, :company_address, 'ACCEPTED'
                )
            """),
            {
                "user_id": user_id,
                "company_name": project_name or "Real Estate Company",
                "company_address": location or "Cairo, Egypt"
            }
        )
    except Exception as e:
        logging.error(f"Error creating provider: {str(e)}")
        raise


def validate_property_data(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and clean property data before insertion."""
    try:
        # Required fields
        required_fields = ['location', 'title']
        for field in required_fields:
            if not row.get(field):
                logging.warning(f"Missing required field: {field}")
                return None

        # Clean and validate data
        data = {
            'address': clean_text(row.get('location', ''))[:255],
            'area': parse_area(row.get('area_m2') or row.get('area', '')),
            'description': clean_text(row.get('details', ''))[:1000],
            'price': parse_price(row.get('price_egp') or row.get('price')),
            'title': clean_text(row.get('title', 'New Property'))[:100],
            'prop_type': get_property_type(row),
            'purpose': 'SALE' if random.random() < 0.8 else 'RENT'
        }

        # Validate numeric fields
        if data['price'] is None or data['price'] <= 0:
            logging.warning(f"Invalid price: {row.get('price_egp') or row.get('price')}")
            return None

        if data['area'] is None or data['area'] <= 0:
            logging.warning(f"Invalid area: {row.get('area_m2') or row.get('area')}")
            return None

        return data

    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        return None


def get_property_type(row: Dict[str, Any]) -> str:
    """Get and validate property type."""
    prop_type = 'APARTMENT'
    if 'property_type' in row and row['property_type']:
        prop_type = row['property_type'].upper()
        prop_type = PROPERTY_TYPE_MAPPING.get(prop_type.lower(), 'APARTMENT')
    return prop_type


def insert_property(conn, row: Dict[str, Any], user_id: int, max_retries: int = 3) -> Optional[int]:
    """Safely insert a property with retry logic."""
    for attempt in range(max_retries):
        try:
            property_data = validate_property_data(row)
            if not property_data:
                return None

            # Generate coordinates
            lat, lon = generate_egypt_coordinates()

            # Insert property details
            property_details_id = conn.execute(
                text("""
                    INSERT INTO property_details (
                        address, area, description, price, title, 
                        type, purpose, latitude, longitude
                    ) VALUES (
                        :address, :area, :description, :price, :title, 
                        :prop_type, :purpose, :latitude, :longitude
                    )
                    RETURNING property_details_id;
                """),
                {
                    **property_data,
                    'latitude': lat,
                    'longitude': lon
                }
            ).scalar()

            # Insert property
            conn.execute(
                text("""
                    INSERT INTO properties (
                        property_details_id, user_id, property_status, 
                        created_at, updated_at
                    ) VALUES (
                        :property_details_id, :user_id, 'AVAILABLE', 
                        NOW(), NOW()
                    )
                """),
                {
                    "property_details_id": property_details_id,
                    "user_id": user_id
                }
            )

            logging.info(f"Successfully inserted property {property_details_id}")
            return property_details_id

        except IntegrityError as e:
            logging.error(f"Integrity error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached, giving up")
                return None
            continue

        except SQLAlchemyError as e:
            logging.error(f"Database error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached, giving up")
                return None
            raise

        except Exception as e:
            logging.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logging.error("Max retries reached, giving up")
                return None
            raise

    return None


def process_properties(conn, df, project_users):
    """Process all properties with proper transaction management."""
    success_count = 0
    total_properties = len(df)

    for idx, row in df.iterrows():
        try:
            project = row.get('project')
            if not project or str(project).strip() == '':
                logging.warning(f"Skipping row {idx + 1}: Empty project name")
                continue

            user_id = project_users.get(project)
            if not user_id:
                logging.warning(f"Skipping row {idx + 1}: No user found for project: {project}")
                continue

            # Process each property in its own transaction
            with conn.begin():
                prop_id = insert_property(conn, row, user_id)
                if prop_id:
                    success_count += 1
                    if success_count % 10 == 0:
                        logging.info(f"Processed {success_count} properties...")
                else:
                    logging.warning(f"Failed to insert property {idx + 1}")

        except Exception as e:
            logging.error(f"Error processing property {idx + 1}: {str(e)}")
            continue

    return success_count


def main():
    try:
        # Load data
        df = pd.read_csv("./data/properties_cleaned.csv")
        logging.info(f"Loaded {len(df)} properties from CSV")

        with engine.connect() as conn:
            # Create a mapping of project names to user IDs
            project_users = {}

            # Process each unique project
            unique_projects = df['project'].dropna().unique()
            logging.info(f"Found {len(unique_projects)} unique projects")

            for project in unique_projects:
                if not project or str(project).strip() == '':
                    continue

                logging.info(f"Processing project: {project}")
                try:
                    with conn.begin():
                        user_id = create_user(conn, project)
                        create_provider(conn, user_id, project,
                                        df[df['project'] == project]['location'].iloc[0])
                        project_users[project] = user_id
                        logging.info(f"Created user {user_id} for project: {project}")
                except Exception as e:
                    logging.error(f"Failed to create user for project {project}: {str(e)}")
                    continue

            # Process all properties
            total_properties = len(df)
            logging.info(f"\nProcessing {total_properties} properties...")

            success_count = process_properties(conn, df, project_users)

            logging.info(f"\n✅ Successfully processed {success_count} out of {total_properties} properties")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise


if __name__ == "__main__":
    main()