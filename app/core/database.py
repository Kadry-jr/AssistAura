from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Create database engine using the URL from settings
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Simple function to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
