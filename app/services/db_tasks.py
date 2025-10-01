from sqlalchemy import text
from app.core.database import engine

def list_tables():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';"))
        return [row[0] for row in result]
