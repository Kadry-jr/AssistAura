from fastapi import APIRouter
from app.services.db_tasks import list_tables

router = APIRouter(prefix="/db", tags=["Database"])

@router.get("/tables")
def get_tables():
    return {"tables": list_tables()}
