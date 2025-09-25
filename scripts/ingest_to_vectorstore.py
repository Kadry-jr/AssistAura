import os
import sys
import shutil
import hashlib
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

import pandas as pd
from dotenv import load_dotenv
from app.services.embeddings import get_embedding_batch
from app.services.vector_store import ChromaVectorStore

load_dotenv()

CSV = os.getenv("PROPERTIES_CSV", "data/properties_cleaned.csv")
BATCH = int(os.getenv("INGEST_BATCH_SIZE", 128))
RESET_DB = os.getenv("RESET_DB", "false").lower() == "true"
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

print(f"Loading CSV: {CSV}")
df = pd.read_csv(CSV)

# Drop duplicates (keep first occurrence)
df = df.drop_duplicates()

# Ensure there is an 'id' column (use numeric incremental IDs)
if "id" not in df.columns:
    df.insert(0, "id", range(1, len(df) + 1))

# Build embedding text (richer context)
df["embedding_text"] = (
    "Title: " + df["title"].astype(str) + ", "
    "Location: " + df["location"].astype(str) + ", "
    "Beds: " + df["beds"].astype(str) + ", "
    "Baths: " + df["baths"].astype(str) + ", "
    "Area: " + df["area_m2"].astype(str) + " sqm, "
    "Price: " + df["price_egp"].astype(str) + " EGP, "
    "Payment Plan: " + df["payment_plan"].astype(str) + ". "
    "Details: " + df["details"].astype(str)
)

# Prepare fields
texts = df["embedding_text"].astype(str).fillna("").tolist()
ids = df["id"].astype(str).tolist()

# Metadata for filtering
metadatas = df[
    ["url", "title", "location", "property_type", "price_egp", "area_m2", "beds", "baths", "project", "city", "country"]
].to_dict(orient="records")

# Reset DB if requested
if RESET_DB and os.path.exists(CHROMA_DIR):
    print("RESET_DB is true → clearing Chroma database...")
    shutil.rmtree(CHROMA_DIR)

vs = ChromaVectorStore.persist_dir(CHROMA_DIR)

# Upsert in batches
for i in range(0, len(texts), BATCH):
    batch_texts = texts[i : i + BATCH]
    batch_ids = ids[i : i + BATCH]
    batch_metadatas = metadatas[i : i + BATCH]

    embeddings = get_embedding_batch(batch_texts)
    vs.upsert(batch_ids, embeddings, batch_metadatas, batch_texts)

    print(f"Upserted batch {i // BATCH + 1} ({len(batch_texts)} items)")

print("Ingestion complete.")
