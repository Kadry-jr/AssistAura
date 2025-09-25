import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

EMBED_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local')

if EMBED_PROVIDER == 'local':
    from sentence_transformers import SentenceTransformer
    MODEL_NAME = os.getenv('HF_LOCAL_MODEL', 'all-MiniLM-L6-v2')
    model = SentenceTransformer(MODEL_NAME)
else:
    import openai
    openai.api_key = os.getenv('OPENAI_API_KEY')


def get_embedding(text: str) -> List[float]:
    if EMBED_PROVIDER == 'local':
        v = model.encode([text], convert_to_numpy=True)[0]
        return v.tolist()
    else:
        resp = openai.Embedding.create(model=os.getenv('OPENAI_EMBEDDING_MODEL'), input=text)
        return resp['data'][0]['embedding']


def get_embedding_batch(texts: List[str]) -> List[List[float]]:
    if EMBED_PROVIDER == 'local':
        arr = model.encode(texts, convert_to_numpy=True)
        return [a.tolist() for a in arr]
    else:
        resp = openai.Embedding.create(model=os.getenv('OPENAI_EMBEDDING_MODEL'), input=texts)
        return [d['embedding'] for d in resp['data']]