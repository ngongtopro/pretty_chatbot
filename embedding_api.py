from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import torch
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant setup
COLLECTION_NAME = "chatbot_collection_v3"
VECTOR_SIZE = 1024

def get_qdrant_client():
    """Get Qdrant client instance"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")
        return None

def ensure_collection_exists():
    """Ensure the collection exists in Qdrant"""
    qdrant = get_qdrant_client()
    if qdrant is None:
        return False
    
    try:
        if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
        return True
    except Exception as e:
        logger.warning(f"Could not ensure collection exists: {e}")
        return False

# Initialize qdrant variable for backwards compatibility
qdrant = None

# Embedding model
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SearchRequest(BaseModel):
    query: str
    filename: Optional[str] = None

# ===========================
# Batch Embedding helper
# ===========================
def embed_texts(texts: List[str], batch_size: int = 16) -> List[List[float]]:
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_input = tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True).to(device)
            model_output = model(**encoded_input)
            batch_emb = model_output.last_hidden_state.mean(dim=1)
            norm_emb = torch.nn.functional.normalize(batch_emb, p=2, dim=1)
            embeddings.extend(norm_emb.cpu().tolist())
    return embeddings

# ===========================
# Simple BM25 implementation for hybrid search
# ===========================
def compute_bm25_scores(query_tokens: List[str], docs: List[str], k1=1.5, b=0.75) -> np.ndarray:
    # Tokenize docs
    doc_tokens = [doc.lower().split() for doc in docs]
    
    # Compute term frequencies
    all_terms = set(query_tokens)
    for dt in doc_tokens:
        all_terms.update(dt)
    term_to_id = {term: idx for idx, term in enumerate(all_terms)}
    
    # TF matrix
    num_docs = len(doc_tokens)
    num_terms = len(term_to_id)
    tf = np.zeros((num_docs, num_terms))
    doc_lengths = np.array([len(dt) for dt in doc_tokens])
    avg_doc_len = np.mean(doc_lengths) if num_docs > 0 else 0
    
    for d_idx, dt in enumerate(doc_tokens):
        for term in dt:
            if term in term_to_id:
                tf[d_idx, term_to_id[term]] += 1
    
    # IDF
    df = np.sum(tf > 0, axis=0)
    idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
    
    # BM25 scores
    scores = np.zeros(num_docs)
    for q_term in query_tokens:
        if q_term in term_to_id:
            t_id = term_to_id[q_term]
            tf_term = tf[:, t_id]
            numerator = tf_term * (k1 + 1)
            denominator = tf_term + k1 * (1 - b + b * doc_lengths / avg_doc_len)
            scores += idf[t_id] * (numerator / denominator)
    
    return scores

# ===========================
# MMR re-ranking
# ===========================
def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, docs: List, top_k: int = 5, lambda_param: float = 0.7):
    selected, selected_idxs = [], []
    sim_to_query = cosine_similarity([query_vec], doc_vecs)[0]
    idx = int(np.argmax(sim_to_query))
    selected_idxs.append(idx)
    selected.append(docs[idx])

    while len(selected) < top_k and len(selected_idxs) < len(docs):
        rest = [i for i in range(len(docs)) if i not in selected_idxs]
        mmr_scores = []
        for i in rest:
            diversity = max(cosine_similarity([doc_vecs[i]], doc_vecs[selected_idxs])[0]) if selected_idxs else 0
            score = lambda_param * sim_to_query[i] - (1 - lambda_param) * diversity
            mmr_scores.append((score, i))
        if not mmr_scores:
            break
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_idxs.append(best_idx)
        selected.append(docs[best_idx])
    return selected
