from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue, MatchText
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time
import uuid
import tempfile
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import emoji
import logging
from text_normalizer import update_dictionary_from_files, normalize_query

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Qdrant setup
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "chatbot_collection_v3"
VECTOR_SIZE = 1024

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

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

# ===========================
# API Upload từ điển
# ===========================
@app.post("/upload_dict")
async def upload_dict(files: List[UploadFile] = File(...)):
    temp_paths = []
    for file in files:
        ext = os.path.splitext(file.filename)[1]
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        temp_paths.append(temp_path)

    result = update_dictionary_from_files(temp_paths)

    for path in temp_paths:
        os.remove(path)

    return {"status": "dictionary updated", **result}


# ===========================
# Search API with Hybrid
# ===========================
@app.post("/search")
async def search(req: SearchRequest):
    start_time = time.time()
    logger.info(f"Raw question: {req.query}")
    cleaned_query = normalize_query(req.query)
    logger.info(f"Normalized question: {cleaned_query}")
    query_vector = embed_texts([f"query: {cleaned_query}"])[0]
    query_tokens = cleaned_query.lower().split()

    # Nếu có filename thì tạo filter
    qdrant_filter = None
    if getattr(req, "filename", None):
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.file_name",  # Trường metadata
                    match=MatchValue(value=req.filename)
                )
            ]
        )
        logger.info(f"Applying filename filter: {req.filename}")

    # Vector search to get candidates
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        with_vectors=True,
        query_filter=qdrant_filter
    )
    if not search_result:
        return {"results": []}
    candidates = [(hit.payload.get("content", ""), hit.payload.get("metadata", {})) for hit in search_result]
    candidate_texts = [content for content, meta in candidates]
    candidate_vecs = np.array([hit.vector for hit in search_result], dtype=np.float32)

    # BM25 on candidates
    bm25_scores = compute_bm25_scores(query_tokens, candidate_texts)
    # Normalize BM25 scores
    if np.max(bm25_scores) - np.min(bm25_scores) != 0:
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    else:
        bm25_norm = np.zeros_like(bm25_scores)
    
    # Vector similarities (already from cosine)
    vec_sim = cosine_similarity([query_vector], candidate_vecs)[0]

    # Hybrid scores: weighted sum (e.g., 0.4 BM25 + 0.6 Vector)
    alpha = 0.4
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * vec_sim

    # Sort candidates by hybrid scores
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    sorted_candidates = [candidates[i] for i in sorted_indices[:50]]  # Keep top 50 after hybrid
    sorted_vecs = candidate_vecs[sorted_indices[:50]]

    # MMR on hybrid-sorted candidates
    selected = mmr(
        np.array(query_vector, dtype=np.float32),
        sorted_vecs,
        sorted_candidates,
        top_k=5
    )

    # Merge consecutive chunks with same metadata
    merged_results = []
    prev_meta, buffer_text = None, []
    for text, meta in selected:
        logger.info(f"Meta: {meta}")
        logger.info(f"Text: {text}")
        if prev_meta == meta:
            buffer_text.append(text)
        else:
            if buffer_text:
                merged_results.append(" ".join(buffer_text))
            buffer_text = [text]
            prev_meta = meta
    if buffer_text:
        merged_results.append(" ".join(buffer_text))

    logger.info(f"Search time: {time.time() - start_time:.2f}s")
    return {"results": merged_results}

# ===========================
# Upload API
# ===========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    start_time = time.time()
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{file_ext}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
            qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )

        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_ext == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif file_ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

        docs = loader.load()

        # Token-based splitter
        text_splitter = TokenTextSplitter(
            chunk_size=300,  # Tokens
            chunk_overlap=45  # ~15%
        )
        chunks = text_splitter.split_documents(docs)

        # Add file_name to metadata
        for chunk in chunks:
            chunk.metadata["file_name"] = file.filename

        # Batch embed
        texts = [f"passage: {chunk.page_content}" for chunk in chunks]
        vectors = embed_texts(texts)

        points = []
        for i, vector in enumerate(vectors):
            points.append(PointStruct(
                id=uuid.uuid4().hex,  # Unique ID
                vector=vector,
                payload={"content": chunks[i].page_content, "metadata": chunks[i].metadata}
            ))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        
        logger.info(f"Upload time: {time.time() - start_time:.2f}s")
        return {
            "status": "completed",
            "vectors": len(points),
            "chunk_size": 300,  # tokens
            "overlap": 45
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_path)