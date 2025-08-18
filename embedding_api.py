from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
import torch
import os
import time
import uuid
import tempfile
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import logging
from text_normalizer import update_dictionary_from_files, normalize_query
from collections import defaultdict
from semantic_text_splitter import TextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Qdrant setup
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "chatbot_collection_v3"
VECTOR_SIZE = 1024
MAX_TOKENS = 500  # Tokens per chunk
OVERLAP = 100  # Overlap tokens

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# Embedding model
model_name = "jinaai/jina-embeddings-v3"
model = SentenceTransformer(model_name, trust_remote_code=True)
splitter = TextSplitter(capacity=MAX_TOKENS, overlap=OVERLAP)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class SearchRequest(BaseModel):
    query: str
    filename: Optional[str] = None

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
    query_vector = model.encode([f"query: {cleaned_query}"], task="text-matching")[0]
    query_tokens = cleaned_query.lower().split()

    # ------------------------------
    # Nếu có filename → dùng luôn để lọc (giữ nguyên)
    # ------------------------------
    qdrant_filter = None
    if req.filename:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.file_name",
                    match=MatchValue(value=req.filename)
                )
            ]
        )
        logger.info(f"Applying filename filter from request: {req.filename}")

    # ------------------------------
    # Search trong Qdrant: Tăng limit lên 200 để cải thiện recall
    # ------------------------------
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

    # Lấy candidates
    candidates = [(hit.payload.get("content", ""), hit.payload.get("metadata", {})) for hit in search_result]
    candidate_texts = [content for content, meta in candidates]
    candidate_vecs = np.array([hit.vector for hit in search_result], dtype=np.float32)

    # BM25 scoring
    bm25_scores = compute_bm25_scores(query_tokens, candidate_texts)
    if np.max(bm25_scores) - np.min(bm25_scores) != 0:
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    else:
        bm25_norm = np.zeros_like(bm25_scores)

    # Vector similarity
    vec_sim = cosine_similarity([query_vector], candidate_vecs)[0]

    # Hybrid score: Tăng alpha lên 0.5 để cân bằng BM25 và vector
    alpha = 0.4  # Có thể dynamic dựa trên len(query_tokens)
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * vec_sim

    # ------------------------------
    # Group by file và chọn file tốt nhất dựa trên MAX hybrid score
    # ------------------------------
    file_to_scores = defaultdict(list)
    file_to_candidates = defaultdict(list)
    file_to_vecs = defaultdict(list)

    for idx, score in enumerate(hybrid_scores):
        meta = candidates[idx][1]
        file_name = meta.get("file_name", "unknown")
        file_to_scores[file_name].append(score)
        file_to_candidates[file_name].append(candidates[idx])
        file_to_vecs[file_name].append(candidate_vecs[idx])

    # Chọn file có chunk với MAX score (thay vì average)
    file_max_scores = {file: np.max(scores) for file, scores in file_to_scores.items() if scores}
    if not file_max_scores:
        return {"results": []}

    best_file = max(file_max_scores, key=file_max_scores.get)
    logger.info(f"Best file selected: {best_file} with max score: {file_max_scores[best_file]:.4f}")

    # Lấy candidates và vecs chỉ từ best_file
    selected_candidates = file_to_candidates[best_file]
    selected_vecs = np.array(file_to_vecs[best_file], dtype=np.float32)
    selected_hybrid_scores = [hybrid_scores[idx] for idx, cand in enumerate(candidates) if cand[1].get("file_name") == best_file]

    # Sort theo hybrid score descending
    sorted_indices = np.argsort(selected_hybrid_scores)[::-1]
    sorted_candidates = [selected_candidates[i] for i in sorted_indices]
    sorted_vecs = selected_vecs[sorted_indices]

    # Áp dụng MMR re-ranking trên chunks của best_file (top_k=7, giảm lambda để ưu tiên relevance)
    mmr_selected = mmr(
        np.array(query_vector, dtype=np.float32),
        sorted_vecs,
        sorted_candidates,
        top_k=7,
        lambda_param=0.5  # Giảm lambda để ưu tiên relevance hơn trong cùng file
    )

    # ------------------------------
    # Merge thông minh: Sort chunks theo thứ tự trong file (nếu có chunk_index)
    # ------------------------------
    merged_results = []
    # Sort mmr_selected theo chunk_index (nếu có) để giữ ngữ cảnh
    sorted_mmr = sorted(
        mmr_selected,
        key=lambda x: x[1].get("chunk_index", float('inf'))  # Nếu không có chunk_index, đẩy xuống cuối
    )
    
    buffer_text = []
    for text, meta in sorted_mmr:
        buffer_text.append(text)
    
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
        chunks = []
        for doc in docs:
            for i, chunk_text in enumerate(splitter.chunks(doc.page_content)):
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "file_name": file.filename,
                            "upsert_timestamp": int(time.time()),
                            "chunk_index": i,
                        },
                    )
                )

        # Add file_name to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["file_name"] = file.filename
            chunk.metadata["upsert_timestamp"] = int(time.time())
            chunk.metadata["chunk_index"] = i
        # Batch embed
        texts = [f"passage: {chunk.page_content}" for chunk in chunks]
        vectors = model.encode(texts, task="text-matching")

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
            "chunk_size": MAX_TOKENS,  # tokens
            "overlap": OVERLAP
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_path)