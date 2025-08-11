from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time
import uuid
import tempfile
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import emoji

app = FastAPI()

# Qdrant setup
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "chatbot_collection_v3"
VECTOR_SIZE = 1024

# Create collection if not exists
if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# Embedding model
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class SearchRequest(BaseModel):
    query: str

# ===========================
# Hàm làm sạch query
# ===========================
def clean_text(text: str) -> str:
    text = text.lower()
    text = emoji.replace_emoji(text, replace=" ")
    greetings = [
        r"\bxin chào\b", r"\bchào\b", r"\bhello+\b", r"\bhi+\b",
        r"\balo+\b", r"\bhey+\b", r"\bchao\b"
    ]
    for g in greetings:
        text = re.sub(g, " ", text)
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
                  r"ỳýỷỹỵđĐ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===========================
# Hàm tạo embedding
# ===========================
def embed_text(text: str) -> List[float]:
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze()
        norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=0)
        return norm_emb.tolist()

# ===========================
# Hàm auto chunk_overlap
# ===========================
def auto_select_overlap(text, max_chunk_size=1024):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        return 50
    token_counts = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences if s.strip()]
    avg_tokens = sum(token_counts) / len(token_counts)
    if avg_tokens > 30:
        overlap = 100
    elif avg_tokens >= 15:
        overlap = 70
    else:
        overlap = 50
    return min(overlap, max_chunk_size // 2)

# ===========================
# MMR re-ranking
# ===========================
def mmr(query_vec, doc_vecs, docs, top_k=5, lambda_param=0.7):
    selected, selected_idxs = [], []
    sim_to_query = cosine_similarity([query_vec], doc_vecs)[0]
    idx = int(np.argmax(sim_to_query))
    selected_idxs.append(idx)
    selected.append(docs[idx])

    while len(selected) < top_k and len(selected_idxs) < len(docs):
        rest = [i for i in range(len(docs)) if i not in selected_idxs]
        mmr_scores = []
        for i in rest:
            diversity = max(cosine_similarity([doc_vecs[i]], [doc_vecs[j] for j in selected_idxs])[0]) if selected_idxs else 0
            score = lambda_param * sim_to_query[i] - (1 - lambda_param) * diversity
            mmr_scores.append((score, i))
        best_idx = max(mmr_scores, key=lambda x: x[0])[1]
        selected_idxs.append(best_idx)
        selected.append(docs[best_idx])
    return selected

# ===========================
# API Search
# ===========================
@app.post("/search")
async def search(req: SearchRequest):
    cleaned_query = clean_text(req.query)
    query_vector = embed_text(f"query: {cleaned_query}")

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        with_vectors=True
    )

    candidates = [(hit.payload.get("content", ""), hit.payload.get("metadata", {})) for hit in search_result]
    candidate_vecs = [hit.vector for hit in search_result]

    selected = mmr(
        np.array(query_vector, dtype=np.float32),
        np.array(candidate_vecs, dtype=np.float32),
        candidates,
        top_k=5
    )

    # Gộp chunk liền kề cùng metadata
    merged_results = []
    prev_meta, buffer_text = None, []
    for text, meta in selected:
        if prev_meta == meta:
            buffer_text.append(text)
        else:
            if buffer_text:
                merged_results.append(" ".join(buffer_text))
            buffer_text = [text]
            prev_meta = meta
    if buffer_text:
        merged_results.append(" ".join(buffer_text))

    return {"results": merged_results}

# ===========================
# API Upload
# ===========================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
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
        full_text = "\n".join([d.page_content for d in docs])
        overlap = auto_select_overlap(full_text, max_chunk_size=1024)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; "]
        )
        chunks = text_splitter.split_documents(docs)

        points = []
        for i, chunk in enumerate(chunks):
            text = f"passage: {chunk.page_content}"
            vector = embed_text(text)
            points.append(PointStruct(
                id=int(time.time() * 1000) + i,
                vector=vector,
                payload={"content": chunk.page_content, "metadata": chunk.metadata}
            ))

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return {"status": "completed", "vectors": len(points), "overlap_used": overlap}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_path)
