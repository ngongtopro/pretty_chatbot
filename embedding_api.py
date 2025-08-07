from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import AutoTokenizer, AutoModel
import torch
import os
import time
import uuid
from langchain_community.document_loaders import TextLoader
import tempfile
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Qdrant setup
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "chatbot_collection_v2"
VECTOR_SIZE = 1024

try:
    qdrant.get_collection(COLLECTION_NAME)
except:
    qdrant.recreate_collection(
        COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# Embedding model
model_name = "intfloat/multilingual-e5-large-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


class SearchRequest(BaseModel):
    query: str


def embed_text(text: str) -> List[float]:
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        model_output = model(**encoded_input)
        embeddings = model_output.last_hidden_state.mean(dim=1).squeeze()
        return torch.nn.functional.normalize(embeddings, p=2, dim=0).tolist()


@app.post("/search")
async def search(req: SearchRequest):
    query = req.query
    query_vector = embed_text(f"query: {query}")

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        with_payload=True,
    )

    return {
        "results": [
            point.payload["content"]
            for point in search_result if "content" in point.payload
        ]
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1].lower()
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        # Load file
        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_ext == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif file_ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        points = []
        for i, chunk in enumerate(chunks):
            text = f"passage: {chunk.page_content}"
            vector = embed_text(text)
            point = PointStruct(
                id=int(time.time() * 1000) + i,
                vector=vector,
                payload={
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
            )
            points.append(point)

        qdrant.upsert(COLLECTION_NAME, points=points)

        return {"status": "completed", "vectors": len(points)}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_path)
