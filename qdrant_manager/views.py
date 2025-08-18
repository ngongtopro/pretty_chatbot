
# Standard library imports
import os
import uuid
import json
import time
import logging
import tempfile

# Third-party imports
import numpy as np
from torch import cosine_similarity
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue

# Django imports
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Q

# Local imports
from .models import QdrantCollection, QdrantPoint
from .forms import QdrantCollectionForm, QdrantPointForm, BulkDeleteForm, SearchForm
from .qdrant_service import QdrantService
from text_normalizer import update_dictionary_from_files, normalize_query
from embedding_api import COLLECTION_NAME, VECTOR_SIZE, compute_bm25_scores, mmr, get_qdrant_client, embed_texts, ensure_collection_exists

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collection_list(request):
    """Hiển thị danh sách các Qdrant collections"""
    qdrant = QdrantService()
    
    # Sync collections từ Qdrant server
    try:
        server_collections = qdrant.get_collections()
    except Exception as e:
        logger.warning(f"Cannot connect to Qdrant server: {e}")
        server_collections = []
    
    # Cập nhật database từ server
    for server_col in server_collections:
        collection, created = QdrantCollection.objects.get_or_create(
            name=server_col['name'],
            defaults={
                'vector_size': server_col.get('vector_size', 1536),
                'distance_metric': server_col.get('distance', 'cosine'),
                'qdrant_synced': True,
                'points_count': server_col['points_count'],
                'vectors_count': server_col['vectors_count']
            }
        )
        if not created:
            collection.vector_size = server_col.get('vector_size', collection.vector_size or 1536)
            collection.distance_metric = server_col.get('distance', collection.distance_metric or 'cosine')
            collection.points_count = server_col['points_count']
            collection.vectors_count = server_col['vectors_count']
            collection.qdrant_synced = True
            collection.save()
    
    collections = QdrantCollection.objects.all()
    
    # Xử lý tìm kiếm
    search_query = request.GET.get('search', '')
    if search_query:
        collections = collections.filter(
            Q(name__icontains=search_query) | 
            Q(description__icontains=search_query)
        )
    
    # Phân trang
    paginator = Paginator(collections, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_collections': QdrantCollection.objects.count(),
        'server_collections': server_collections
    }

    return render(request, 'qdrant_manager/collection_list.html', context)

def collection_create(request):
    """Tạo mới Qdrant collection"""
    if request.method == 'POST':
        form = QdrantCollectionForm(request.POST)
        if form.is_valid():
            collection = form.save(commit=False)
            
            # Tạo collection trên Qdrant server
            qdrant = QdrantService()
            success = qdrant.create_collection(
                name=collection.name,
                vector_size=collection.vector_size,
                distance=collection.distance_metric
            )
            
            if success:
                collection.qdrant_synced = True
                collection.save()
                messages.success(request, f'Collection "{collection.name}" đã được tạo thành công!')
                return redirect('qdrant_manager:collection_list')
            else:
                messages.error(request, f'Không thể tạo collection trên Qdrant server!')
    else:
        form = QdrantCollectionForm()
    
    return render(request, 'qdrant_manager/collection_form.html', {
        'form': form,
        'title': 'Tạo Collection Mới'
    })

def collection_edit(request, pk):
    """Chỉnh sửa Qdrant collection"""
    collection = get_object_or_404(QdrantCollection, pk=pk)
    
    if request.method == 'POST':
        form = QdrantCollectionForm(request.POST, instance=collection)
        if form.is_valid():
            form.save()
            messages.success(request, f'Collection "{collection.name}" đã được cập nhật!')
            return redirect('qdrant_manager:collection_list')
    else:
        form = QdrantCollectionForm(instance=collection)
    
    return render(request, 'qdrant_manager/collection_form.html', {
        'form': form,
        'title': f'Chỉnh sửa Collection "{collection.name}"',
        'collection': collection
    })

@require_POST
def collection_delete(request, pk):
    """Xóa Qdrant collection"""
    collection = get_object_or_404(QdrantCollection, pk=pk)
    collection_name = collection.name
    
    # Xóa từ Qdrant server
    qdrant = QdrantService()
    success = qdrant.delete_collection(collection_name)
    
    if success:
        collection.delete()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'message': f'Collection "{collection_name}" đã được xóa!'})
        messages.success(request, f'Collection "{collection_name}" đã được xóa!')
    else:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'message': f'Không thể xóa collection từ Qdrant server!'})
        messages.error(request, f'Không thể xóa collection từ Qdrant server!')
    
    return redirect('qdrant_manager:collection_list')

def point_list(request, collection_pk):
    """Hiển thị danh sách points trong một collection"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    
    # Sync points từ Qdrant server
    qdrant = QdrantService()
    server_points = qdrant.get_points(collection.name)
    
    # Cập nhật database từ server
    for server_point in server_points:
        point, created = QdrantPoint.objects.get_or_create(
            collection=collection,
            point_id=str(server_point['id']),
            defaults={
                'content': server_point['payload'].get('content', ''),
                'title': server_point['payload'].get('title', ''),
                'source': server_point['payload'].get('source', ''),
                'file_name': server_point['payload'].get('metadata', {}).get('file_name', ''),
                'category': server_point['payload'].get('category', ''),
                'tags': server_point['payload'].get('tags', []),
                'metadata': {k: v for k, v in server_point['payload'].get('metadata', {}).items() 
                           if k not in ['content', 'title', 'source', 'file_name', 'category', 'tags']},
                'qdrant_synced': True
            }
        )
        if not created:
            # Cập nhật thông tin point đã tồn tại
            point.content = server_point['payload'].get('content', point.content)
            point.title = server_point['payload'].get('title', point.title)
            point.source = server_point['payload'].get('source', point.source)
            point.file_name = server_point['payload'].get('metadata', {}).get('file_name', ''),
            point.category = server_point['payload'].get('category', point.category)
            point.tags = server_point['payload'].get('tags', point.tags)
            point.metadata = {k: v for k, v in server_point['payload'].get('metadata', {}).items() 
                            if k not in ['content', 'title', 'source', 'file_name', 'category', 'tags']}
            point.qdrant_synced = True
            point.save()
    
    points = collection.points.all()
    
    # Xử lý tìm kiếm
    search_query = request.GET.get('search', '')
    if search_query:
        points = points.filter(
            Q(point_id__icontains=search_query) |
            Q(content__icontains=search_query) |
            Q(title__icontains=search_query) |
            Q(file_name__icontains=search_query)
        )
    
    # Phân trang
    paginator = Paginator(points, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'collection': collection,
        'page_obj': page_obj,
        'search_query': search_query,
        'total_points': collection.points.count(),
        'bulk_delete_form': BulkDeleteForm(),
        'search_form': SearchForm()
    }
    
    return render(request, 'qdrant_manager/point_list.html', context)

def point_create(request, collection_pk):
    """Tạo mới point trong collection"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    
    if request.method == 'POST':
        form = QdrantPointForm(request.POST)
        if form.is_valid():
            point = form.save(commit=False)
            point.collection = collection
            
            # Tạo vector từ content nếu được yêu cầu
            qdrant = QdrantService()
            if form.cleaned_data.get('generate_vector', True):
                vector = qdrant.generate_dummy_vector(collection.vector_size)
                point.vector_data = vector
            else:
                vector = point.vector_data or qdrant.generate_dummy_vector(collection.vector_size)
            
            # Thêm point vào Qdrant server
            success = qdrant.add_point(
                collection_name=collection.name,
                point_id=point.point_id,
                vector=vector,
                payload=point.payload
            )
            
            if success:
                point.qdrant_synced = True
                point.save()
                messages.success(request, 'Point đã được tạo thành công!')
                return redirect('qdrant_manager:point_list', collection_pk=collection.pk)
            else:
                messages.error(request, 'Không thể thêm point vào Qdrant server!')
    else:
        # Tạo point_id tự động
        form = QdrantPointForm(initial={'point_id': str(uuid.uuid4())})
    
    return render(request, 'qdrant_manager/point_form.html', {
        'form': form,
        'collection': collection,
        'title': 'Thêm Point Mới'
    })

def point_detail(request, collection_pk, pk):
    """Chi tiết point"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    point = get_object_or_404(QdrantPoint, pk=pk, collection=collection)
    
    # Lấy thông tin chi tiết từ Qdrant server
    qdrant = QdrantService()
    qdrant_point_data = None
    
    try:
        qdrant_point_data = qdrant.get_point(collection.name, point.point_id)
    except Exception as e:
        messages.warning(request, f'Không thể lấy dữ liệu từ Qdrant server: {str(e)}')
    
    context = {
        'collection': collection,
        'point': point,
        'qdrant_point_data': qdrant_point_data,
        'title': f'Chi tiết Point "{point.point_id}"'
    }
    
    return render(request, 'qdrant_manager/point_detail.html', context)

def point_edit(request, collection_pk, pk):
    """Chỉnh sửa point"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    point = get_object_or_404(QdrantPoint, pk=pk, collection=collection)
    
    if request.method == 'POST':
        form = QdrantPointForm(request.POST, instance=point)
        if form.is_valid():
            point = form.save(commit=False)
            
            # Cập nhật vector nếu cần
            qdrant = QdrantService()
            if form.cleaned_data.get('generate_vector', False):
                vector = qdrant.generate_dummy_vector(collection.vector_size)
                point.vector_data = vector
            else:
                vector = point.vector_data or qdrant.generate_dummy_vector(collection.vector_size)
            
            # Cập nhật point trên Qdrant server
            success = qdrant.update_point(
                collection_name=collection.name,
                point_id=point.point_id,
                vector=vector,
                payload=point.payload
            )
            
            if success:
                point.qdrant_synced = True
                point.save()
                messages.success(request, 'Point đã được cập nhật!')
                return redirect('qdrant_manager:point_list', collection_pk=collection.pk)
            else:
                messages.error(request, 'Không thể cập nhật point trên Qdrant server!')
    else:
        # Convert tags list to string for form
        initial_data = {}
        if point.tags:
            initial_data['tags'] = ', '.join(point.tags)
        form = QdrantPointForm(instance=point, initial=initial_data)
    
    return render(request, 'qdrant_manager/point_form.html', {
        'form': form,
        'collection': collection,
        'point': point,
        'title': f'Chỉnh sửa Point "{point.point_id}"'
    })

@require_POST
def point_delete(request, collection_pk, pk):
    """Xóa point và tất cả points có cùng file_name"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    point = get_object_or_404(QdrantPoint, pk=pk, collection=collection)
    
    point_id = point.point_id
    file_name = point.file_name
    
    # Tìm tất cả points có cùng file_name trong collection này
    if file_name:
        related_points = QdrantPoint.objects.filter(
            collection=collection, 
            file_name=file_name
        ).exclude(pk=pk)
        related_point_ids = [p.point_id for p in related_points]
        all_point_ids = [point_id] + related_point_ids
        total_points = len(all_point_ids)
    else:
        # Nếu không có file_name, chỉ xóa point hiện tại
        all_point_ids = [point_id]
        related_points = QdrantPoint.objects.none()
        total_points = 1
    
    # Xóa từ Qdrant server
    qdrant = QdrantService()
    if total_points == 1:
        success = qdrant.delete_point(collection.name, point_id)
    else:
        success = qdrant.delete_points(collection.name, all_point_ids)
    
    if success:
        # Xóa point chính
        point.delete()
        # Xóa các points liên quan
        if related_points.exists():
            deleted_count = related_points.delete()[0]
        else:
            deleted_count = 0
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            if file_name and total_points > 1:
                message = f'Đã xóa {total_points} points có cùng file "{file_name}"!'
            else:
                message = f'Point "{point_id}" đã được xóa!'
            return JsonResponse({'success': True, 'message': message})
        
        if file_name and total_points > 1:
            messages.success(request, f'Đã xóa {total_points} points có cùng file "{file_name}"!')
        else:
            messages.success(request, f'Point "{point_id}" đã được xóa!')
    else:
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': False, 'message': f'Không thể xóa point từ Qdrant server!'})
        messages.error(request, f'Không thể xóa point từ Qdrant server!')
    
    return redirect('qdrant_manager:point_list', collection_pk=collection.pk)

@require_POST
def point_bulk_delete(request, collection_pk):
    """Xóa nhiều points cùng lúc"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    
    try:
        data = json.loads(request.body)
        point_ids = data.get('point_ids', [])
        
        if not point_ids:
            return JsonResponse({'success': False, 'message': 'Không có point nào được chọn!'})
        
        # Lấy danh sách qdrant_ids
        points = QdrantPoint.objects.filter(pk__in=point_ids, collection=collection)
        qdrant_point_ids = [p.point_id for p in points]
        
        # Xóa từ Qdrant server
        qdrant = QdrantService()
        success = qdrant.delete_points(collection.name, qdrant_point_ids)
        
        if success:
            deleted_count = points.delete()[0]
            return JsonResponse({
                'success': True, 
                'message': f'Đã xóa {deleted_count} points!'
            })
        else:
            return JsonResponse({'success': False, 'message': 'Không thể xóa points từ Qdrant server!'})
        
    except Exception as e:
        return JsonResponse({'success': False, 'message': str(e)})

def search_points(request, collection_pk):
    """Tìm kiếm points bằng vector similarity"""
    collection = get_object_or_404(QdrantCollection, pk=collection_pk)
    
    results = []
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            limit = form.cleaned_data['limit']
            
            # Tạo vector từ query (dummy vector cho demo)
            qdrant = QdrantService()
            query_vector = qdrant.generate_dummy_vector(collection.vector_size)
            
            # Tìm kiếm
            results = qdrant.search_points(collection.name, query_vector, limit)
    else:
        form = SearchForm()
    
    context = {
        'collection': collection,
        'form': form,
        'results': results
    }
    
    return render(request, 'qdrant_manager/search_points.html', context)

# ===========================
# API Upload từ điển
# ===========================

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_dict(request):
    temp_paths = []
    files = request.FILES.getlist('files')
    for file in files:
        ext = os.path.splitext(file.name)[1]
        temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        with open(temp_path, "wb") as f:
            for chunk in file.chunks():
                f.write(chunk)
        temp_paths.append(temp_path)

    result = update_dictionary_from_files(temp_paths)

    for path in temp_paths:
        os.remove(path)

    response_data = {"status": "dictionary updated"}
    response_data.update(result)
    return Response(response_data)


# ===========================
# Search API with Hybrid
# ===========================

@api_view(['POST'])
def search(request):
    start_time = time.time()
    query = request.data.get('query', '')
    filename = request.data.get('filename', None)
    logger.info(f"Raw question: {query}")
    cleaned_query = normalize_query(query)
    logger.info(f"Normalized question: {cleaned_query}")
    query_vector = embed_texts([f"query: {cleaned_query}"])[0]
    query_tokens = cleaned_query.lower().split()

    # ------------------------------
    # Nếu có filename → dùng luôn để lọc
    # ------------------------------
    qdrant_filter = None
    if filename:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.file_name",
                    match=MatchValue(value=filename)
                )
            ]
        )
        logger.info(f"Applying filename filter from request: {filename}")

    # ------------------------------
    # Search trong file được chọn
    # ------------------------------
    qdrant = get_qdrant_client()
    if qdrant is None:
        return Response({"error": "Qdrant service unavailable"}, status=503)
        
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=50,
        with_payload=True,
        with_vectors=True,
        query_filter=qdrant_filter
    )

    if not search_result:
        return Response({"results": []})

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

    # Hybrid score
    alpha = 0.4
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * vec_sim

    # Sort & MMR
    sorted_indices = np.argsort(hybrid_scores)[::-1]
    sorted_candidates = [candidates[i] for i in sorted_indices[:50]]
    sorted_vecs = candidate_vecs[sorted_indices[:50]]

    selected = mmr(
        np.array(query_vector, dtype=np.float32),
        sorted_vecs,
        sorted_candidates,
        top_k=7
    )

    # Merge kết quả cùng file
    merged_results = []
    prev_meta, buffer_text = None, []
    for text, meta in selected:
        logger.info(f"Meta: {meta} \n")
        logger.info(f"Text: {text} \n")
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
    return Response({"results": merged_results})

# ===========================
# Upload API
# ===========================

@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload(request):
    start_time = time.time()
    file = request.FILES.get('file')
    if not file:
        return Response({"error": "No file uploaded"}, status=400)

    file_ext = os.path.splitext(file.name)[1].lower()
    temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{file_ext}")
    with open(temp_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    try:
        qdrant = get_qdrant_client()
        if qdrant is None:
            return Response({"error": "Qdrant service unavailable"}, status=503)
            
        ensure_collection_exists()

        if file_ext == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_ext == ".docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif file_ext == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            os.remove(temp_path)
            return Response({"error": "Unsupported file type"}, status=400)

        docs = loader.load()

        # Token-based splitter
        text_splitter = TokenTextSplitter(
            chunk_size=300,  # Tokens
            chunk_overlap=45  # ~15%
        )
        chunks = text_splitter.split_documents(docs)

        # Add file_name to metadata
        for chunk in chunks:
            chunk.metadata["file_name"] = file.name
            chunk.metadata["upsert_timestamp"] = int(time.time())

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

        qdrant = get_qdrant_client()
        if qdrant:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        
        logger.info(f"Upload time: {time.time() - start_time:.2f}s")
        return Response({
            "status": "completed",
            "vectors": len(points),
            "chunk_size": 300,  # tokens
            "overlap": 45
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return Response({"error": str(e)}, status=500)
    finally:
        os.remove(temp_path)