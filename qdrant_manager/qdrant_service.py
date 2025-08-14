import os
import requests
from requests.auth import HTTPBasicAuth
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, 
    VectorParams, 
    CreateCollection, 
    PointStruct,
    SearchRequest,
    Filter,
    FieldCondition,
    MatchValue
)

class QdrantService:
    """Service class để tương tác với Qdrant server dùng HTTP Basic Auth"""
    
    def __init__(self):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = os.getenv("QDRANT_PORT", 6330)
        self.base_url = f"http://{self.host}:{self.port}"
        self.username = "admin"
        self.password = "0fgD0hegTpYETmZCbze4OZy"
        self.auth = HTTPBasicAuth(self.username, self.password)
    
    def _make_request(self, method: str, endpoint: str, json_data: Dict = None) -> Optional[Dict]:
        """Thực hiện HTTP request với Basic Auth"""
        url = f"{self.base_url}/{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, auth=self.auth)
            elif method.upper() == "POST":
                response = requests.post(url, auth=self.auth, json=json_data)
            elif method.upper() == "PUT":
                response = requests.put(url, auth=self.auth, json=json_data)
            elif method.upper() == "DELETE":
                response = requests.delete(url, auth=self.auth)
            else:
                return None
            
            if response.status_code in [200, 201, 204]:
                return response.json() if response.content else {"status": "ok"}
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Lấy danh sách tất cả collections với thông tin chi tiết"""
        response = self._make_request("GET", "collections")
        if response and "result" in response:
            collections = response["result"]["collections"]
            detailed_collections = []
            
            for collection in collections:
                collection_name = collection["name"]
                # Lấy thông tin chi tiết từng collection
                detail_info = self.get_collection_info(collection_name)
                if detail_info:
                    detailed_collections.append({
                        'name': detail_info['name'],
                        'vectors_count': detail_info['vectors_count'],
                        'points_count': detail_info['points_count'],
                        'status': detail_info['status'],
                        'optimizer_status': detail_info['optimizer_status'],
                        'indexed_vectors_count': detail_info['indexed_vectors_count'],
                        'vector_size': detail_info['config']['vector_size'],
                        'distance': detail_info['config']['distance']
                    })
                else:
                    # Fallback với giá trị mặc định
                    detailed_collections.append({
                        'name': collection_name,
                        'vectors_count': 0,
                        'points_count': 0,
                        'status': 'green',
                        'optimizer_status': 'ok',
                        'indexed_vectors_count': 0,
                        'vector_size': 1536,  # Default
                        'distance': 'Cosine'  # Default
                    })
            
            return detailed_collections
        return []
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một collection"""
        response = self._make_request("GET", f"collections/{collection_name}")
        if response and "result" in response:
            info = response["result"]
            # Lấy vector config an toàn
            config = info.get("config", {})
            params = config.get("params", {})
            vectors = params.get("vectors", {})
            
            vector_size = vectors.get("size", 1536)  # Default 1536 nếu null
            distance = vectors.get("distance", "Cosine")  # Default Cosine nếu null
            
            return {
                'name': collection_name,
                'vectors_count': info.get("vectors_count", 0),
                'points_count': info.get("points_count", 0),
                'status': info.get("status", "green"),
                'optimizer_status': info.get("optimizer_status", "ok"),
                'indexed_vectors_count': info.get("indexed_vectors_count", 0),
                'config': {
                    'vector_size': vector_size,
                    'distance': distance
                }
            }
        return None
    
    def create_collection(self, name: str, vector_size: int = 1536, distance: str = "Cosine") -> bool:
        """Tạo collection mới"""
        data = {
            "vectors": {
                "size": vector_size,
                "distance": distance
            }
        }
        response = self._make_request("PUT", f"collections/{name}", data)
        return response is not None
    
    def delete_collection(self, name: str) -> bool:
        """Xóa collection"""
        response = self._make_request("DELETE", f"collections/{name}")
        return response is not None
    
    def get_points(self, collection_name: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Lấy danh sách points từ collection"""
        data = {
            "limit": limit,
            "offset": offset,
            "with_payload": True,
            "with_vector": False
        }
        response = self._make_request("POST", f"collections/{collection_name}/points/scroll", data)
        if response and "result" in response:
            points = response["result"]["points"]
            return [
                {
                    'id': point["id"],
                    'payload': point.get("payload", {}),
                    'vector': point.get("vector")
                }
                for point in points
            ]
        return []
    
    def get_point(self, collection_name: str, point_id: str) -> Dict[str, Any]:
        """Lấy chi tiết một point cụ thể"""
        data = {
            "ids": [point_id],
            "with_payload": True,
            "with_vector": True
        }
        response = self._make_request("POST", f"collections/{collection_name}/points", data)
        if response and "result" in response and response["result"]:
            point = response["result"][0]
            return {
                'id': point["id"],
                'payload': point.get("payload", {}),
                'vector': point.get("vector", [])
            }
        return None
    
    def add_point(self, collection_name: str, point_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """Thêm point mới vào collection"""
        data = {
            "points": [
                {
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                }
            ]
        }
        response = self._make_request("PUT", f"collections/{collection_name}/points", data)
        return response is not None
    
    def update_point(self, collection_name: str, point_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """Cập nhật point"""
        return self.add_point(collection_name, point_id, vector, payload)  # Upsert operation
    
    def delete_point(self, collection_name: str, point_id: str) -> bool:
        """Xóa point"""
        data = {
            "points": [point_id]
        }
        response = self._make_request("POST", f"collections/{collection_name}/points/delete", data)
        return response is not None
    
    def delete_points(self, collection_name: str, point_ids: List[str]) -> bool:
        """Xóa nhiều points cùng lúc"""
        data = {
            "points": point_ids
        }
        response = self._make_request("POST", f"collections/{collection_name}/points/delete", data)
        return response is not None
    
    def search_points(self, collection_name: str, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Tìm kiếm points tương tự"""
        data = {
            "vector": vector,
            "limit": limit,
            "with_payload": True
        }
        response = self._make_request("POST", f"collections/{collection_name}/points/search", data)
        if response and "result" in response:
            results = response["result"]
            return [
                {
                    'id': result["id"],
                    'score': result["score"],
                    'payload': result.get("payload", {})
                }
                for result in results
            ]
        return []
    
    def generate_dummy_vector(self, size: int = 1536) -> List[float]:
        """Tạo vector giả để test"""
        import random
        return [random.random() for _ in range(size)]
    
    @staticmethod
    def test_basic_auth_connection():
        """Test kết nối Qdrant dùng HTTP Basic Auth, trả về response collections"""
        url = "http://10.10.9.24:6330/collections"
        username = "admin"
        password = "0fgD0hegTpYETmZCbze4OZy"
        try:
            response = requests.get(url, auth=HTTPBasicAuth(username, password))
            print(f"Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Error connecting with basic auth: {e}")
            return None
