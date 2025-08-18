from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
import json

# Giữ models cũ để tránh lỗi migration
class QdrantDatabase(models.Model):
    """Model để quản lý Qdrant databases (deprecated)"""
    name = models.CharField(max_length=200, unique=True, help_text="Tên database")
    description = models.TextField(blank=True, help_text="Mô tả về database")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'qdrant_database'
        verbose_name = "Qdrant Database (Deprecated)"
        verbose_name_plural = "Qdrant Databases (Deprecated)"

    def __str__(self):
        return self.name

class QdrantDocument(models.Model):
    """Model để quản lý documents trong Qdrant (deprecated)"""
    database = models.ForeignKey(QdrantDatabase, on_delete=models.CASCADE, related_name='documents')
    qdrant_id = models.CharField(max_length=200, help_text="ID trong Qdrant")
    content = models.TextField(help_text="Nội dung document")
    file_name = models.CharField(max_length=255, blank=True, help_text="Tên file gốc")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'qdrant_document'
        verbose_name = "Qdrant Document (Deprecated)"
        verbose_name_plural = "Qdrant Documents (Deprecated)"
        unique_together = ['database', 'qdrant_id']

    def __str__(self):
        return f"{self.database.name} - {self.qdrant_id}"

# Models mới cho Qdrant Collection và Point
class QdrantCollection(models.Model):
    """Model để quản lý Qdrant collections"""
    
    DISTANCE_CHOICES = [
        ('cosine', 'Cosine'),
        ('euclidean', 'Euclidean'),
        ('dot', 'Dot Product'),
    ]
    
    name = models.CharField(
        max_length=200, 
        unique=True, 
        help_text="Tên collection trên Qdrant server"
    )
    description = models.TextField(
        blank=True, 
        help_text="Mô tả về collection"
    )
    vector_size = models.PositiveIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(10000)],
        help_text="Kích thước vector (số chiều)"
    )
    distance_metric = models.CharField(
        max_length=20,
        choices=DISTANCE_CHOICES,
        default='cosine',
        help_text="Độ đo khoảng cách sử dụng"
    )
    qdrant_synced = models.BooleanField(
        default=False,
        help_text="Đã đồng bộ với Qdrant server"
    )
    points_count = models.PositiveIntegerField(
        default=0,
        help_text="Số lượng points trong collection"
    )
    vectors_count = models.PositiveIntegerField(
        default=0,
        help_text="Số lượng vectors trong collection"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'qdrant_collection'
        verbose_name = "Qdrant Collection"
        verbose_name_plural = "Qdrant Collections"
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.vector_size}D)"

    @property
    def sync_status(self):
        """Trả về trạng thái đồng bộ"""
        return "Synced" if self.qdrant_synced else "Not Synced"

class QdrantPoint(models.Model):
    """Model để quản lý points trong Qdrant collection"""
    
    collection = models.ForeignKey(
        QdrantCollection, 
        on_delete=models.CASCADE, 
        related_name='points',
        help_text="Collection chứa point này"
    )
    point_id = models.CharField(
        max_length=200, 
        help_text="ID của point trong Qdrant (UUID, số, hoặc string)"
    )
    content = models.TextField(
        help_text="Nội dung chính của point (sẽ được vector hóa)"
    )
    title = models.CharField(
        max_length=500, 
        blank=True, 
        help_text="Tiêu đề hoặc tên ngắn gọn"
    )
    source = models.URLField(
        blank=True, 
        help_text="URL nguồn của nội dung"
    )
    file_name = models.CharField(
        max_length=255, 
        blank=True, 
        help_text="Tên file nguồn"
    )
    category = models.CharField(
        max_length=100, 
        blank=True, 
        help_text="Danh mục phân loại"
    )
    tags = models.JSONField(
        default=list, 
        blank=True, 
        help_text="Danh sách tags"
    )
    vector_data = models.JSONField(
        default=list, 
        blank=True, 
        help_text="Dữ liệu vector (array of floats)"
    )
    metadata = models.JSONField(
        default=dict, 
        blank=True, 
        help_text="Metadata bổ sung dưới dạng JSON"
    )
    qdrant_synced = models.BooleanField(
        default=False,
        help_text="Đã đồng bộ với Qdrant server"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'qdrant_point'
        verbose_name = "Qdrant Point"
        verbose_name_plural = "Qdrant Points"
        unique_together = ['collection', 'point_id']
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.collection.name} - {self.point_id}"

    @property
    def payload(self):
        """Trả về payload để gửi lên Qdrant server"""
        payload = {
            'content': self.content,
            'title': self.title,
            'source': self.source,
            'file_name': self.file_name,
            'category': self.category,
            'tags': self.tags,
        }
        
        # Thêm metadata bổ sung
        if self.metadata:
            payload.update(self.metadata)
        
        # Loại bỏ các field rỗng
        return {k: v for k, v in payload.items() if v}

    @property
    def has_vector(self):
        """Kiểm tra xem point có vector data không"""
        return bool(self.vector_data)

    @property
    def vector_dimension(self):
        """Trả về số chiều của vector"""
        return len(self.vector_data) if self.vector_data else 0

    def get_tags_display(self):
        """Hiển thị tags dưới dạng chuỗi"""
        return ', '.join(self.tags) if self.tags else ''
