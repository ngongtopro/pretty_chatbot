from django.contrib import admin
from .models import QdrantCollection, QdrantPoint

@admin.register(QdrantCollection)
class QdrantCollectionAdmin(admin.ModelAdmin):
    list_display = ['name', 'vector_size', 'distance_metric', 'points_count', 'qdrant_synced', 'created_at']
    list_filter = ['distance_metric', 'qdrant_synced', 'created_at']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'points_count', 'vectors_count']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description')
        }),
        ('Vector Configuration', {
            'fields': ('vector_size', 'distance_metric')
        }),
        ('Sync Status', {
            'fields': ('qdrant_synced', 'points_count', 'vectors_count')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )

@admin.register(QdrantPoint)
class QdrantPointAdmin(admin.ModelAdmin):
    list_display = ['point_id', 'collection', 'title', 'category', 'qdrant_synced', 'created_at']
    list_filter = ['collection', 'category', 'qdrant_synced', 'created_at']
    search_fields = ['point_id', 'title', 'content', 'file_name']
    readonly_fields = ['created_at', 'updated_at']
    raw_id_fields = ['collection']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('collection', 'point_id', 'title')
        }),
        ('Content', {
            'fields': ('content',)
        }),
        ('Metadata', {
            'fields': ('file_name', 'source', 'category', 'tags')
        }),
        ('Vector Data', {
            'fields': ('vector_data', 'metadata'),
            'classes': ('collapse',)
        }),
        ('Sync Status', {
            'fields': ('qdrant_synced',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('collection')
