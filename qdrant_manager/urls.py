from django.urls import path
from . import views

app_name = 'qdrant_manager'

from django.urls import path
from . import views

app_name = 'qdrant_manager'

urlpatterns = [
    # Collection URLs
    path('', views.collection_list, name='collection_list'),
    path('collections/create/', views.collection_create, name='collection_create'),
    path('collections/<int:pk>/edit/', views.collection_edit, name='collection_edit'),
    path('collections/<int:pk>/delete/', views.collection_delete, name='collection_delete'),
    
    # Point URLs
    path('collections/<int:collection_pk>/points/', views.point_list, name='point_list'),
    path('collections/<int:collection_pk>/points/create/', views.point_create, name='point_create'),
    path('collections/<int:collection_pk>/points/<int:pk>/', views.point_detail, name='point_detail'),
    path('collections/<int:collection_pk>/points/<int:pk>/edit/', views.point_edit, name='point_edit'),
    path('collections/<int:collection_pk>/points/<int:pk>/delete/', views.point_delete, name='point_delete'),
    path('collections/<int:collection_pk>/points/bulk-delete/', views.point_bulk_delete, name='point_bulk_delete'),
    
    # Search URLs
    path('collections/<int:collection_pk>/search/', views.search_points, name='search_points'),

    # Upload URLs
    # path('')
]
