from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.i18n import set_language
from django.views.generic import RedirectView
from qdrant_manager import views

urlpatterns = [
    # path("", RedirectView.as_view(url='/qdrant/', permanent=False), name='home'),
    path("", include("qdrant_manager.urls")),  # Qdrant manager
    path("admin/", admin.site.urls),
    path("i18n/setlang/", set_language, name="set_language"),
    path("accounts/", include("django.contrib.auth.urls")),  # login, logout, password views
    path("upload_dict/", views.upload_dict, name="upload_dict"),  # Upload dictionary API
    path("search/", views.search, name="search"),  # Search API
    path("upload/", views.upload, name="upload"),  # Upload API
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)