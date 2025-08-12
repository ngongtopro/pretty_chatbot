from django.urls import path
from . import views

app_name = "chat"

urlpatterns = [
    path("", views.index, name="index"),
    path("c/<int:pk>/", views.index, name="conversation"),
    path("new/", views.new_conversation, name="new_conversation"),
    path("send/<int:pk>/", views.send_message, name="send_message"),
    path("accounts/register/", views.register, name="register"),
]