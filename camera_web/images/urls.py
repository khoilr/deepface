from django.urls import path

from .views import *

urlpatterns = [
    path("<str:slug>", getImage, name="get_slug_image"),
]
