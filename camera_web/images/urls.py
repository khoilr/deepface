from django.urls import path

from .views import *

urlpatterns = [
    path('',test_camera,name='get_camera_test'),
    path('<str:slug>', getImage, name="get_slug_image"),
]
