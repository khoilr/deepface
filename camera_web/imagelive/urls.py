from django.urls import path
from .views import test, mask_feed, test_video, mask_feed_video

urlpatterns = [
    path("", test, name="get_image"),
    path("camera/", test_video, name="get_video"),
    path("mask_feed", mask_feed, name="mask_feed"),
    path("mask_test_video", mask_feed_video, name="mask_feed_video"),

]
