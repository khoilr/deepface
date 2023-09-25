from django.apps import AppConfig
from django.core import management


class ImageliveConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "imagelive"

    # def ready(self):
    #     management.call_command("camera_recognize_background")
