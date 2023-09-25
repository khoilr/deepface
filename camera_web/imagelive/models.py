import threading
import cv2
import time
from django.db.models import (
    Model,
    TextField,
    DateTimeField,
    FloatField,
    CASCADE,
    ForeignKey,
)


class Person(Model):
    name = TextField()


class Face(Model):
    person = ForeignKey(Person, on_delete=CASCADE)
    confidence = FloatField()
    x = FloatField()
    y = FloatField()
    width = FloatField()
    height = FloatField()
    image_path = TextField()
    datetime = DateTimeField()


class VideoCamera(object):
    def __init__(self, URL):
        self.video = cv2.VideoCapture(URL)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        image = self.frame
        _, jpeg = cv2.imencode(" .jpg", image)
        return jpeg.tobytes()

    def update(self):
        while True:
            time.sleep(0.05)
            (self.grabbed, self.frame) = self.video.read()
