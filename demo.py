import cv2
from deepface import DeepFace
import os

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
"""
'opencv', 
'ssd', 
'dlib', 
'mtcnn', 
'retinaface', 
'mediapipe',
'yolov8',
'yunet',
"""
# read all images in 'images/frames' directory
for file in os.listdir("images/faces"):
    if not file.lower().endswith(IMAGE_EXTENSIONS):
        continue

    img = cv2.imread(os.path.join("images/faces", file))

    face_objs = DeepFace.extract_faces(
        img_path=img,
        detector_backend="opencv",
        enforce_detection=False,
    )

    for face_obj in face_objs:
        if face_obj["confidence"] == 0:
            cv2.imwrite(os.path.join("not_faces", file), img)
        else:
            cv2.imwrite(os.path.join("faces", file), img)
            print(face_obj)
