from typing import Any, Union
import cv2
import numpy as np
from deepface.detectors import FaceDetector
import tensorflow as tf

def _extract(
    image: Union[str, np.ndarray],
    target_size: tuple,
    detector_backend: str,
    grayscale: bool,
    enforce_detection: bool,
    align: bool,
):
    extracted_faces = []
    if isinstance(image, str):
        image = cv2.imread(image)
    img_region = [0, 0, image.shape[1], image.shape[0]]
    face_detector = FaceDetector.build_model(detector_backend)
    face_objs = FaceDetector.detect_faces(face_detector, detector_backend, image, align)
    if not face_objs:
        if enforce_detection:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo or consider setting enforce_detection param to False."
            )
        else:
            face_objs = [(image, img_region, 0)]
    for current_img, current_region, confidence in face_objs:
        resized_image = cv2.resize(current_img, target_size)
        if grayscale:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            resized_image = np.expand_dims(resized_image, axis=-1)
        region_obj = {
            "x": int(current_region[0]),
            "y": int(current_region[1]),
            "w": int(current_region[2]),
            "h": int(current_region[3]),
        }
        extracted_face = [resized_image, region_obj, confidence]
        extracted_faces.append(extracted_face)
    if not extracted_faces and enforce_detection:
        raise ValueError(f"Detected face shape is {image.shape}. Consider setting enforce_detection arg to False.")
    return extracted_faces


def extract(
    image: Union[str, np.ndarray],
    target_size: tuple = (112, 112),
    face_detector: str = "retinaface",
    enforce_detection: bool = False,
    align: bool = True,
    grayscale: bool = False,
) -> list[dict[str, Union[np.ndarray, dict[str, int], float]]]:
    """
    Extract faces from an image using a pre-processing pipeline.
    Parameters:
        image (Union[str, np.ndarray]): Path to the image or image array (BGR format).
        target_size (tuple): Final shape of facial images (height, width).
        detector_backend (str): Face detection backend (e.g., "opencv", "retinaface").
        enforce_detection (bool): Raise error if no face is detected.
        align (bool): Perform alignment based on eye positions.
        grayscale (bool): Extract faces in grayscale.
    Returns:
        List of dictionaries, each containing the resized face image, facial area, and confidence.
    """
    resp_objs = []
    img_objs = _extract(
        image=image,
        target_size=target_size,
        detector_backend=face_detector,
        grayscale=grayscale,
        enforce_detection=enforce_detection,
        align=align,
    )
    for img, region, confidence in img_objs:
        resp_obj = {
            "face": img[0] if len(img.shape) == 4 else img,
            "facial_area": region,
            "confidence": confidence,
        }
        resp_objs.append(resp_obj)
    return resp_objs


if __name__ == "__main__":
    face_detector = FaceDetector.build_model()
