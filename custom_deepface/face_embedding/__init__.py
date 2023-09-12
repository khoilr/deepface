import numpy as np
from typing import Any, Union
from deepface.DeepFace import build_model
from custom_deepface.face_detection import extract
from deepface.commons import functions


def represent(
    image: Union[str, np.ndarray],
    model_name: Any = "ArcFace",
    enforce_detection: bool = False,
    detector_backend: Any = "retinaface",
    align: bool = True,
):
    """
    This function represents facial images as vectors. The function uses convolutional neural
    networks models to generate vector embeddings.

    Parameters:
            img_path (string): exact image path. Alternatively, numpy array (BGR) or based64
            encoded images could be passed. Source image can have many faces. Then, result will
            be the size of number of faces appearing in the source image.

            model_name (string): VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib,
            ArcFace, SFace

            enforce_detection (boolean): If no face could not be detected in an image, then this
            function will return exception by default. Set this to False not to have this exception.
            This might be convenient for low resolution images.

            detector_backend (string): set face detector backend to opencv, retinaface, mtcnn, ssd,
            dlib or mediapipe

            align (boolean): alignment according to the eye positions.

            normalization (string): normalize the input image before feeding to model

    Returns:
            Represent function returns a list of object with multidimensional vector (embedding).
            The number of dimensions is changing based on the reference model.
            E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
    """
    resp_objs = []

    model = build_model(model_name)

    # ---------------------------------
    target_size = functions.find_target_size(model_name=model_name)
    img_objs = extract(
        image=image,
        target_size=target_size,
        face_detector=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )

    for image, region, _ in img_objs:
        # custom normalization
        image = functions.normalize_input(image)

        # represent
        print(str(type(model)))
        if "keras" in str(type(model)):
            # new tf versions show progress bar and it is annoying
            embedding = model.predict(image, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(image)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_objs.append(resp_obj)

    return resp_objs
