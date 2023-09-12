import os
import gdown


def load_weights(model):
    exact_file = "/.deepface/weights/retinaface.h5"
    url = "https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5"

    if not os.path.exists("/.deepface"):
        os.mkdir("/.deepface")
        print("Directory ", "/.deepface created")

    if not os.path.exists("/.deepface/weights"):
        os.mkdir("/.deepface/weights")
        print("Directory ", "/.deepface/weights created")

    if os.path.isfile(exact_file) != True:
        print("retinaface.h5 will be downloaded from the url " + url)
        gdown.download(url, exact_file, quiet=False)

    # gdown should download the pretrained weights here. If it does not still exist, then throw an exception.
    if os.path.isfile(exact_file) != True:
        raise ValueError(
            f"Pre-trained weight could not be loaded! You might try to download the pre-trained weights from the url {url} and copy it to the {exact_file} manually."
        )

    model.load_weights(exact_file)

    return model
