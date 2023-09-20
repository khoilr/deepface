import concurrent.futures
import os

from typing import Union
import uuid
import cv2
import pandas as pd
from loguru import logger
from deepface import DeepFace
from mask_face import cropFace
from clustering import verify_image_pairs
from retinaface import RetinaFace

# Constants
LOG_FILE = "camera.log"
CSV_FILE = "new_data.csv"
URL = "rtsp://0.tcp.ap.ngrok.io:18505/user:1cinnovation;pwd:1cinnovation123"
MAX_WORKERS = 16
MAX_CAP_OPEN_FAILURES = 10
MAX_READ_FRAME_FAILURES = 10
FRAME_FREQUENCY = 1
FACE_THRESHOLD = 5
PACK_SIZE = 16


# Initialize the DataFrame
def initialize_dataframe():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["Frame File Path", "Confidence", "X", "Y", "Width", "Height", "Face ID"])
    else:
        return pd.read_csv(CSV_FILE)


df_faces: pd.DataFrame = initialize_dataframe()


# Configure the Loguru logger
def configure_logger():
    logger.add(LOG_FILE, rotation="500 MB")


configure_logger()


def recognize_face(extended_face: str) -> str:
    # extract distinct value of face ID
    distinct_face_id = df_faces["Face ID"].dropna().unique()

    print(list(distinct_face_id))
    for face_id in list(distinct_face_id):
        threshhold = FACE_THRESHOLD
        paths = df_faces.loc[df_faces['Face ID'] == face_id]['Frame File Path'].values
        if (len(paths) < threshhold):
            threshhold = len(paths)
        count_true = 0
        count_false = 0
        for path in paths:
            result = DeepFace.verify(extended_face,
                                     path,
                                     model_name='ArcFace',
                                     detector_backend='retinaface',
                                     enforce_detection=False,
                                     )['verified']
            if result:
                count_true += 1
            else:
                count_false += 1

            if count_true == threshhold:
                return str(face_id)
            if count_false == threshhold:
                break

    # no match faces
    return "0"


# Process and save faces detected in a frame
def detect_faces(frame):
    global df_faces

    is_change = False

    face_objs = DeepFace.extract_faces(
        frame,
        target_size=(512, 512),
        detector_backend="ssd",
        enforce_detection=False,
        align=True,
    )


    with  concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for face in face_objs:
            if face["confidence"] == 0:
                continue

            is_change = True
            print("Dected Face")
            frame_path = save_image(frame=frame, dir="images/frames", logger=logger)

            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]

            new_row = {
                "Frame File Path": frame_path,
                "Confidence": face["confidence"],
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h,

            }

            regconizeFuture = executor.submit(recognize_face, frame_path)
            regcoResult = regconizeFuture.result()
            new_row['Face ID'] = int(float(regcoResult))
            print(new_row)
            df_faces.loc[len(df_faces)] = new_row.values()

            logger.info(f"Face detected with confidence {face['confidence']}. Saved to {frame_path}.")

        if is_change:
            df_faces.to_csv(CSV_FILE, index=False)



# Save an image to a specified directory
def save_image(frame, dir: str = ".", name: str = None, logger=None) -> Union[str, None]:
    if name is None:
        _uuid = uuid.uuid4()
        uuid_str = str(_uuid)
        image_path = f"{dir}/{uuid_str}.jpg"
    else:
        image_path = f"{dir}/{name}.jpg"

    try:
        cv2.imwrite(image_path, frame)
        if logger:
            logger.info(f"Saved image to {image_path}")
        return image_path
    except Exception as e:
        if logger:
            logger.error(f"Error saving image: {str(e)}")
        return None


# Main function to capture frames, process faces, and save results
def main():
    frame_counter = 0
    read_frame_failures_counter = 0
    cap_open_counter = 0

    # Clear 'images/processing' directory if it already exists
    if os.path.exists("images/processing"):
        for image in os.listdir("images/processing"):
            os.remove(f"images/processing/{image}")

    # Create directories for storing images
    os.makedirs("images/frames", exist_ok=True)
    os.makedirs("images/processing", exist_ok=True)


    while cap_open_counter < MAX_CAP_OPEN_FAILURES:
        cap = cv2.VideoCapture(URL)

        if not cap.isOpened():
            logger.error("Failed to connect to the camera.")
            cap_open_counter += 1
            continue
        else:
            logger.info("Connected to the camera.")
            cap_open_counter = 0
            read_frame_failures_counter = 0
            faces = []

            while read_frame_failures_counter < MAX_READ_FRAME_FAILURES:
                ret, frame = cap.read()

                if not ret:
                    logger.error("Failed to capture frame.")
                    read_frame_failures_counter += 1
                    continue
                else:
                    read_frame_failures_counter = 0
                frame_counter += 1

                if frame_counter % FRAME_FREQUENCY == 0:
                    cv2.imwrite(f"frame.jpg", frame)
                    detect_faces(frame)  # Sequential
                    # faces.extend(detectedFaces)
                    # # integrate clusterings
                    # if (len(faces) >= 10):
                    #     with concurrent.futures.ThreadPoolExecutor() as executor:
                    #         verify_faces = [(faces[i], faces[i + 1]) for i in range(10, 2)]
                    #         faces.clear()
                    #         pairResultsFuture = executor.submit(verify_image_pairs, verify_faces)
                    #         print("Pair results:", pairResultsFuture.result())
                    #     # reset faces list
                    #     faces.clear()

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                logger.error(f"Read frame failures reached {MAX_READ_FRAME_FAILURES}. Restarting the camera...")

    else:
        logger.error(f"Capture open failures reached {MAX_CAP_OPEN_FAILURES}. Exiting the program...")


if __name__ == "__main__":
    main()
