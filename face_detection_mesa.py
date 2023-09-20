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
FRAME_PATH = "images/frames"
MAX_WORKERS = 16
MAX_CAP_OPEN_FAILURES = 10
MAX_READ_FRAME_FAILURES = 10
FRAME_FREQUENCY = 1
FACE_THRESHOLD = 5
PACK_SIZE = 16


# Init faces DataFrame
df_faces: pd.DataFrame = (
    pd.read_csv(CSV_FILE)
    if os.path.exists(CSV_FILE)
    else pd.DataFrame(columns=["Datetime", "Frame File Path", "Confidence", "X", "Y", "Width", "Height", "Face ID"])
)

# Configure logger
logger.add(LOG_FILE, rotation="500 MB")


def recognize_face(extended_face: str) -> int:
    # Extract distinct value of face ID
    distinct_face_id = df_faces["Face ID"].dropna().unique()

    # Iterate through distinct face ID
    for face_id in distinct_face_id:
        # Get image paths of this face ID
        paths = df_faces.loc[df_faces["Face ID"] == face_id]["Frame File Path"].values

        # threshold is the minimum of number of paths and FACE_THRESHOLD
        threshold = min([len(paths), FACE_THRESHOLD])

        # count_true is the number of paths that are verified as the same person;
        count_true = 0
        count_false = 0

        # Iterate through paths
        for path in paths:
            # Verify similarity between a pair of images using DeepFace library
            result = DeepFace.verify(
                extended_face,
                path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False,
            )

            # Increase count_true or count_false
            verified = result["verified"]
            if verified:
                count_true += 1
            else:
                count_false += 1

            # If count_true is equal to threshold, return face_id
            # Else if count_false is equal to threshold, break
            if count_true == threshold:
                return face_id
            elif count_false == threshold:
                break

    # no match faces
    return 0


# Process and save faces detected in a frame
def detect_faces(frame):
    # Use global variables
    global df_faces

    # Init is_change, when is_change is True, save df_faces to CSV file
    is_change = False

    # Detect faces in frame
    face_objs = DeepFace.extract_faces(
        frame,
        target_size=(512, 512),
        detector_backend="ssd",
        enforce_detection=False,
        align=True,
    )

    # Create a thread pool to process faces in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Iterate through faces
        for face in face_objs:
            # Skip if confidence is 0 or infinity
            if face["confidence"] == 0 or face["confidence"] == float("inf"):
                continue

            # Set is_change to True
            is_change = True

            # Save frame to file
            frame_path = save_image(frame=frame, dir=FRAME_PATH, logger=logger)

            # Get face coordinates
            x = face["facial_area"]["x"]
            y = face["facial_area"]["y"]
            w = face["facial_area"]["w"]
            h = face["facial_area"]["h"]

            # Create row to add to df_faces
            new_row = {
                "Datetime": pd.Timestamp.now(),
                "Frame File Path": frame_path,
                "Confidence": face["confidence"],
                "X": x,
                "Y": y,
                "Width": w,
                "Height": h,
            }

            # Regconize whose face is this
            recognize_future = executor.submit(recognize_face, frame_path)
            recognize_result = recognize_future.result()

            # Add Face ID to new_row
            new_row["Face ID"] = recognize_result

            # Append new_row to df_faces
            df_faces.loc[len(df_faces)] = new_row.values()

            # Log
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
    os.makedirs(FRAME_PATH, exist_ok=True)
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

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                logger.error(f"Read frame failures reached {MAX_READ_FRAME_FAILURES}. Restarting the camera...")

    else:
        logger.error(f"Capture open failures reached {MAX_CAP_OPEN_FAILURES}. Exiting the program...")


if __name__ == "__main__":
    main()
