import concurrent.futures
import os

from typing import Union
import uuid
import cv2
import pandas as pd
from loguru import logger
from deepface import DeepFace

# Constants
LOG_FILE = "camera.log"
CSV_FILE = "new_data.csv"
URL = "rtsp://0.tcp.ap.ngrok.io:13067/user:1cinnovation;pwd:1cinnovation123"
MAX_WORKERS = 16
MAX_CAP_OPEN_FAILURES = 10
MAX_READ_FRAME_FAILURES = 10
FRAME_FREQUENCY = 1


# Initialize the DataFrame
def initialize_dataframe():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["Frame File Path", "Confidence", "X", "Y", "Width", "Height"])
    else:
        return pd.read_csv(CSV_FILE)


df_faces: pd.DataFrame = initialize_dataframe()


# Configure the Loguru logger
def configure_logger():
    logger.add(LOG_FILE, rotation="500 MB")


configure_logger()


# Process and save faces detected in a frame
def detect_faces(frame):
    global df_faces

    is_change = False

    face_objs = DeepFace.extract_faces(
        frame,
        target_size=(512, 512),
        detector_backend="dlib",
        enforce_detection=False,
        align=True,
    )

    for face in face_objs:
        if face["confidence"] == 0:
            continue

        is_change = True

        frame_path = save_image(frame=frame, dir="images/frames", logger=logger)

        new_row = {
            "Frame File Path": frame_path,
            "Confidence": face["confidence"],
            "X": face["facial_area"]["x"],
            "Y": face["facial_area"]["y"],
            "Width": face["facial_area"]["w"],
            "Height": face["facial_area"]["h"],
        }
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
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
                    # executor.submit(detect_faces, frame) # Concurrent
                    cv2.imwrite(f"frame.jpg", frame)
                    detect_faces(frame) # Sequential

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                logger.error(f"Read frame failures reached {MAX_READ_FRAME_FAILURES}. Restarting the camera...")

    else:
        logger.error(f"Capture open failures reached {MAX_CAP_OPEN_FAILURES}. Exiting the program...")


if __name__ == "__main__":
    main()
