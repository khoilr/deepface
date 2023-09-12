import concurrent.futures
import os
from datetime import datetime
from typing import Union

import cv2
import pandas as pd
from loguru import logger
from deepface import DeepFace

# Constants
LOG_FILE = "camera.log"
CSV_FILE = "data.csv"
URL = "rtsp://0.tcp.ap.ngrok.io:18852/user:1cinnovation;pwd:1cinnovation123"
MAX_WORKERS = 16
MAX_CAP_OPEN_FAILURES = 10
MAX_READ_FRAME_FAILURES = 10
FRAME_FREQUENCY = 20


# Initialize the DataFrame
def initialize_dataframe():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["Frame File Name", "Face File Name", "Facial Area", "Confidence"])
    else:
        return pd.read_csv(CSV_FILE)


df_faces: pd.DataFrame = initialize_dataframe()


# Configure the Loguru logger
def configure_logger():
    logger.add(LOG_FILE, rotation="500 MB")


configure_logger()


# Calculate an extended bounding box around the detected face
def calculate_extended_bbox(bbox, frame_shape, extend_by=20):
    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]

    extended_x = max(0, x - extend_by)
    extended_y = max(0, y - extend_by)
    extended_w = min(frame_shape[1] - extended_x, w + 2 * extend_by)
    extended_h = min(frame_shape[0] - extended_y, h + 2 * extend_by)

    return extended_x, extended_y, extended_w, extended_h


# Process and save faces detected in a frame
def detect_faces(frame):
    global df_faces

    processing_image = save_image(frame=frame, dir="images/processing")

    is_change = False

    face_objs = DeepFace.extract_faces(
        frame,
        target_size=(112, 112),
        detector_backend="retinaface",
        enforce_detection=False,
        align=True,
    )

    for face in face_objs:
        if face["confidence"] == 0:
            continue

        is_change = True

        extended_bbox = calculate_extended_bbox(face["facial_area"], frame.shape)

        cropped_face = frame[
            extended_bbox[1] : extended_bbox[1] + extended_bbox[3],
            extended_bbox[0] : extended_bbox[0] + extended_bbox[2],
        ]

        cropped_face = cv2.resize(cropped_face, (112, 112))

        frame_path = save_image(frame=frame, dir="images/frames", logger=logger)
        face_path = save_image(frame=cropped_face, dir="images/faces", logger=logger)

        new_row = {
            "Frame File Name": frame_path,
            "Face File Name": face_path,
            "Facial Area": extended_bbox,
            "Confidence": face["confidence"],
        }
        df_faces.loc[len(df_faces)] = new_row.values()

        logger.info(
            f"Face detected. Frame File Name: {frame_path}, Face File Name: {face_path}, Facial Area: {extended_bbox}, Confidence: {face['confidence']}"
        )

    if is_change:
        df_faces.to_csv(CSV_FILE, index=False)

    # Delete the processing image
    os.remove(processing_image)


# Create directories if they don't exist
def create_directories(dir: str, logger=None) -> None:
    exist = False

    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        exist = True

    if logger and not exist:
        logger.info(f"Created '{dir}' directory.")
    elif logger:
        logger.info(f"'{dir}' directory already exists.")


# Save an image to a specified directory
def save_image(frame, dir: str = ".", name: str = None, logger=None) -> Union[str, None]:
    if name is None:
        timestamp = int(datetime.now().timestamp())
        image_path = f"{dir}/{timestamp}.jpg"
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
    for directory in ["images/frames", "images/faces", "images/processing"]:
        create_directories(directory, logger)

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
                    executor.submit(detect_faces, frame)

                if cv2.waitKey(1) == ord("q"):
                    break

            else:
                logger.error(f"Read frame failures reached {MAX_READ_FRAME_FAILURES}. Restarting the camera...")

    else:
        logger.error(f"Capture open failures reached {MAX_CAP_OPEN_FAILURES}. Exiting the program...")


if __name__ == "__main__":
    main()
