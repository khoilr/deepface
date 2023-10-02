import concurrent.futures
import os
import uuid
from pprint import pprint
from typing import Union

import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np
import pandas as pd
from loguru import logger

from deepface import DeepFace


# Constants
LOG_FILE = "camera.log"
FACES_CSV_FILE = "faces.csv"
PERSONS_CSV_FILE = "persons.csv"
URL = "rtsp://0.tcp.ap.ngrok.io:15592/user:1cinnovation;pwd:1cinnovation123"
FRAME_PATH = "camera_web/images/frames"
MAX_WORKERS = 4
MAX_CAP_OPEN_FAILURES = 10
MAX_READ_FRAME_FAILURES = 10
FRAME_FREQUENCY = 5
FACE_THRESHOLD = 5


# Init DataFrame
df_faces: pd.DataFrame = (
    pd.read_csv(FACES_CSV_FILE)
    if os.path.exists(FACES_CSV_FILE)
    else pd.DataFrame(columns=["DateTime", "FrameFilePath", "Confidence", "X", "Y", "Width", "Height", "FaceID"])
)
df_persons: pd.DataFrame = (
    pd.read_csv(PERSONS_CSV_FILE) if os.path.exists(PERSONS_CSV_FILE) else pd.DataFrame(columns=["FaceID", "Name"])
)


# Configure logger
logger.add(LOG_FILE, rotation="500 MB")


def recognize_face(frame):
    # Extract distinct value of face ID
    distinct_face_id = df_faces["FaceID"].dropna().unique()

    # Shuffle distinct face ID
    np.random.shuffle(distinct_face_id)

    # Get max face ID or 0 if there is no face ID
    max_id = max(distinct_face_id) if len(distinct_face_id) > 0 else 0

    # Iterate through distinct face ID
    for face_id in distinct_face_id:
        # Get image paths of this face ID
        paths = df_faces.loc[df_faces["FaceID"] == face_id]["FrameFilePath"].values

        # threshold is the minimum of number of paths and FACE_THRESHOLD
        if len(paths) < FACE_THRESHOLD:
            threshold = len(paths)
            threshold_distance = 0.5
        else:
            threshold = FACE_THRESHOLD
            threshold_distance = None

        # count_true is the number of paths that are verified as the same person;
        count_true = 0
        count_false = 0

        # Iterate through paths
        for path in paths:
            # Verify similarity between a pair of images using DeepFace library
            result = DeepFace.verify(
                frame,
                path,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False,
            )

            # When threshold distance is not None, then verify by distance. Otherwise, verify by verified
            if threshold_distance is not None:
                # If distance is less than threshold_distance, count_true += 1
                # Else count_false += 1
                if result["distance"] < threshold_distance:
                    count_true += 1
                else:
                    count_false += 1
            else:
                # If verified is True, count_true += 1
                # Else count_false += 1
                if result["verified"]:
                    count_true += 1
                else:
                    count_false += 1

            # If count_true is equal to threshold, return face_id
            # Else if count_false is equal to threshold, break
            if count_true == threshold:
                logger.info(f"Face ID {face_id} is verified as the same person.")
                return face_id
            elif count_false == threshold:
                break

    # no match faces
    logger.info(f"Face ID {max_id + 1} is verified as a new person.")

    new_id = max_id + 1
    new_row = {"FaceID": new_id, "Name": new_id}
    df_persons.loc[len(df_persons)] = new_row
    df_persons.to_csv(PERSONS_CSV_FILE, index=False)

    return new_id


def get_focel_length(URL: str) -> float:
    cap = cv2.VideoCapture(URL)
    sucess, img = cap.read()
    if sucess:
        cv2.imshow("Image", img)
        cv2.waitKey(10)
    W = float(input("Please input the distance between two eyes (cm): "))
    d = float(input("Please input the distance to the camera (cm): "))
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        sucess, img = cap.read()
        if sucess:
            img, faces = detector.findFaceMesh(img=img, draw=False)
            if faces:
                face = faces[0]
                point_left = face[145]
                point_right = face[374]
                w, _ = detector.findDistance(point_right,point_left)
                focal_length = (w*d)/W     # calculate the focal length
                print(f"{focal_length=}")
                return focal_length, W
                


# Process and save faces detected in a frame
def detect_faces(frame):
    # Use global variables
    global df_faces

    # Extract faces from frame
    face_objs = DeepFace.extract_faces(
        frame,
        detector_backend="opencv",
        enforce_detection=False,
        align=True,
    )

    # Init is_change, when is_change is True, save df_faces to CSV file
    is_change = False

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        # Iterate through faces
        for face in face_objs:
            # Skip if confidence is 0 or infinity
            if face["confidence"] <= 0:
                continue

            # Set is_change to True
            is_change = True

            # Save frame to file
            frame_path = save_image(frame=frame, dir=FRAME_PATH, logger=logger)

            # Create row to add to df_faces
            new_face_row = {
                "Datetime": str(pd.Timestamp.now()),
                "FrameFilePath": frame_path,
                "Confidence": face["confidence"],
                "X": face["facial_area"]["x"],
                "Y": face["facial_area"]["y"],
                "Width": face["facial_area"]["w"],
                "Height": face["facial_area"]["h"]
            }

            # Submit the recognize_face function as a background task
            future = executor.submit(recognize_face, frame)
            futures.append((new_face_row, future))

        # Wait for all background tasks to complete and process their results
        for new_face_row, future in futures:
            face_id = future.result()
            new_face_row["FaceID"] = face_id
            df_faces.loc[len(df_faces)] = new_face_row

        if is_change:
            df_faces.to_csv(FACES_CSV_FILE, index=False)


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
    focal_length, true_eyes_width = get_focel_length(0)
    frame_counter = 0
    read_frame_failures_counter = 0
    cap_open_counter = 0

    # # Clear 'images/processing' directory if it already exists
    # if os.path.exists("images/processing"):
    #     for image in os.listdir("images/processing"):
    #         os.remove(f"images/processing/{image}")

    # Create directories for storing images
    os.makedirs(FRAME_PATH, exist_ok=True)
    # os.makedirs("images/processing", exist_ok=True)

    while cap_open_counter < MAX_CAP_OPEN_FAILURES:
        cap = cv2.VideoCapture(0) #URLURL
        detector = FaceMeshDetector()
        if not cap.isOpened():
            logger.error("Failed to connect to the camera.")
            cap_open_counter += 1
            continue

        logger.info("Connected to the camera.")
        cap_open_counter = 0
        read_frame_failures_counter = 0

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
                _, faces = detector.findFaceMesh(img=frame, draw=False)
                distance = 0
                if faces:
                    face = faces[0]
                    point_left = face[145]
                    point_right = face[374]
                    camera_eyes_width, _ = detector.findDistance(point_right,point_left)
                    distance = round((true_eyes_width*focal_length)/camera_eyes_width,2)
                    cvzone.putTextRect(frame, f"Distance: {distance}cm", (face[10][0]-75, face[10][1]-50), scale=2)
                # cv2.imwrite(f"frame.jpg", frame)
                # detect_faces(frame)  # Sequential
                cv2.imshow("Image", frame)
                cv2.waitKey(1)
                # executor.submit(detect_faces, frame)  # Parallel

            if cv2.waitKey(1) == ord("q"):
                break

        else:
            logger.error(f"Read frame failures reached {MAX_READ_FRAME_FAILURES}. Restarting the camera...")

    else:
        logger.error(f"Capture open failures reached {MAX_CAP_OPEN_FAILURES}. Exiting the program...")


if __name__ == "__main__":
    main()
