import concurrent.futures
import os
import time
import traceback
from datetime import datetime
from pprint import pprint
import cv2
from custom_deepface import face_detection
from deepface import DeepFace

url = "rtsp://admin:admin123122@95bachdang112021.dyndns.org/cam/realmonitor"


def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to retrieve a frame from the stream.")
    return frame


def save_frame(frame, name):
    cv2.imwrite(name, frame)


def process_and_save_faces(frame):
    face_objs = face_detection.extract(
        image=frame,
        target_size=(112, 112),
    )

    # if there is any face detected with confidence != 0 then save the frame
    is_save_frame = False

    for face in face_objs:
        if face["confidence"] != 0:
            is_save_frame = True
            name = f"images/faces/{int(datetime.now().timestamp())}.jpg"
            cv2.imwrite(name, face["face"])
            print(f"Saved face to {name}")

    if is_save_frame:
        save_frame(frame, f"images/frames/{int(datetime.now().timestamp())}.jpg")


def run_camera(channel):
    params = {
        "channel": channel,
        "subtype": 1,
    }
    param_string = "&".join([f"{key}={value}" for key, value in params.items()])
    full_url = f"{url}?{param_string}"
    cap = cv2.VideoCapture(full_url)
    if not cap.isOpened():
        return

    while True:
        try:
            frame = capture_frame(cap)
            save_frame(frame, 'frame.jpg')
            process_and_save_faces(frame)

            if cv2.waitKey(1) == ord("q"):
                break

        except Exception as e:
            print(f"Error occurred in channel {channel}: {e}")
            return  # Exit the function and allow it to restart


def main():
    while True:
        with concurrent.futures.ThreadPoolExecutor() as executor_channel:
            try:
                executor_channel.map(run_camera, range(1, 5))
            except Exception as e:
                print(f"Error in the ThreadPoolExecutor: {e}")
                print("Restarting")
                time.sleep(1)  # Delay before restarting


if __name__ == "__main__":
    main()
