import concurrent.futures

import cv2
import pandas as pd
from deepface.basemodels import ArcFace
from deepface import DeepFace


arc_face = ArcFace.loadModel()


def draw(x, y, h, w, most_common_person_name, frame):
    # Draw the boundary around the detected face
    boundary_color = (0, 255, 0)  # Green color (BGR format)
    cv2.rectangle(frame, (x, y), (x + w, y + h), boundary_color, 2)

    # Draw a filled rectangle as the background for the person's name
    text_bg_color = boundary_color
    text_size = cv2.getTextSize(most_common_person_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    cv2.rectangle(frame, (x, y - text_size[1] - 16), (x + text_size[0], y), text_bg_color, -1)

    # Draw the person's name on the top-left of the boundary
    text_color = (255, 255, 255)  # White color (BGR format)
    cv2.putText(frame, most_common_person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)


def process_face(face, frame, index):
    facial_area = face["facial_area"]
    y = facial_area["y"]
    x = facial_area["x"]
    w = facial_area["w"]
    h = facial_area["h"]
    crop_img = frame[y : y + h, x : x + w]

    temp_name_2 = f"temp_{index}.jpg"
    cv2.imwrite(temp_name_2, crop_img)

    result = DeepFace.find(
        img_path=temp_name_2,
        db_path="database/cropped",
        model_name="ArcFace",
        distance_metric="euclidean_l2",
        detector_backend="retinaface",
        enforce_detection=False,
        # silent=True,
    )

    df = pd.concat(result)

    try:
        # sort df by ArcFace_euclidean_l2 asc
        df = df.sort_values(by=["ArcFace_euclidean_l2"]).head(10)
        df["person_name"] = df["identity"].str.split("/").str[2]
        # find first five same person name
        
        person_name_counts = df["person_name"].value_counts()
        names_at_least_5_times = person_name_counts[person_name_counts >= 5]
        most_common_person_name = names_at_least_5_times.idxmax()
    except:
        most_common_person_name = "Unknown"

    draw(x, y, h, w, most_common_person_name, frame)

    # Displaying the result
    print(f"Result for face {index}: The person name that appears the most is: {most_common_person_name}")


# Start the video capture
cap = cv2.VideoCapture('rtsp://192.168.68.182:554/user:1cinnovation;pwd:1cinnovation123')
temp_name = "temp.jpeg"
num_workers = 4  # You can adjust this based on your system's capabilities

while True:
    ret, frame = cap.read()

    # if not ret:
    #     continue

    # cv2.imwrite(temp_name, frame)
    # face_objs = DeepFace.extract_faces(
    #     temp_name,
    #     detector_backend="retinaface",
    #     enforce_detection=False,
    # )

    # # Process faces in parallel using ThreadPoolExecutor
    # with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(process_face, face, frame, index) for index, face in enumerate(face_objs)]

    #     # Wait for all tasks to complete
    #     concurrent.futures.wait(futures)

    cv2.imshow("Most Common Name on Face", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
