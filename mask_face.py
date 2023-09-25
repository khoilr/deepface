import uuid
import cv2
import pandas as pd


# Calculate an extended bounding box around the detected face
def calculate_extended_bbox(x, y, w, h, frame_shape, extend_by=20):
    extended_x = max(0, x - extend_by)
    extended_y = max(0, y - extend_by)
    extended_w = min(frame_shape[1] - extended_x, w + 2 * extend_by)
    extended_h = min(frame_shape[0] - extended_y, h + 2 * extend_by)

    return extended_x, extended_y, extended_w, extended_h


def main():
    # Read data from the CSV file
    df = pd.read_csv("new_data.csv")

    for index, row in df.iterrows():
        # Read image in "Frame File Path" column
        frame = cv2.imread(row["Frame File Path"])

        # Read "Facial Area" column
        x = row["X"]
        y = row["Y"]
        w = row["Width"]
        h = row["Height"]

        extended_bbox = calculate_extended_bbox(x, y, w, h, frame.shape)

        # Save the cropped face to 'images/faces' directory with name as GUID
        cropped_face = frame[
            extended_bbox[1] : extended_bbox[1] + extended_bbox[3],
            extended_bbox[0] : extended_bbox[0] + extended_bbox[2],
        ]

        # Save the cropped face to 'images/faces' directory with name as GUID
        _uuid = uuid.uuid4()
        uuid_str = str(_uuid)
        cv2.imwrite(f"images/faces/{uuid_str}.jpg", cropped_face)


if __name__ == "__main__":
    main()
