import json
import os
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from deepface import DeepFace
from deepface.DeepFace import build_model
from deepface.detectors import FaceDetector

# Constants
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_WORKERS = 2
IMAGES_PATH = "images/faces"
SAVE_IMAGE_PAIR_PATH = "images/distances"

# Singletons
face_detector = FaceDetector.build_model("opencv")
model = build_model("ArcFace")


def verify_image_pair(pair):
    """
    Verify similarity between a pair of images using DeepFace library.

    Parameters:
        pair (tuple): A tuple containing two image filenames.

    Returns:
        tuple: A tuple containing the two image filenames and the verification result.
    """
    image_1, image_2 = pair

    result = DeepFace.verify(
        img1_path=os.path.join(IMAGES_PATH, image_1),
        img2_path=os.path.join(IMAGES_PATH, image_2),
        align=True,
        enforce_detection=False,
        detector_backend="opencv",
        distance_metric="euclidean_l2",
        model_name="ArcFace",
    )

    return image_1, image_2, result


def create_image_pairs(image_files):
    """
    Generate all possible pairs of images.

    Parameters:
        image_files (list): List of image filenames.

    Returns:
        list: A list of tuples representing image pairs.
    """
    num_files = len(image_files)
    pairs = [(image_files[i], image_files[j]) for i in range(num_files) for j in range(i + 1, num_files)]
    return pairs


def verify_image_pairs(image_pairs):
    """
    Verify image pairs using ThreadPoolExecutor.

    Parameters:
        image_pairs (list): List of image pairs.

    Returns:
        list: A list of dictionaries containing verification results.
    """
    results = []
    with tqdm(total=len(image_pairs)) as pbar, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(verify_image_pair, pair) for pair in image_pairs]

        for future in as_completed(futures):
            try:
                result = future.result()
                image_1, image_2, result_data = result
                # result_data["verified"] = True if result_data["distance"] < 0.4 else False
                result_data["verified"] = bool(result_data["verified"])  # Convert numpy.bool_ to bool

                results.append({"image_1": image_1, "image_2": image_2, "result": result_data})

            except Exception as e:
                traceback.print_exc()
            finally:
                pbar.update(1)

    return results


def save_results_as_json(results):
    """
    Save verification results as JSON.

    Parameters:
        results (list): A list of dictionaries containing verification results.
    """
    try:
        with open("table.json", "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(e)


def create_similarity_table(image_files, results):
    """
    Create a similarity table based on image pairs.

    Parameters:
        image_files (list): List of image filenames.
        results (list): A list of dictionaries containing verification results.

    Returns:
        DataFrame: A DataFrame storing similarity values.
    """

    # Delete 'images/distances' directory if it exists
    dirpath = Path(SAVE_IMAGE_PAIR_PATH)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)

    # Make directory 'images/distances' if it doesn't exist
    os.makedirs(SAVE_IMAGE_PAIR_PATH, exist_ok=True)

    # Create a DataFrame to store similarity values
    df = pd.DataFrame(columns=image_files, index=image_files)

    for result in results:
        image_1 = result["image_1"]
        image_2 = result["image_2"]
        result_data = result["result"]

        distance = round(result_data["distance"], 2)
        df.loc[image_1, image_2] = distance
        df.loc[image_2, image_1] = distance

        if result_data["verified"]:
            draw_and_save_images(image_1, image_2, result_data, distance)

    # Save the DataFrame as a CSV file
    df.to_csv("table.csv")
    return df


def draw_and_save_images(image_1_name, image_2_name, result_data, distance):
    """
    Draw bounding boxes, merge images, and save the result.

    Parameters:
        image_1 (str): Filename of the first image.
        image_2 (str): Filename of the second image.
        distance (float): Similarity distance.
    """
    # Round the distance to 2 decimal place
    distance = round(result_data["distance"], 2)

    # Create a directory for the distance if it doesn't exist
    os.makedirs(f"{SAVE_IMAGE_PAIR_PATH}/f{distance}", exist_ok=True)

    # # Get bbox
    # image_1_bbox = result_data["facial_areas"]["img1"]
    # image_2_bbox = result_data["facial_areas"]["img2"]

    # Read images
    image_1 = cv2.imread(os.path.join(IMAGES_PATH, image_1_name))
    image_2 = cv2.imread(os.path.join(IMAGES_PATH, image_2_name))

    # Get images' shape
    height_1, width_1, _ = image_1.shape
    height_2, width_2, _ = image_2.shape

    # Resize images to the same height
    if height_1 > height_2:
        image_1 = cv2.resize(image_1, (int(width_1 * height_2 / height_1), height_2))
    else:
        image_2 = cv2.resize(image_2, (int(width_2 * height_1 / height_2), height_1))

    # # Draw bounding boxes
    # cv2.rectangle(
    #     image_1,
    #     (image_1_bbox["x"], image_1_bbox["y"]),
    #     (image_1_bbox["x"] + image_1_bbox["w"], image_1_bbox["y"] + image_1_bbox["h"]),
    #     (255, 0, 0),
    #     2,
    # )
    # cv2.rectangle(
    #     image_2,
    #     (image_2_bbox["x"], image_2_bbox["y"]),
    #     (image_2_bbox["x"] + image_2_bbox["w"], image_2_bbox["y"] + image_2_bbox["h"]),
    #     (255, 0, 0),
    #     2,
    # )

    # Merge 2 images
    image = cv2.hconcat([image_1, image_2])

    # Write distance value on top center
    cv2.putText(image, str(distance), (int(image.shape[1] / 2), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Save image
    cv2.imwrite(os.path.join(f"{SAVE_IMAGE_PAIR_PATH}/f{distance}", f"{image_1_name}_{image_2_name}.jpg"), image)


def main():
    """
    Main function to create a similarity table and heatmap.
    """
    # Clear console
    print("\033[H\033[J")

    # Generate images list
    image_files = [file for file in os.listdir(IMAGES_PATH) if file.lower().endswith(IMAGE_EXTENSIONS)]

    num_files = len(image_files)
    assert num_files > 1, "Number of files must be greater than 1"

    # Generate all possible pairs of images
    image_pairs = create_image_pairs(image_files)

    # Verify image pairs using ThreadPoolExecutor
    results = verify_image_pairs(image_pairs)

    # Sar results as JSON
    save_results_as_json(results)

    # Create a similarity table and save it as a CSV file
    similarity_table = create_similarity_table(image_files, results)
    print("Table created and saved.")


if __name__ == "__main__":
    main()
