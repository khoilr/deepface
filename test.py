import numpy as np
import pandas as pd
import os
from deepface import  DeepFace
from concurrent.futures import ThreadPoolExecutor
CSV_FILE = "new_data.csv"

FACE_THRESHOLD = 5
def initialize_dataframe():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame(columns=["Frame File Path", "Confidence", "X", "Y", "Width", "Height", "Face ID"])
    else:
        return pd.read_csv(CSV_FILE)

df_faces =initialize_dataframe()
def recognize_face(extended_face: str) -> str:

    # extract distinct value of face ID
    distinct_face_id = df_faces["Face ID"].dropna().unique()

    print(list(distinct_face_id))
    for face_id in list(distinct_face_id):
        threshhold=FACE_THRESHOLD
        paths=df_faces.loc[df_faces['Face ID'] == face_id]['Frame File Path'].values
        if(len(paths)<threshhold):
            threshhold=len(paths)
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


res=recognize_face("images/frames/f718a5d7-7f8f-4922-81b9-573f1c8f301b.jpg")
print(res)