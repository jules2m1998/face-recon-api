import os.path
from fastapi import FastAPI, File
import face_recognition
from file import create_file, face_distance_to_conf
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


class ComparisonResponse(BaseModel):
    # is_same: bool
    percentage: float


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/", response_model=ComparisonResponse)
def read_root(cni: bytes = File(...), img: bytes = File(...)):
    print("**** Request ****")
    img1_location = create_file(cni, str(uuid.uuid1()))
    face1 = face_recognition.load_image_file(img1_location)
    f1_face_encoding = face_recognition.face_encodings(face1)[0]

    img2_location = create_file(img, str(uuid.uuid1()))
    face2 = face_recognition.load_image_file(img2_location)
    f2_face_encoding = face_recognition.face_encodings(face2)[0]

    # result_bool: bool = face_recognition.compare_faces([f1_face_encoding], f2_face_encoding)[0]
    face_distance = face_recognition.face_distance([f1_face_encoding], f2_face_encoding)[0]

    if os.path.exists(img1_location):
        os.remove(img1_location)

    if os.path.exists(img2_location):
        os.remove(img2_location)

    result = {
        # "is_same": bool(result_bool),
        "percentage": face_distance_to_conf(face_distance)*100
    }

    return result
