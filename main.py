from typing import Union

from fastapi import FastAPI, File
import face_recognition
from file import create_file, face_distance_to_conf

app = FastAPI()


@app.post("/")
def read_root(cni: bytes = File(...), img: bytes = File(...)):
    img1_location = create_file(cni, "tester")
    face1 = face_recognition.load_image_file(img1_location)
    f1_face_encoding = face_recognition.face_encodings(face1)[0]

    img2_location = create_file(img, "tester2")
    face2 = face_recognition.load_image_file(img2_location)
    f2_face_encoding = face_recognition.face_encodings(face2)[0]

    result: bool = face_recognition.compare_faces([f1_face_encoding], f2_face_encoding)[0]
    face_distance = face_recognition.face_distance([f1_face_encoding], f2_face_encoding)[0]

    result = {
        "is_same": True if result else False,
        "percentage": face_distance_to_conf(face_distance)
    }
    return result
