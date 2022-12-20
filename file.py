import math


def create_file(file: bytes, filename: str) -> str:
    name: str = f"assets/{filename}"
    with open(name, "wb") as img:
        img.write(file)
        img.close()
    return name


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        ran = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (ran * 2.0)
        return linear_val
    else:
        ran = face_match_threshold
        linear_val = 1.0 - (face_distance / (ran * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))
