from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from typing import Optional

app = FastAPI(title="Face AI Service")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class RegisterFaceRequest(BaseModel):
    image: str


class FaceVerifyRequest(BaseModel):
    reference_image: str
    candidate_image: str


def decode_base64_image(base64_string: str) -> np.ndarray:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Impossible de décoder l'image.")

    return image


def extract_face(image: np.ndarray) -> Optional[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None

    # garder le plus grand visage
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)

    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + w + margin_x, image.shape[1])
    y2 = min(y + h + margin_y, image.shape[0])

    face = image[y1:y2, x1:x2]
    return face


def preprocess_face(face: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    normalized = cv2.equalizeHist(resized)
    return normalized


def compare_faces(reference_face: np.ndarray, candidate_face: np.ndarray) -> dict:
    ref_processed = preprocess_face(reference_face)
    cand_processed = preprocess_face(candidate_face)

    # différence pixel
    abs_diff = cv2.absdiff(ref_processed, cand_processed)
    pixel_diff = float(np.mean(abs_diff))

    # 👉 règle simple (plus fiable pour ton cas)
    match = pixel_diff <= 40

    similarity_score = max(0.0, 100.0 - pixel_diff)

    print("PIXEL DIFF =", pixel_diff)
    print("MATCH =", match)

    return {
        "match": bool(match),
        "hist_score": 0.0,
        "pixel_diff": pixel_diff,
        "similarity_score": float(similarity_score)
    }


@app.get("/")
def home():
    return {"message": "Face AI Service with OpenCV is running"}


@app.post("/check-face")
def check_face(request: RegisterFaceRequest):
    try:
        image = decode_base64_image(request.image)
        face = extract_face(image)

        if face is None:
            return {
                "success": False,
                "message": "Aucun visage détecté."
            }

        return {
            "success": True,
            "message": "Visage détecté avec succès."
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


@app.post("/verify-face")
def verify_face(request: FaceVerifyRequest):
    try:
        ref_image = decode_base64_image(request.reference_image)
        cand_image = decode_base64_image(request.candidate_image)

        ref_face = extract_face(ref_image)
        cand_face = extract_face(cand_image)

        if ref_face is None:
            return {
                "success": False,
                "match": False,
                "message": "Aucun visage détecté dans l'image de référence."
            }

        if cand_face is None:
            return {
                "success": False,
                "match": False,
                "message": "Aucun visage détecté dans l'image candidate."
            }

        result = compare_faces(ref_face, cand_face)

        return {
            "success": True,
            "match": result["match"],
            "hist_score": result["hist_score"],
            "pixel_diff": result["pixel_diff"],
            "similarity_score": result["similarity_score"],
            "message": "Comparaison effectuée avec succès."
        }

    except Exception as e:
        return {
            "success": False,
            "match": False,
            "message": str(e)
        }