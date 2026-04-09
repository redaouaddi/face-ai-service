from fastapi import FastAPI
from pydantic import BaseModel
from deepface import DeepFace
import base64
import os
import uuid
import cv2
import numpy as np

app = FastAPI(title="Face AI Service Optimized")

TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)

# Charger le modèle une seule fois
print("Loading face recognition model...")
DeepFace.build_model("Facenet")
print("Face model loaded successfully")


class RegisterFaceRequest(BaseModel):
    image: str


class FaceVerifyRequest(BaseModel):
    reference_image: str
    candidate_image: str


def save_base64_image(base64_string: str, prefix: str) -> str:
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]

    image_data = base64.b64decode(base64_string)
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(TEMP_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(image_data)

    return filepath


def save_face_crop(face_array: np.ndarray, prefix: str) -> str:
    filename = f"{prefix}_face_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(TEMP_DIR, filename)

    # Normalisation DeepFace
    if face_array.max() <= 1.0:
        face_array = (face_array * 255).astype("uint8")
    else:
        face_array = face_array.astype("uint8")

    # RGB vers BGR pour OpenCV
    face_bgr = cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filepath, face_bgr)

    return filepath


def extract_and_crop_face(image_path: str, prefix: str) -> str:
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="opencv",
        enforce_detection=True,
        align=True
    )

    if not faces or len(faces) == 0:
        raise ValueError("No face detected")

    # On prend le premier visage détecté
    face_array = faces[0]["face"]
    return save_face_crop(face_array, prefix)


@app.get("/")
def home():
    return {"message": "Face AI Service Optimized is running"}


@app.post("/check-face")
def check_face(request: RegisterFaceRequest):
    original_path = None
    cropped_path = None

    try:
        original_path = save_base64_image(request.image, "check")
        cropped_path = extract_and_crop_face(original_path, "check")

        return {
            "success": True,
            "message_code": "face_detected"
        }

    except Exception as e:
        print("CHECK FACE ERROR:", str(e))
        return {
            "success": False,
            "message_code": "no_face_detected"
        }

    finally:
        if original_path and os.path.exists(original_path):
            os.remove(original_path)
        if cropped_path and os.path.exists(cropped_path):
            os.remove(cropped_path)


@app.post("/verify-face")
def verify_face(request: FaceVerifyRequest):
    ref_original = None
    cand_original = None
    ref_cropped = None
    cand_cropped = None

    try:
        ref_original = save_base64_image(request.reference_image, "ref")
        cand_original = save_base64_image(request.candidate_image, "cand")

        # Extraction réelle du visage sans arrière-plan
        ref_cropped = extract_and_crop_face(ref_original, "ref")
        cand_cropped = extract_and_crop_face(cand_original, "cand")

        result = DeepFace.verify(
            img1_path=ref_cropped,
            img2_path=cand_cropped,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False
        )

        print("Face verification result:", result)

        return {
            "success": True,
            "match": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "model": result["model"],
            "detector_backend": "cropped_face",
            "message_code": "comparison_done"
        }

    except Exception as e:
        print("VERIFY FACE ERROR:", str(e))
        return {
            "success": False,
            "match": False,
            "message_code": "verification_failed"
        }

    finally:
        for path in [ref_original, cand_original, ref_cropped, cand_cropped]:
            if path and os.path.exists(path):
                os.remove(path)