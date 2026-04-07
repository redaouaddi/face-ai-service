from fastapi import FastAPI
from pydantic import BaseModel
from deepface import DeepFace
import base64
import os
import uuid

app = FastAPI(title="Face AI Service Optimized")

TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)

# 🔥 Charger le modèle une seule fois (très important pour la vitesse)
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


@app.get("/")
def home():
    return {"message": "Face AI Service Optimized is running"}


@app.post("/check-face")
def check_face(request: RegisterFaceRequest):
    path = None
    try:
        path = save_base64_image(request.image, "check")

        objs = DeepFace.extract_faces(
            img_path=path,
            detector_backend="opencv",
            enforce_detection=True
        )

        if not objs:
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

    finally:
        if path and os.path.exists(path):
            os.remove(path)


@app.post("/verify-face")
def verify_face(request: FaceVerifyRequest):
    ref_path = None
    cand_path = None

    try:
        ref_path = save_base64_image(request.reference_image, "ref")
        cand_path = save_base64_image(request.candidate_image, "cand")

        result = DeepFace.verify(
            img1_path=ref_path,
            img2_path=cand_path,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=True
        )

        print("Face verification result:", result)

        return {
            "success": True,
            "match": bool(result["verified"]),
            "distance": float(result["distance"]),
            "threshold": float(result["threshold"]),
            "model": result["model"],
            "detector_backend": result["detector_backend"],
            "message": "Comparaison effectuée avec succès."
        }

    except Exception as e:
        return {
            "success": False,
            "match": False,
            "message": str(e)
        }

    finally:
        if ref_path and os.path.exists(ref_path):
            os.remove(ref_path)
        if cand_path and os.path.exists(cand_path):
            os.remove(cand_path)