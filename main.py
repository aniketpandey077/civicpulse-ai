from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import tempfile
import os

app = FastAPI(title="CivicPulse AI")

# Allow your frontend to call this API from any origin.
# Tighten this to your actual frontend URL before final submission if you want.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("models/pothole.pt")


@app.get("/")
def root():
    return {"status": "CivicPulse AI running"}


def severity_from_box(confidence: float, box_area: float, image_area: float) -> int:
    """
    Heuristic severity score (0-100) combining detection confidence
    with how much of the image the pothole occupies.
    Replace with your own logic if you want something fancier.
    """
    area_ratio = box_area / image_area if image_area else 0
    score = (confidence * 60) + (min(area_ratio * 400, 40))
    return int(min(max(score, 0), 100))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp.write(contents)
        image_path = temp.name

    try:
        results = model(image_path)

        detections = []
        for result in results:
            img_h, img_w = result.orig_shape
            image_area = img_h * img_w

            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                box_area = (x2 - x1) * (y2 - y1)

                detections.append({
                    "confidence": round(confidence, 3),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "severity": severity_from_box(confidence, box_area, image_area),
                })

        return {
            "detected": len(detections) > 0,
            "count": len(detections),
            "detections": detections,
        }
    finally:
        os.remove(image_path)
