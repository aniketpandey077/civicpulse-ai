from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import google.generativeai as genai
import tempfile
import os
import base64
import json
import re

app = FastAPI(title="CivicPulse AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import numpy as np

# Lazy-load YOLO model
model = None

def get_model():
    global model
    if model is None:
        model = YOLO("models/pothole.pt")
    return model

def _warmup_yolo():
    try:
        m = get_model()
        # Run dummy 416x416 matrix through model to warm up PyTorch CPU kernels
        dummy = np.zeros((416, 416, 3), dtype=np.uint8)
        m(dummy, verbose=False)
    except Exception:
        pass

@app.on_event("startup")
def startup_event():
    # Full PyTorch CPU warmup in background thread on startup
    threading.Thread(target=_warmup_yolo, daemon=True).start()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Issue types handled by YOLO vs Gemini
YOLO_ISSUES = {"pothole", "potholes", "road_damage", "road_damages"}

GEMINI_PROMPTS = {
    "garbage": "Is there uncollected garbage, waste, or litter visible in this image? Assess severity 0-100 (0=clean, 100=severe overflow). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
    "manhole": "Is there an open, damaged, or missing manhole cover visible in this image? Assess severity 0-100 (0=safe, 100=fully open/dangerous). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
    "streetlight": "Is there a broken, non-functional, or damaged street light visible in this image? Assess severity 0-100 (0=working fine, 100=completely broken/missing). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
    "water_leakage": "Is there visible water leakage, flooding, burst pipe, or waterlogging in this image? Assess severity 0-100 (0=none, 100=severe flooding). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
}


@app.get("/")
def root():
    return {"status": "CivicPulse AI running"}


@app.get("/issue-types")
def issue_types():
    return {
        "types": [
            "pothole",
            "road_damage",
            "garbage",
            "manhole",
            "streetlight",
            "water_leakage",
        ]
    }


def severity_from_box(confidence: float, box_area: float, image_area: float) -> int:
    area_ratio = box_area / image_area if image_area else 0
    score = (confidence * 60) + (min(area_ratio * 400, 40))
    return int(min(max(score, 0), 100))


def analyze_with_yolo(image_path: str, issue_type: str):
    try:
        # Convert image to RGB & resize to 416x416 max for 3x faster CPU detection
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((416, 416))
        img.save(image_path, format="JPEG")

        results = get_model()(image_path, verbose=False)
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

        max_severity = max((d["severity"] for d in detections), default=0)
        return {
            "detected": len(detections) > 0,
            "issue_type": issue_type,
            "count": len(detections),
            "severity": max_severity,
            "detections": detections,
            "description": f"{len(detections)} {issue_type.replace('_', ' ')}(s) detected." if detections else "No issue detected.",
        }
    except Exception as e:
        return {
            "detected": False,
            "issue_type": issue_type,
            "count": 0,
            "severity": 0,
            "detections": [],
            "description": f"Analysis error: {str(e)}",
        }


def analyze_with_gemini(image_path: str, issue_type: str):
    if not GEMINI_API_KEY:
        res = analyze_with_yolo(image_path, issue_type)
        res["description"] = f"[YOLO Fallback: GEMINI_API_KEY not set] {res['description']}"
        return res

    try:
        model_names = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-1.5-flash"]
        prompt = GEMINI_PROMPTS.get(issue_type.lower().strip(), GEMINI_PROMPTS["auto"])
        img = Image.open(image_path)

        response = None
        last_err = None
        for m_name in model_names:
            try:
                g_model = genai.GenerativeModel(m_name)
                response = g_model.generate_content([prompt, img])
                if response and response.text:
                    break
            except Exception as err:
                last_err = err
                continue

        if not response or not response.text:
            raise last_err or Exception("Gemini generation failed")

        text = response.text.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = {"detected": False, "severity": 0, "description": text or "Could not parse response."}

        return {
            "detected": data.get("detected", False),
            "issue_type": data.get("issue_type", issue_type),
            "count": 1 if data.get("detected") else 0,
            "severity": data.get("severity", 0),
            "detections": [],
            "description": data.get("description", ""),
        }
    except Exception as e:
        # Fallback to YOLO model if Gemini encounters any API error
        res = analyze_with_yolo(image_path, issue_type)
        res["description"] = f"[YOLO Fallback: Gemini API error: {str(e)}] {res['description']}"
        return res


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    issue_type: str = Query(default="pothole", description="Select issue type: pothole, road_damage, garbage, manhole, streetlight, water_leakage")
):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        temp.write(contents)
        image_path = temp.name

    clean_issue_type = issue_type.lower().strip()

    try:
        if clean_issue_type in YOLO_ISSUES:
            return analyze_with_yolo(image_path, clean_issue_type)
        else:
            return analyze_with_gemini(image_path, clean_issue_type)
    finally:
        os.remove(image_path)
