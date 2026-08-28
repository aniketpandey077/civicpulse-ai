import io
import json
import os
import re
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import google.generativeai as genai
from ultralytics import YOLO

app = FastAPI(title="CivicPulse AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# 1. Global Model Initialization (Loaded ONCE at server boot)
# ---------------------------------------------------------
MODEL_PATH = "models/pothole.pt"
yolo_model = YOLO(MODEL_PATH)

# Configure Gemini API if key exists
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

YOLO_ISSUES = {"pothole", "potholes", "road_damage", "road_damages"}

GEMINI_PROMPTS = {
    "pothole": "Is there a pothole or road surface damage visible in this image? Assess severity 0-100 (0=smooth road, 100=severe deep pothole). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
    "road_damage": "Is there road damage, cracking, or pothole visible in this image? Assess severity 0-100 (0=good condition, 100=severe destruction). Reply ONLY with valid JSON: {\"detected\": true/false, \"severity\": 0-100, \"description\": \"one sentence\"}",
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


# ---------------------------------------------------------
# 2. In-Memory YOLO Analysis (No Disk I/O, 416x416 Input)
# ---------------------------------------------------------
def _sync_analyze_yolo(img: Image.Image, issue_type: str):
    try:
        img_resized = img.copy()
        img_resized.thumbnail((416, 416))

        results = yolo_model(img_resized, conf=0.15, verbose=False)
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


# ---------------------------------------------------------
# 3. Gemini Vision Analysis
# ---------------------------------------------------------
def _sync_analyze_gemini(img: Image.Image, issue_type: str):
    if not GEMINI_API_KEY:
        return _sync_analyze_yolo(img, issue_type)

    try:
        model_names = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-1.5-flash"]
        prompt = GEMINI_PROMPTS.get(issue_type.lower().strip(), GEMINI_PROMPTS["pothole"])

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
    except Exception:
        return _sync_analyze_yolo(img, issue_type)


# ---------------------------------------------------------
# 4. Non-Blocking Async Endpoint
# ---------------------------------------------------------
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    issue_type: str = Query(default="pothole", description="Select issue type: pothole, road_damage, garbage, manhole, streetlight, water_leakage")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    clean_issue_type = issue_type.lower().strip()

    if clean_issue_type in YOLO_ISSUES:
        return await run_in_threadpool(_sync_analyze_yolo, image, clean_issue_type)
    else:
        return await run_in_threadpool(_sync_analyze_gemini, image, clean_issue_type)
