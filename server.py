import base64
import io
import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.hf_model import (
    DR_CLASSES,
    ENSEMBLE_REPOS,
    HF_MODEL_ID,
    attention_heatmap,
    count_parameters,
    load_hf_model,
    overlay_heatmap,
    predict_dr,
)

BASE = Path(__file__).parent
STATIC = BASE / "static"
CKPT_DIR = BASE / "checkpoints"

app = FastAPI(title="RetinaXcelerate")

# ---- Paper pipeline (default): EnsembleDR = rafalosa + Kontawat ViT-B/16, CCER ----
print("Loading ViT ensemble from HuggingFace…")
model, processor, device = load_hf_model()
print(f"Model ready on {device}.")

# ---- Optional Enhanced mode (paper Future Work §VII-1), loaded on first use ----
_enhanced = None


def enhanced_available() -> bool:
    return all((CKPT_DIR / f).exists()
               for f in ["dr_vit.pt", "dr_effnet.pt", "ccer_weights.json"])


def get_enhanced():
    global _enhanced
    if _enhanced is None:
        from src.enhanced import EnhancedCCER
        print("Loading Enhanced CCER members…")
        _enhanced = EnhancedCCER(model, CKPT_DIR)
    return _enhanced


def _json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _ranking(ranking):
    return [{"severity": r["severity"], "code": r["code"], "name": r["name"],
             "description": r["description"], "prob": round(float(r["prob"]), 6)}
            for r in ranking]


def _run(image: Image.Image, heatmap: bool, mode: str) -> dict:
    if mode == "enhanced":
        if not enhanced_available():
            raise HTTPException(400, "Enhanced mode needs checkpoints/dr_vit.pt, dr_effnet.pt "
                                     "and ccer_weights.json (see README).")
        r = get_enhanced().predict(image)
        out = {"prediction": r["pred"], "confidence": float(r["probs"][r["pred_idx"]]),
               "ranking": _ranking(r["ranking"]), "per_model": r["per_model"], "mode": "enhanced",
               "input_image": _img_to_data_url(image.resize((512, 512)))}
        if heatmap:
            out["heatmap"] = _img_to_data_url(
                overlay_heatmap(r["heat_base"], r["heatmap"]).resize((512, 512)))
        return out
    # paper pipeline (Algorithm 1)
    result = predict_dr(model, processor, image, device)
    out = {"prediction": result["pred"],
           "confidence": float(result["probs"][result["pred_idx"]]),
           "ranking": _ranking(result["ranking"]), "mode": "paper",
           "input_image": _img_to_data_url(image.resize((512, 512)))}
    if heatmap:
        heat = attention_heatmap(result["attentions"], tuple(result["input_size"]))
        out["heatmap"] = _img_to_data_url(overlay_heatmap(image, heat).resize((512, 512)))
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


# Curated evaluation gallery (paper Table III) — Wikimedia Commons, CC0 / public domain.
SAMPLES = [
    {"id": "normal",       "label": "Normal retina",
     "expected": "No DR",
     "caption": "Healthy fundus — no signs of retinopathy.",
     "url": "/static/samples/normal.jpg"},
    {"id": "dr_general",   "label": "Mild DR",
     "expected": "Mild NPDR",
     "caption": "Early diabetic retinopathy — scattered microaneurysms.",
     "url": "/static/samples/dr_general.png"},
    {"id": "dr_eda03",     "label": "Moderate DR",
     "expected": "Moderate NPDR",
     "caption": "Diabetic retinopathy with hemorrhages and exudates.",
     "url": "/static/samples/dr_eda03.jpg"},
    {"id": "cotton_wool",  "label": "Severe DR",
     "expected": "Severe NPDR",
     "caption": "Retina with cotton-wool spots and hemorrhages.",
     "url": "/static/samples/cotton_wool.png"},
    {"id": "proliferative", "label": "Proliferative DR",
     "expected": "Proliferative DR",
     "caption": "Advanced proliferative DR with neovascularization.",
     "url": "/static/samples/proliferative.jpg"},
    {"id": "laser_dr",     "label": "Treated DR",
     "expected": "Severe / post-laser",
     "caption": "Fundus after scatter laser photocoagulation.",
     "url": "/static/samples/laser_dr.jpg"},
    {"id": "cmv",          "label": "CMV retinitis",
     "expected": "non-DR pathology",
     "caption": "CMV retinitis — out-of-distribution pathology.",
     "url": "/static/samples/cmv.jpg"},
]


@app.get("/api/info")
def info():
    p = count_parameters(model)
    return {
        "model_id": HF_MODEL_ID,
        "ensemble_repos": ENSEMBLE_REPOS,
        "device": device,
        "params_millions": round(p["total"] / 1e6, 1),
        "classes": DR_CLASSES,
        "samples": SAMPLES,
        "enhanced": {
            "available": enhanced_available(),
            "weights": _json(CKPT_DIR / "ccer_weights.json"),
            "evaluation": _json(BASE / "results" / "ccer_eval.json"),
        },
    }


@app.get("/api/sample/{sid}")
async def sample(sid: str, mode: str = "paper"):
    s = next((x for x in SAMPLES if x["id"] == sid), None)
    if s is None:
        raise HTTPException(404, "unknown sample")
    image = Image.open(BASE / s["url"].lstrip("/")).convert("RGB")
    return {**_run(image, True, mode), "sample": s}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), heatmap: bool = True, mode: str = "paper"):
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    return JSONResponse(_run(image, heatmap, mode))


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
