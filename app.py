import os, sys, pickle, numpy as np
ROOT = "/content/drive/MyDrive/exovision_data"
sys.path.insert(0, ROOT)
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import lumina_pipeline as lp

pipe = lp.LuminaPipeline()   # <- must match how your notebook created pipe (see output below)
FEATS = ROOT + "/features_kepler_v2/"
KEEP = ["period","epoch","depth","duration","flux_std","flux_mean","flux_median","flux_min","flux_max","model_snr","local_view","global_view"]

def js(o):
    if isinstance(o, dict): return {str(k): js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [js(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.generic): return o.item()
    return o

app = FastAPI()

@app.get("/predict")
def predict(target: str):
    path = f"{FEATS}{target}.pkl"
    if not os.path.exists(path):
        raise HTTPException(404, "target not in this deployment's sample set")
    out = pipe.run_from_pickle(path)
    r = pickle.load(open(path, "rb"))
    return js({"pipe_output": out, "record": {k: r[k] for k in KEEP if k in r}})

class S2(BaseModel):
    spectrum: dict
    target: str

@app.post("/stage2")
def stage2(req: S2):
    path = f"{FEATS}{req.target}.pkl"
    if not os.path.exists(path):
        raise HTTPException(404, "target not found")
    sp = {k: np.asarray(v, dtype=np.float32) for k, v in req.spectrum.items()}
    try:
        out = pipe.run_from_pickle(path, spectrum=sp)
    except Exception as e:
        raise HTTPException(400, f"Stage 2 failed: {e}")
    if out["stage2"] is None:
        raise HTTPException(409, out.get("note", "Stage 2 not run"))
    return js(out["stage2"])

app.mount("/", StaticFiles(directory="static", html=True), name="web")
