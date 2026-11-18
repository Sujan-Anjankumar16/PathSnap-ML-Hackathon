"""PathSnap FastAPI backend (runs on Laptop 1).

Start (from the project root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables
    PATHSNAP_EVAL_DATA   folder with the evaluator trips (one sub-folder per trip id); default
                         "<Student folder>/evaluator_test"
    PATHSNAP_PARTITION   which partition the API serves: evaluator_test (default) or student_test (demo)

Endpoints
    GET  /health                -> status, model info
    GET  /trips                 -> list of trip ids available for evaluation
    POST /predict {"trip_id"}   -> predicted route + accuracy computed server-side

Ground truth (.route) is read ONLY inside `evaluate_server_side`, after the prediction has been made
from .track/.nodes/.arcs; it is never passed to the model and never returned to the UI.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config
from pathsnap.data import list_trips, load_trip
from pathsnap.hmm import HMMParams, decode_trip, prepare_trip
from pathsnap.labels import make_point_labels
from pathsnap.metrics import route_continuity, route_overlap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(ROOT / "outputs" / "api_requests.log")])
log = logging.getLogger("pathsnap.api")

PARTITION = os.environ.get("PATHSNAP_PARTITION", "evaluator_test")
MODEL = joblib.load(config.MODEL_DIR / "final_model.joblib")
MODEL_CFG = json.loads((config.MODEL_DIR / "final_model_config.json").read_text())
HMM = HMMParams(**json.loads((config.MODEL_DIR / "hmm_params.json").read_text()))

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "partition": PARTITION, "model": MODEL_CFG["model"],
            "model_params": MODEL_CFG["params"], "hmm_params": HMM.__dict__,
            "n_trips": len(list_trips(PARTITION))}


@app.get("/trips")
def trips(request: Request):
    ids = list_trips(PARTITION)
    log.info(f"GET /trips from {request.client.host} -> {len(ids)} trips ({PARTITION})")
    return {"partition": PARTITION, "trips": ids}


def evaluate_server_side(trip_id: str, clf_pred, hmm_pred, path):
    """Loads the reference route (server-side only) and computes accuracy metrics."""
    ref_trip = load_trip(trip_id, PARTITION, with_route=True)
    if ref_trip.route is None:
        return {"available": False}
    L = make_point_labels(ref_trip)
    ok = L["reliable"]
    ov = route_overlap(path, ref_trip.route, ref_trip.arc_len)
    return {
        "available": True,
        "n_evaluated_points": int(ok.sum()),
        "point_accuracy_hmm": float((hmm_pred == L["label"])[ok].mean()),
        "point_accuracy_classifier_only": float((clf_pred == L["label"])[ok].mean()),
        "route_precision": float(ov["route_precision"]),
        "route_recall": float(ov["route_recall"]),
        "route_f1": float(ov["route_f1"]),
    }


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    t0 = time.time()
    if req.trip_id not in list_trips(PARTITION):
        log.warning(f"POST /predict from {request.client.host}: unknown trip {req.trip_id}")
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info(f"POST /predict from {request.client.host}: trip {req.trip_id} - running pipeline")

    # 1) prediction: the model sees only .track/.nodes/.arcs
    trip = load_trip(req.trip_id, PARTITION, with_route=False)
    prep = prepare_trip(trip, MODEL)
    t_clf = time.time()
    res = decode_trip(trip, prep, HMM)
    t_hmm = time.time()

    # 2) evaluation against server-side reference data
    metrics = evaluate_server_side(req.trip_id, res["classifier_pred"], res["hmm_pred"], res["path"])

    path = res["path"]
    a, b = trip.nodes_ll[trip.arcs[path, 0]], trip.nodes_ll[trip.arcs[path, 1]]
    route_latlon = [[[float(p[1]), float(p[0])], [float(q[1]), float(q[0])]] for p, q in zip(a, b)]
    step = max(1, trip.n_points // 2000)
    gps = [[float(lat), float(lon)] for lon, lat in trip.track_ll[::step]]
    out = {
        "trip_id": req.trip_id,
        "n_gps_points": trip.n_points,
        "n_decoder_points": res["n_decoder_points"],
        "n_route_segments": int(len(path)),
        "route_length_km": float(trip.arc_len[path].sum() / 1000),
        "route_continuity": route_continuity(path, trip.arcs),
        "chain_breaks": res["decoder"]["n_breaks"],
        "timing_s": {"classifier": round(t_clf - t0, 3), "hmm": round(t_hmm - t_clf, 3),
                     "total": round(time.time() - t0, 3)},
        "metrics": metrics,
        "predicted_arc_ids": [int(x) for x in path],
        "predicted_route_latlon": route_latlon,
        "gps_latlon": gps,
    }
    log.info(f"trip {req.trip_id}: points={trip.n_points} segments={len(path)} "
             f"acc_hmm={metrics.get('point_accuracy_hmm', float('nan')):.4f} "
             f"acc_clf={metrics.get('point_accuracy_classifier_only', float('nan')):.4f} "
             f"route_f1={metrics.get('route_f1', float('nan')):.4f} total={out['timing_s']['total']}s")
    return out
