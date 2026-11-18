"""Central configuration for PathSnap.

All paths are relative to the project root so the code runs on any laptop.
The raw data folder can be overridden with the PATHSNAP_DATA environment variable.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw dataset: by default the "Student folder" that sits next to ML-HACKATHON
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", PROJECT_ROOT.parent / "Student folder"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips will be placed here on the API laptop on evaluation day
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DATA", DATA_DIR / "evaluator_test"))

OUT_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUT_DIR / "eda"
FEAT_DIR = OUT_DIR / "features"
MODEL_DIR = OUT_DIR / "models"
RESULT_DIR = OUT_DIR / "results"
MAP_DIR = OUT_DIR / "maps"
for _d in (EDA_DIR, FEAT_DIR, MODEL_DIR, RESULT_DIR, MAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ---- Candidate generation -------------------------------------------------
CANDIDATE_RADIUS_M = 50.0     # road segments within this distance of a GPS point
MAX_CANDIDATES = 12           # keep at most this many (closest first)
FALLBACK_K = 3                # if nothing is within the radius, keep the K nearest
DENSIFY_STEP_M = 10.0         # sampling step used to index long road segments

# ---- Training-set construction -------------------------------------------
TRAIN_POINT_STRIDE = 5        # use every 5th GPS point of a training trip (1 Hz data is very redundant)
LABEL_MAX_DIST_M = 30.0       # reference labels farther than this from the GPS point are treated as unreliable (EDA: p95 ~ 13 m)

# ---- HMM ---------------------------------------------------------------------
HMM_MIN_MOVE_M = 10.0         # skip GPS points that moved less than this since the last kept point
HMM_BETA = 20.0               # transition scale (metres); tuned later
HMM_GAP_SECONDS = 60.0        # a time gap larger than this starts a new HMM segment
