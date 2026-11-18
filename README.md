# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning Hackathon, PES University (RR campus).

PathSnap takes a noisy GPS trajectory and a road network and works out which road segments the vehicle actually drove on:

```
raw trip files ─► cleaning + local metric projection ─► candidate road segments (≤ 50 m)
   ─► 20 engineered features per (GPS point, candidate) ─► ML classifier: P(candidate is correct)
   ─► HMM + Viterbi over the road graph ─► connected predicted route ─► FastAPI ─► evaluator UI
```

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/` | library code: `data.py` (I/O, projection), `labels.py` (route→point labels), `candidates.py`, `features.py`, `metrics.py`, `bayesopt.py` (GP + Expected Improvement), `hmm.py` (Viterbi), `viz.py` (matplotlib + Folium) |
| `scripts/` | the pipeline, run in order `01` → `05` |
| `api/main.py` | FastAPI backend (Laptop 1) |
| `ui/index.html` | evaluator UI (Laptop 2), a single page using Leaflet |
| `outputs/eda/` | dataset inventory, data-quality notes, EDA plots, candidate-radius study |
| `outputs/results/` | model comparison tables/plots, CV folds, Bayesian trials, HMM comparison |
| `outputs/maps/` | interactive Folium maps (HTML) |
| `outputs/models/` | `final_model.joblib`, `final_model_config.json`, `hmm_params.json` |

The raw data is **not** in the repo. By default the code expects it at `../Student folder` (next to this folder). To use another location, set `PATHSNAP_DATA=/path/to/Student folder`.

## Setup (macOS / Linux)

```bash
cd ML-HACKATHON
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce everything

```bash
python scripts/01_eda.py               # dataset exploration, data-quality notes, Folium maps   (~1 min)
python scripts/02_build_features.py    # candidate generation + feature tables                 (~1 min)
python scripts/03_train_models.py      # 9 models, GroupKFold CV + student-test comparison     (~15 min)
python scripts/04_bayes_opt.py 30      # Bayesian optimisation of the chosen model            (~10 min)
python scripts/05_hmm_eval.py          # decoder tuning + classifier-only vs classifier+HMM   (~5 min)
```

Random seed: 42 (`src/pathsnap/config.py`). Every number in the report comes from the CSV/JSON files these scripts write to `outputs/`.

## Two-laptop deployment

**Laptop 1 (API + model):**

```bash
source .venv/bin/activate
export PATHSNAP_EVAL_DATA="/path/to/evaluator_trips"     # folder with one sub-folder per trip id
uvicorn api.main:app --host 0.0.0.0 --port 8000           # leave this terminal visible: it shows the request logs
```

- For a rehearsal with the student-test trips, use `export PATHSNAP_PARTITION=student_test` instead.
- Find Laptop 1's IP address with `ipconfig getifaddr en0` (macOS).
- Allow incoming connections if the macOS firewall asks.

**Laptop 2 (UI):**

```bash
cd ui
python3 -m http.server 5500
```

Then open `http://localhost:5500` in a browser, enter `http://<laptop-1-ip>:8000`, click **Connect**, and pick a trip.

Both laptops must be on the same Wi-Fi network. The UI only receives the predicted route and the accuracy numbers. The reference route is read on the API side, after prediction, to compute the metrics.

## Method summary

- **Point labels.** The `.route` file lists the arcs in driving order. A monotone dynamic-programming alignment assigns each GPS point to one route arc. Points with no reference arc within 30 m are excluded.
- **Candidates.** Road segments are densified every 10 m and indexed in a KD-tree. The search radius is 50 m, chosen by the radius study in `outputs/eda/radius_study.png`.
- **Features.** Distance, rank among candidates, smoothed distance, heading agreement (to tell apart the two directions of a two-way road), speed, time gap, node degrees, twin-arc flag, and consistency with the previous point.
- **Models.** Logistic regression, Gaussian NB, KNN, decision tree, random forest, extra trees, HistGradientBoosting, MLP and RBF-SVM, compared with 5-fold GroupKFold by trip.
- **Bayesian optimisation.** A Gaussian-process surrogate with the Expected-Improvement acquisition function, tuning HistGradientBoosting on CV top-1 accuracy.
- **HMM.** Hidden states are the candidate segments. Emission = classifier probability. Transition = exp(−|network distance − straight-line distance| / β). Viterbi decoding, with chain restarts at GPS gaps or disconnected segments. The decoder parameters are tuned on training trips with out-of-fold classifiers.

## AI-assistance disclosure

Parts of the code were developed with the help of an AI coding assistant (Claude). All experiments were run by the team, and every team member must be able to explain the code and results.
