"""Step 5 - HMM / Viterbi route reconstruction: tuning and classifier-only vs classifier+HMM comparison.

Decoder tuning (training trips only, no leakage):
  For each training trip we use a classifier trained on the OTHER folds (GroupKFold, 5 folds), so the
  emission probabilities on that trip are honest out-of-fold predictions. beta / emission weight /
  minimum movement are chosen by grid search on a fixed sample of training trips.

Final comparison (student-test trips) with the final tuned classifier:
  * observation-level top-1 accuracy (points with a reliable reference label)
  * route continuity of the point sequence and of the reconstructed path
  * sequence-level length-weighted route precision / recall / F1 against the .route file
  * runtime per trip
Outputs: outputs/results/hmm_tuning.csv, hmm_params.json, hmm_vs_classifier.csv, hmm_vs_classifier.png,
         outputs/maps/student_test_<trip>.html (Folium), outputs/results/pred_<trip>.png

Run:  python scripts/05_hmm_eval.py
"""
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config, viz
from pathsnap.data import list_trips, load_trip
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.hmm import HMMParams, decode_trip, prepare_trip
from pathsnap.labels import make_point_labels
from pathsnap.metrics import route_continuity, route_overlap

N_TUNE_TRIPS = 16


def evaluate(trip, res, L):
    ok = L["reliable"]
    out = {
        "trip_id": trip.trip_id, "n_points": trip.n_points, "n_eval_points": int(ok.sum()),
        "clf_top1": float((res["classifier_pred"] == L["label"])[ok].mean()),
        "hmm_top1": float((res["hmm_pred"] == L["label"])[ok].mean()),
        "clf_point_continuity": route_continuity(res["classifier_pred"], trip.arcs),
        "hmm_point_continuity": route_continuity(res["hmm_pred"], trip.arcs),
        "hmm_path_continuity": route_continuity(res["path"], trip.arcs),
        "n_chain_breaks": res["decoder"]["n_breaks"],
    }
    clf_ov = route_overlap(res["classifier_pred"], trip.route, trip.arc_len)
    hmm_ov = route_overlap(res["path"], trip.route, trip.arc_len)
    out.update({f"clf_{k}": v for k, v in clf_ov.items()})
    out.update({f"hmm_{k}": v for k, v in hmm_ov.items()})
    return out


def main():
    cfg = json.loads((config.MODEL_DIR / "final_model_config.json").read_text())
    final = joblib.load(config.MODEL_DIR / "final_model.joblib")
    params = cfg["params"]

    # ---------------- decoder tuning on training trips with out-of-fold classifiers -----------
    tr = pd.read_pickle(config.FEAT_DIR / "train_candidates.pkl")
    trips = list_trips("train")
    rng = np.random.default_rng(config.RANDOM_SEED)
    sizes = {t: len(load_trip(t).t) for t in trips}
    pool = [t for t in trips if sizes[t] <= 4000]
    tune_trips = sorted(rng.choice(pool, size=min(N_TUNE_TRIPS, len(pool)), replace=False))
    gkf = GroupKFold(n_splits=5)
    fold_of = {}
    fold_models = []
    for k, (itr, iva) in enumerate(gkf.split(tr, tr.y, groups=tr.trip_id)):
        for t in tr.iloc[iva].trip_id.unique():
            fold_of[t] = k
        m = HistGradientBoostingClassifier(**params, max_iter=500, early_stopping=True,
                                           n_iter_no_change=20, random_state=config.RANDOM_SEED)
        fold_models.append(m.fit(tr.iloc[itr][FEATURE_COLUMNS].to_numpy(), tr.iloc[itr].y.to_numpy()))

    prepared = {}
    for t in tune_trips:
        trip = load_trip(t)
        prepared[t] = (trip, prepare_trip(trip, fold_models[fold_of[t]]), make_point_labels(trip))
    print(f"Tuning decoder on {len(tune_trips)} training trips: {tune_trips}")

    grid = list(itertools.product([5, 15, 40, 100], [0.5, 1.0, 2.0], [5.0, 15.0]))
    rows = []
    for beta, ew, mm in grid:
        p = HMMParams(beta=beta, emission_weight=ew, min_move_m=mm)
        accs, f1s = [], []
        t0 = time.time()
        for t, (trip, prep, L) in prepared.items():
            ev = evaluate(trip, decode_trip(trip, prep, p), L)
            accs.append(ev["hmm_top1"]); f1s.append(ev["hmm_route_f1"])
        rows.append({"beta": beta, "emission_weight": ew, "min_move_m": mm, "train_hmm_top1": np.mean(accs),
                     "train_route_f1": np.mean(f1s), "seconds": time.time() - t0})
        print(rows[-1])
    tune = pd.DataFrame(rows)
    tune["clf_top1_reference"] = np.mean([evaluate(tr_, {"classifier_pred": pr["classifier_pred"],
                                                       "hmm_pred": pr["classifier_pred"],
                                                       "path": pr["classifier_pred"],
                                                       "decoder": {"n_breaks": 0}}, L)["clf_top1"]
                                          for tr_, pr, L in prepared.values()])
    tune.to_csv(config.RESULT_DIR / "hmm_tuning.csv", index=False)
    best = tune.loc[tune.train_hmm_top1.idxmax()]
    best_params = HMMParams(beta=float(best.beta), emission_weight=float(best.emission_weight),
                            min_move_m=float(best.min_move_m))
    (config.MODEL_DIR / "hmm_params.json").write_text(json.dumps(best_params.__dict__, indent=2))
    print("Best decoder parameters:", best_params)

    # ---------------- final comparison on student-test trips ----------------------------------
    rows = []
    for t in list_trips("student_test"):
        trip = load_trip(t, "student_test")
        L = make_point_labels(trip)
        t0 = time.time()
        prep = prepare_trip(trip, final)
        t1 = time.time()
        res = decode_trip(trip, prep, best_params)
        t2 = time.time()
        ev = evaluate(trip, res, L)
        ev.update({"time_classifier_s": t1 - t0, "time_hmm_s": t2 - t1})
        rows.append(ev)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.items()})
        viz.static_trip_plot(trip, config.RESULT_DIR / f"pred_{t}.png", pred_arcs=res["path"])
        viz.folium_trip_map(trip, config.MAP_DIR / f"student_test_{t}.html", pred_arcs=res["path"])
    comp = pd.DataFrame(rows)
    w = comp.n_eval_points
    summary = {
        "classifier_only_top1 (point-weighted)": float(np.average(comp.clf_top1, weights=w)),
        "classifier+HMM_top1 (point-weighted)": float(np.average(comp.hmm_top1, weights=w)),
        "classifier_only_point_continuity": float(comp.clf_point_continuity.mean()),
        "classifier+HMM_point_continuity": float(comp.hmm_point_continuity.mean()),
        "classifier+HMM_path_continuity": float(comp.hmm_path_continuity.mean()),
        "classifier_only_route_f1": float(comp.clf_route_f1.mean()),
        "classifier+HMM_route_f1": float(comp.hmm_route_f1.mean()),
        "mean_time_per_trip_s": float((comp.time_classifier_s + comp.time_hmm_s).mean()),
    }
    comp.to_csv(config.RESULT_DIR / "hmm_vs_classifier.csv", index=False)
    (config.RESULT_DIR / "hmm_vs_classifier_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(comp))
    ax[0].bar(x - 0.2, comp.clf_top1, 0.4, label="classifier only")
    ax[0].bar(x + 0.2, comp.hmm_top1, 0.4, label="classifier + HMM")
    ax[0].set_xticks(x, comp.trip_id, rotation=45)
    ax[0].set(title="Observation-level top-1 accuracy (student test)", ylim=(0.6, 1.0))
    ax[0].legend()
    ax[1].bar(x - 0.2, comp.clf_route_f1, 0.4, label="classifier only")
    ax[1].bar(x + 0.2, comp.hmm_route_f1, 0.4, label="classifier + HMM")
    ax[1].set_xticks(x, comp.trip_id, rotation=45)
    ax[1].set(title="Sequence-level route F1 (length weighted)", ylim=(0.6, 1.0))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "hmm_vs_classifier.png", dpi=120)


if __name__ == "__main__":
    main()
