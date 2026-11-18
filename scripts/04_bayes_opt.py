"""Step 4 - Bayesian hyper-parameter optimisation of the selected classifier.

Model     : HistGradientBoostingClassifier (chosen in step 3 - see model_comparison.csv)
Objective : mean observation-level top-1 accuracy over 5-fold GroupKFold on the 80 training trips
            (same folds for every trial; student-test and evaluator data are NOT used for the search)
Search    : Gaussian-process + Expected-Improvement optimiser (src/pathsnap/bayesopt.py)
Outputs   : outputs/results/bayes_trials.csv, bayes_opt_history.png, tuned_vs_untuned.csv
            outputs/models/final_model.joblib, outputs/models/final_model_config.json

Run:  python scripts/04_bayes_opt.py [n_trials]
"""
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

from pathsnap import config
from pathsnap.bayesopt import Param, bayes_optimize
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import candidate_metrics, top1_accuracy

SEED = config.RANDOM_SEED

SPACE = [
    Param("learning_rate", 0.01, 0.3, log=True),          # step size of boosting
    Param("max_leaf_nodes", 8, 128, integer=True),        # tree complexity
    Param("min_samples_leaf", 5, 300, log=True, integer=True),   # regularisation of leaves
    Param("l2_regularization", 1e-4, 10.0, log=True),     # L2 penalty on leaf values
    Param("max_features", 0.3, 1.0),                      # feature sub-sampling per split
]


def make_model(p):
    return HistGradientBoostingClassifier(
        learning_rate=p["learning_rate"], max_leaf_nodes=p["max_leaf_nodes"],
        min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2_regularization"],
        max_features=p["max_features"], max_iter=500, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED)


def main(n_trials=30):
    tr = pd.read_pickle(config.FEAT_DIR / "train_candidates.pkl")
    st = pd.read_pickle(config.FEAT_DIR / "student_test_candidates.pkl")
    X, y = tr[FEATURE_COLUMNS].to_numpy(), tr.y.to_numpy()
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=tr.trip_id))

    def objective(p):
        top1, f1, ll = [], [], []
        for itr, iva in folds:
            m = make_model(p).fit(X[itr], y[itr])
            pr = m.predict_proba(X[iva])[:, 1]
            top1.append(top1_accuracy(tr.iloc[iva].assign(score=pr)))
            cm = candidate_metrics(y[iva], pr)
            f1.append(cm["f1_macro"]); ll.append(cm["log_loss"])
        return {"objective": float(np.mean(top1)), "cv_top1_std": float(np.std(top1)),
                "cv_f1_macro": float(np.mean(f1)), "cv_log_loss": float(np.mean(ll))}

    t0 = time.time()
    trials = bayes_optimize(objective, SPACE, n_trials=n_trials, n_init=6, seed=SEED)
    total = time.time() - t0
    tdf = pd.DataFrame(trials)
    tdf.to_csv(config.RESULT_DIR / "bayes_trials.csv", index=False)
    best = tdf.loc[tdf.objective.idxmax()]
    best_p = {p.name: (int(best[p.name]) if p.integer else float(best[p.name])) for p in SPACE}
    print("\nBest parameters:", best_p, f"CV top-1 = {best.objective:.5f}")

    # untuned baseline = default configuration from step 3, evaluated the same way
    base_p = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20,
              "l2_regularization": 0.0, "max_features": 1.0}
    rows = []
    for tag, p in [("untuned (defaults)", base_p), ("tuned (Bayesian)", best_p)]:
        cv = objective(p)
        t1 = time.time()
        m = make_model(p).fit(X, y)
        train_s = time.time() - t1
        t1 = time.time()
        pr = m.predict_proba(st[FEATURE_COLUMNS].to_numpy())[:, 1]
        inf_s = time.time() - t1
        cm = candidate_metrics(st.y.to_numpy(), pr)
        rows.append({"model": tag, **p, "cv_top1": cv["objective"], "cv_f1_macro": cv["cv_f1_macro"],
                     "cv_log_loss": cv["cv_log_loss"],
                     "st_top1": top1_accuracy(st.assign(score=pr)), "st_f1_macro": cm["f1_macro"],
                     "st_log_loss": cm["log_loss"], "train_time_s": train_s, "inference_time_s": inf_s,
                     "n_boosting_iters": int(m.n_iter_)})
        if tag.startswith("tuned"):
            joblib.dump(m, config.MODEL_DIR / "final_model.joblib")
            (config.MODEL_DIR / "final_model_config.json").write_text(json.dumps(
                {"model": "HistGradientBoostingClassifier", "params": best_p, "features": FEATURE_COLUMNS,
                 "cv_top1": cv["objective"], "bayes_trials": n_trials, "search_seconds": total}, indent=2))
    comp = pd.DataFrame(rows)
    comp.to_csv(config.RESULT_DIR / "tuned_vs_untuned.csv", index=False)
    print(comp.T.to_string())
    print(f"\nTotal optimisation time: {total / 60:.1f} min for {n_trials} trials")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tdf.trial, tdf.objective, "o", label="trial objective")
    ax.plot(tdf.trial, tdf.best_so_far, "-", label="best so far")
    ax.axvline(5.5, color="grey", ls=":", label="random init | GP-EI")
    ax.set(xlabel="trial", ylabel="CV top-1 accuracy", title="Bayesian optimisation of HistGradientBoosting")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "bayes_opt_history.png", dpi=120)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
