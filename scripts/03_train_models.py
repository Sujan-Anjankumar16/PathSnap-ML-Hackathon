"""Step 3 - Train and compare candidate classifiers from several algorithm families.

Protocol
  * 5-fold GroupKFold on the 80 training trips (groups = trip_id, so no trip is split across folds)
    -> CV macro-F1, CV log loss, CV top-1 accuracy (mean and std over folds)
  * refit on all training trips -> evaluate on the 10 student-testing trips:
    candidate-level accuracy / precision / recall / macro-F1 / log loss / confusion matrix,
    observation-level top-1 accuracy, candidate miss rate, route continuity (classifier only),
    training time and inference time.

Primary selection metric: student-test observation-level top-1 accuracy (it is exactly the
classifier-only map-matching accuracy), with CV macro-F1 / log loss as supporting evidence
(log loss matters because probabilities are used as HMM emission scores).

Outputs: outputs/results/model_comparison.csv, model_results.json, cv_folds.csv, *.png
         outputs/models/<model>.joblib

Run:  python scripts/03_train_models.py            (all models)
      python scripts/03_train_models.py hgb rf      (selected models)
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
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: F401  (optional extra)
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pathsnap import config
from pathsnap.data import list_trips, load_trip
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import (candidate_metrics, candidate_miss_rate, route_continuity, top1_accuracy,
                              top1_predictions)

SEED = config.RANDOM_SEED


def model_zoo():
    """name -> (family, estimator, description of key configuration, max training rows or None)."""
    return {
        "logreg": ("Linear / probabilistic",
                   make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
                   "StandardScaler + L2 logistic regression, C=1.0", None),
        "gnb": ("Linear / probabilistic", make_pipeline(StandardScaler(), GaussianNB()),
                "Gaussian Naive Bayes, var_smoothing=1e-9", None),
        "knn": ("Instance-based",
                make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
                "StandardScaler + KNN, k=25, distance-weighted, 60k-row training subsample (prediction cost)",
                60000),
        "dtree": ("Tree-based", DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, random_state=SEED),
                  "max_depth=10, min_samples_leaf=20", None),
        "rf": ("Ensemble (bagging)",
               RandomForestClassifier(n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                                      n_jobs=-1, random_state=SEED),
               "200 trees, min_samples_leaf=5, max_features=sqrt", None),
        "extratrees": ("Ensemble (bagging)",
                       ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
                       "200 trees, min_samples_leaf=5", None),
        "hgb": ("Boosting",
                HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                               early_stopping=True, random_state=SEED),
                "lr=0.1, max_iter=300, max_leaf_nodes=31, early stopping", None),
        "mlp": ("Neural network",
                make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                                                              batch_size=512, max_iter=60, early_stopping=True,
                                                              random_state=SEED)),
                "StandardScaler + MLP (64,32), ReLU, Adam, early stopping", None),
        "svm_rbf": ("Kernel-based",
                    make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale", probability=True, random_state=SEED)),
                    "StandardScaler + RBF SVC, C=1, Platt-scaled probabilities, trained on a 20k-row subsample",
                    20000),
    }


def subsample_by_points(df, max_rows, seed=SEED):
    """Subsample whole GPS points (all of a point's candidates together) to roughly max_rows rows."""
    if max_rows is None or len(df) <= max_rows:
        return df
    pts = df[["trip_id", "point_idx"]].drop_duplicates()
    frac = max_rows / len(df)
    keep = pts.sample(frac=frac, random_state=seed)
    return df.merge(keep, on=["trip_id", "point_idx"])


def fit(est, df):
    t0 = time.time()
    est.fit(df[FEATURE_COLUMNS].to_numpy(), df.y.to_numpy())
    return est, time.time() - t0


def predict(est, df):
    t0 = time.time()
    p = est.predict_proba(df[FEATURE_COLUMNS].to_numpy())[:, 1]
    return p, time.time() - t0


def continuity_on(df, scores):
    d = df.assign(score=scores)
    top = top1_predictions(d)
    vals = []
    for tid, g in top.groupby("trip_id"):
        arcs = load_trip(tid, "student_test").arcs
        vals.append(route_continuity(g.arc_id.to_numpy(), arcs))
    return float(np.mean(vals))


def main(selected=None):
    tr = pd.read_pickle(config.FEAT_DIR / "train_candidates.pkl")
    st = pd.read_pickle(config.FEAT_DIR / "student_test_candidates.pkl")
    zoo = model_zoo()
    if selected:
        zoo = {k: v for k, v in zoo.items() if k in selected}

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(tr, tr.y, groups=tr.trip_id))
    results, fold_rows = {}, []
    for name, (family, est, desc, max_rows) in zoo.items():
        print(f"\n=== {name} ({family}) ===")
        cv_f1, cv_ll, cv_top1 = [], [], []
        for k, (itr, iva) in enumerate(folds):
            dtr = subsample_by_points(tr.iloc[itr], max_rows)
            dva = tr.iloc[iva]
            m, _ = fit(clone(est), dtr)
            p, _ = predict(m, dva)
            cm = candidate_metrics(dva.y.to_numpy(), p)
            t1 = top1_accuracy(dva.assign(score=p))
            cv_f1.append(cm["f1_macro"]); cv_ll.append(cm["log_loss"]); cv_top1.append(t1)
            fold_rows.append({"model": name, "fold": k, "f1_macro": cm["f1_macro"], "log_loss": cm["log_loss"],
                              "top1": t1})
            print(f"  fold {k}: macroF1={cm['f1_macro']:.4f}  logloss={cm['log_loss']:.4f}  top1={t1:.4f}")

        # train-set fit (for over/under-fitting evidence) and final student-test evaluation
        dtr_full = subsample_by_points(tr, max_rows)
        m, t_train = fit(clone(est), dtr_full)
        p_tr, _ = predict(m, dtr_full)
        train_cm = candidate_metrics(dtr_full.y.to_numpy(), p_tr)
        p_st, t_inf = predict(m, st)
        st_cm = candidate_metrics(st.y.to_numpy(), p_st)
        res = {
            "family": family, "config": desc,
            "cv_f1_macro_mean": float(np.mean(cv_f1)), "cv_f1_macro_std": float(np.std(cv_f1)),
            "cv_log_loss_mean": float(np.mean(cv_ll)), "cv_top1_mean": float(np.mean(cv_top1)),
            "cv_top1_std": float(np.std(cv_top1)),
            "train_f1_macro": train_cm["f1_macro"], "train_log_loss": train_cm["log_loss"],
            "train_top1": top1_accuracy(dtr_full.assign(score=p_tr)),
            "st_accuracy": st_cm["accuracy"], "st_precision_macro": st_cm["precision_macro"],
            "st_recall_macro": st_cm["recall_macro"], "st_f1_macro": st_cm["f1_macro"],
            "st_log_loss": st_cm["log_loss"], "st_confusion": st_cm["confusion"],
            "st_top1": top1_accuracy(st.assign(score=p_st)),
            "st_candidate_miss_rate": candidate_miss_rate(st),
            "st_route_continuity_classifier_only": continuity_on(st, p_st),
            "train_time_s": t_train, "train_rows": len(dtr_full),
            "inference_time_s": t_inf, "inference_us_per_row": 1e6 * t_inf / len(st),
        }
        results[name] = res
        joblib.dump(m, config.MODEL_DIR / f"{name}.joblib")
        print(f"  CV macroF1={res['cv_f1_macro_mean']:.4f}±{res['cv_f1_macro_std']:.4f}  "
              f"ST top1={res['st_top1']:.4f}  ST macroF1={res['st_f1_macro']:.4f}  "
              f"ST logloss={res['st_log_loss']:.4f}  continuity={res['st_route_continuity_classifier_only']:.4f}  "
              f"train {t_train:.1f}s  infer {t_inf:.2f}s")

    # merge with earlier runs so models can be (re)trained one at a time
    rpath = config.RESULT_DIR / "model_results.json"
    old = json.loads(rpath.read_text()) if rpath.exists() else {}
    old.update(results)
    rpath.write_text(json.dumps(old, indent=2))
    fpath = config.RESULT_DIR / "cv_folds.csv"
    folds_df = pd.DataFrame(fold_rows)
    if fpath.exists():
        prev = pd.read_csv(fpath)
        folds_df = pd.concat([prev[~prev.model.isin(folds_df.model.unique())], folds_df])
    folds_df.to_csv(fpath, index=False)
    make_report(old)


def make_report(results):
    rows = []
    for name, r in results.items():
        rows.append({"Model": name, "Algorithm family": r["family"], "Key configuration": r["config"],
                     "CV macro F1": f"{r['cv_f1_macro_mean']:.4f} ± {r['cv_f1_macro_std']:.4f}",
                     "Top-1 accuracy (student test)": round(r["st_top1"], 4),
                     "Route continuity (classifier only)": round(r["st_route_continuity_classifier_only"], 4),
                     "Student-test macro F1": round(r["st_f1_macro"], 4),
                     "Log loss (student test)": round(r["st_log_loss"], 4),
                     "Inference time (s, all student-test rows)": round(r["inference_time_s"], 3),
                     "Train time (s)": round(r["train_time_s"], 1)})
    table = pd.DataFrame(rows).sort_values("Top-1 accuracy (student test)", ascending=False)
    table.to_csv(config.RESULT_DIR / "model_comparison.csv", index=False)
    print("\n", table.to_string(index=False))

    names = list(table.Model)
    R = [results[n] for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].bar(names, [r["cv_f1_macro_mean"] for r in R], yerr=[r["cv_f1_macro_std"] for r in R], capsize=4)
    ax[0].set(title="5-fold GroupKFold CV macro F1 (train trips)", ylim=(0.5, 1))
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, [r["st_top1"] for r in R], 0.4, label="Top-1 accuracy")
    ax[1].bar(x + 0.2, [r["st_f1_macro"] for r in R], 0.4, label="Macro F1")
    ax[1].set_xticks(x, names)
    ax[1].set(title="Student-test performance", ylim=(0.5, 1))
    ax[1].legend()
    ax[2].bar(names, [r["st_log_loss"] for r in R], color="tab:red")
    ax[2].set(title="Student-test log loss (lower is better)")
    for a in ax:
        a.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)

    # confusion matrices
    n = len(names)
    fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for a, nm, r in zip(np.atleast_1d(ax), names, R):
        cm = np.array(r["st_confusion"])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v}", ha="center", va="center", fontsize=8)
        a.set(title=nm, xticks=[0, 1], yticks=[0, 1], xlabel="pred", ylabel="true")
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "confusion_matrices.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
