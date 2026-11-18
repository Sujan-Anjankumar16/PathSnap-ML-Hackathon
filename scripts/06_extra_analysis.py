"""Step 6 - Extra analysis for the report.

(a) Training / validation curves for the models that train iteratively:
      * MLP: training loss per epoch (loss_curve_) and validation accuracy per epoch
        (validation_scores_, from early stopping's internal 10 % hold-out)
      * HistGradientBoosting (final tuned model): training and validation loss per boosting iteration
        (train_score_ / validation_score_, from its internal 10 % early-stopping hold-out)
    Other models (logistic regression, trees, forests, KNN, NB, SVM) do not produce epoch-wise
    histories, so no curves are drawn for them.

(b) Failure analysis of the student-test trip where the HMM reduced accuracy (00000007).

Outputs: outputs/results/loss_curves.png, trip_00000007_analysis.png, trip_00000007_analysis.md

Run:  python scripts/06_extra_analysis.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathsnap import config
from pathsnap.data import load_trip
from pathsnap.hmm import HMMParams, decode_trip, prepare_trip
from pathsnap.labels import make_point_labels


def loss_curves():
    mlp = joblib.load(config.MODEL_DIR / "mlp.joblib")[-1]        # last pipeline step = MLPClassifier
    hgb = joblib.load(config.MODEL_DIR / "final_model.joblib")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, "o-", ms=3, label="training loss (log loss)")
    ax[0].set(xlabel="epoch", ylabel="training log loss", title="MLP (64, 32): training history")
    a2 = ax[0].twinx()
    a2.plot(np.arange(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_, "s--", ms=3,
            color="tab:orange", label="validation accuracy")
    a2.set_ylabel("validation accuracy (early-stopping hold-out)", color="tab:orange")
    best = int(np.argmax(mlp.validation_scores_)) + 1
    ax[0].axvline(best, color="grey", ls=":", label=f"best epoch = {best}")
    ax[0].legend(loc="upper center")

    it = np.arange(len(hgb.train_score_))
    ax[1].plot(it, -hgb.train_score_, label="training loss")
    ax[1].plot(it, -hgb.validation_score_, label="validation loss")
    ax[1].set(xlabel="boosting iteration", ylabel="log loss",
              title=f"HistGradientBoosting (tuned): {hgb.n_iter_} iterations")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / "loss_curves.png", dpi=120)
    plt.close(fig)
    gap = float(-hgb.validation_score_[-1] - (-hgb.train_score_[-1]))
    print(f"MLP: {len(mlp.loss_curve_)} epochs, best validation epoch {best}; "
          f"HGB: {hgb.n_iter_} iterations, final train-val loss gap = {gap:.4f}")


def trip7_analysis(tid="00000007"):
    model = joblib.load(config.MODEL_DIR / "final_model.joblib")
    params = HMMParams(**json.loads((config.MODEL_DIR / "hmm_params.json").read_text()))
    trip = load_trip(tid, "student_test")
    L = make_point_labels(trip)
    res = decode_trip(trip, prepare_trip(trip, model), params)
    ok, lab = L["reliable"], L["label"]
    clf_ok = res["classifier_pred"] == lab
    hmm_ok = res["hmm_pred"] == lab

    xy, t = trip.track_xy, trip.t
    step = np.r_[0, np.hypot(*np.diff(xy, axis=0).T)]
    speed = np.convolve(step, np.ones(7) / 7, mode="same") * 3.6          # km/h, 7-s average
    T = trip.n_points
    i0, i1 = np.clip(np.arange(T) - 3, 0, T - 1), np.clip(np.arange(T) + 3, 0, T - 1)
    v = xy[i1] - xy[i0]
    hmm_err = ok & clf_ok & ~hmm_ok
    # is the HMM's wrong answer the reverse twin of the reference arc?
    a_h, a_r = trip.arcs[res["hmm_pred"]], trip.arcs[lab]
    twin = (a_h[:, 0] == a_r[:, 1]) & (a_h[:, 1] == a_r[:, 0])

    first = int(np.flatnonzero(hmm_err)[np.searchsorted(np.flatnonzero(hmm_err), int(0.8 * T))] if hmm_err.any() else 0)
    tail = np.arange(first, T)
    stats = {
        "trip": tid, "points_evaluated": int(ok.sum()),
        "classifier_accuracy": float(clf_ok[ok].mean()), "hmm_accuracy": float(hmm_ok[ok].mean()),
        "points_classifier_right_hmm_wrong": int(hmm_err.sum()),
        "of_which_reverse_twin_of_reference": int((hmm_err & twin).sum()),
        "tail_start_index": first, "tail_points": int(len(tail)),
        "tail_median_speed_kmh": float(np.median(speed[tail])),
        "rest_median_speed_kmh": float(np.median(speed[:first])),
        "tail_median_dist_to_reference_m": float(np.median(L["dist"][tail])),
        "rest_median_dist_to_reference_m": float(np.median(L["dist"][:first])),
        "hmm_accuracy_excluding_tail": float(hmm_ok[:first][ok[:first]].mean()),
        "classifier_accuracy_excluding_tail": float(clf_ok[:first][ok[:first]].mean()),
        "track_end_to_route_end_m": float(np.hypot(*(xy[-1] - trip.arc_p1[trip.route[-1]]))),
    }

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t - t[0], speed, lw=1)
    ax[0].axvspan(t[first] - t[0], t[-1] - t[0], color="tab:red", alpha=0.12, label="final low-speed segment")
    ax[0].set(ylabel="speed (km/h, 7-s mean)", title=f"Trip {tid}: where classifier+HMM loses to classifier-only")
    ax[0].legend()
    ax[1].plot(t - t[0], L["dist"], lw=1, color="0.4", label="distance GPS -> aligned reference segment")
    ax[1].scatter((t - t[0])[hmm_err], L["dist"][hmm_err], s=6, color="tab:red", label="HMM wrong, classifier right")
    ax[1].set(xlabel="time since start (s)", ylabel="metres")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.RESULT_DIR / f"trip_{tid}_analysis.png", dpi=120)
    plt.close(fig)

    md = f"""# Why the HMM reduced accuracy on trip {tid} (generated by scripts/06_extra_analysis.py)

* Classifier only: {stats['classifier_accuracy']:.2%}; classifier + HMM: {stats['hmm_accuracy']:.2%}
  on {stats['points_evaluated']} evaluated points.
* {stats['points_classifier_right_hmm_wrong']} points are right for the classifier but wrong after the HMM;
  {stats['of_which_reverse_twin_of_reference']} of them are the **reverse-direction twin** of the reference segment
  (same geometry, opposite direction).
* They are concentrated in the last {stats['tail_points']} observations (from index {stats['tail_start_index']}):
  median speed there is {stats['tail_median_speed_kmh']:.1f} km/h versus {stats['rest_median_speed_kmh']:.1f} km/h
  for the rest of the trip, and the GPS is a median {stats['tail_median_dist_to_reference_m']:.1f} m from the reference
  segment versus {stats['rest_median_dist_to_reference_m']:.1f} m elsewhere.
* The GPS track ends {stats['track_end_to_route_end_m']:.0f} m away from the end of the reference route.

Interpretation: at the end of the trip the vehicle drives much more slowly, with stops and frequent changes of
direction, 10-30 m beside a two-way road (e.g. manoeuvring near its destination). With little net movement the
measured direction of travel is dominated by GPS noise, so the direction feature cannot reliably separate the two
twin arcs, and the HMM transition model - which assumes forward driving along directed segments - prefers whichever
direction is consistent with the noisy zig-zag. The reference label itself is uncertain here (the alignment pins these
points to one arc of the route although the vehicle is 10-30 m away and not clearly moving along it).

Excluding this final segment, accuracy is {stats['classifier_accuracy_excluding_tail']:.2%} (classifier) vs
{stats['hmm_accuracy_excluding_tail']:.2%} (HMM).

Possible improvements: detect stationary / low-speed segments and keep the previous segment instead of decoding
direction from noise; allow a U-turn transition with a penalty.
"""
    (config.RESULT_DIR / f"trip_{tid}_analysis.md").write_text(md)
    print(md)


if __name__ == "__main__":
    loss_curves()
    trip7_analysis()
