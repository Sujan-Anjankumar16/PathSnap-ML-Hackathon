"""Step 2 - Build the candidate / feature tables used for model training.

train        : every TRAIN_POINT_STRIDE-th GPS point with a reliable reference label
student_test : every GPS point with a reliable reference label (used for model comparison)

Outputs (outputs/features/):
  train_candidates.pkl, student_test_candidates.pkl
  candidate_coverage.csv   per-trip candidate miss rate (reference segment not in candidate set)

Run:  python scripts/02_build_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.data import list_trips, load_trip
from pathsnap.features import build_candidate_table
from pathsnap.labels import make_point_labels


def build(partition, stride):
    tables, cov = [], []
    for tid in list_trips(partition):
        t0 = time.time()
        trip = load_trip(tid, partition)
        L = make_point_labels(trip)
        pts = np.arange(0, trip.n_points, stride)
        pts = pts[L["reliable"][pts]]
        if len(pts) == 0:
            continue
        df = build_candidate_table(trip, pts, L["label"], L["dist"])
        hit = df.groupby("point_idx").y.max()
        cov.append({"partition": partition, "trip_id": tid, "points": len(pts),
                    "candidate_rows": len(df), "miss_rate": 1 - hit.mean()})
        tables.append(df)
        print(f"{partition:12s} {tid}  points={len(pts):5d}  rows={len(df):6d}  "
              f"miss={1 - hit.mean():.3f}  ({time.time() - t0:.1f}s)")
    return pd.concat(tables, ignore_index=True), pd.DataFrame(cov)


def main():
    tr, cov_tr = build("train", config.TRAIN_POINT_STRIDE)
    st, cov_st = build("student_test", 1)
    tr.to_pickle(config.FEAT_DIR / "train_candidates.pkl")
    st.to_pickle(config.FEAT_DIR / "student_test_candidates.pkl")
    cov = pd.concat([cov_tr, cov_st])
    cov.to_csv(config.FEAT_DIR / "candidate_coverage.csv", index=False)
    for name, df in [("train", tr), ("student_test", st)]:
        pts = df.groupby(["trip_id", "point_idx"]).y.max()
        print(f"\n{name}: {len(df)} candidate rows, {len(pts)} GPS points, "
              f"positives={df.y.mean():.3f}, candidate miss rate={1 - pts.mean():.4f}")


if __name__ == "__main__":
    main()
