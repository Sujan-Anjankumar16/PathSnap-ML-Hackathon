"""Sanity tests for the PathSnap pipeline.

Run from the project root (needs the data folder next to the project):
    python tests/test_pipeline.py          # plain Python
    python -m pytest tests/ -q             # or with pytest, if installed

They check the building blocks the whole pipeline depends on:
geometry, projection, label alignment, candidate generation and feature table shape.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pathsnap.data import LocalProjection, point_segment_distance, list_trips, load_trip


def test_point_segment_distance():
    # point above the middle of a horizontal 10 m segment -> distance 3, t = 0.5
    d, t = point_segment_distance(np.array(5.0), np.array(3.0), 0.0, 0.0, 10.0, 0.0)
    assert abs(float(d) - 3.0) < 1e-9 and abs(float(t) - 0.5) < 1e-9
    # point beyond the end -> distance to the end point, t clipped to 1
    d, t = point_segment_distance(np.array(13.0), np.array(4.0), 0.0, 0.0, 10.0, 0.0)
    assert abs(float(d) - 5.0) < 1e-9 and float(t) == 1.0


def test_projection_round_trip():
    proj = LocalProjection(4.75, 50.83)
    lon, lat = np.array([4.7512, 4.7601]), np.array([50.8301, 50.8255])
    back = proj.to_lonlat(proj.to_xy(lon, lat))
    assert np.allclose(back[:, 0], lon) and np.allclose(back[:, 1], lat)
    # 0.001 degree of latitude is about 111 m everywhere
    xy = proj.to_xy(np.array([4.75, 4.75]), np.array([50.83, 50.831]))
    assert 110 < np.hypot(*(xy[1] - xy[0])) < 112


def test_labels_and_candidates_on_a_small_trip():
    from pathsnap.labels import make_point_labels
    from pathsnap.features import FEATURE_COLUMNS, build_candidate_table

    trips = list_trips("train")
    if not trips:
        print("  (skipped: data folder not found)")
        return
    trip = load_trip("00000046" if "00000046" in trips else trips[0], "train")
    L = make_point_labels(trip)
    assert len(L["label"]) == trip.n_points
    assert L["reliable"].mean() > 0.8                       # most points get a reference segment
    assert np.all(np.diff(L["route_pos"]) >= 0)             # alignment only moves forward

    pts = np.arange(0, trip.n_points, 10)
    table = build_candidate_table(trip, pts, L["label"], L["dist"])
    assert set(FEATURE_COLUMNS) <= set(table.columns)
    assert table.groupby("point_idx").size().max() <= 12    # candidate cap
    hit = table[table.point_idx.isin(pts[L["reliable"][pts]])].groupby("point_idx").y.max()
    assert hit.mean() > 0.95                                 # reference segment is almost always a candidate


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS  {name}")
    print("All tests passed.")
