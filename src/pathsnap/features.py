"""Feature engineering: one row per (GPS point, candidate road segment).

The ML model is a binary classifier: "is this candidate the road the vehicle was on?".
For each GPS point, the candidate with the highest predicted probability is the
classifier-only answer; the probabilities also become HMM emission probabilities.

Feature groups
  geometric   : distance to the segment, rank / relative distance among the point's candidates,
                distance from a smoothed position, projection position along the segment
  directional : agreement between the vehicle's direction of travel and the segment direction
                (crucial for two-way roads, which appear as two arcs with identical geometry)
  kinematic   : speed, time gap since the previous observation
  topological : node degrees, whether a reverse twin arc exists, segment length
  contextual  : whether the candidate was also a candidate / connected to the nearest road
                at the previous GPS point (cheap temporal consistency)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .candidates import SegmentIndex
from .data import Trip, point_segment_distance

FEATURE_COLUMNS = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "dist_z", "n_cand",
    "smooth_dist", "t_along", "at_endpoint",
    "heading_cos", "heading_absdiff", "heading_valid",
    "speed", "dt_prev",
    "arc_len", "from_in_deg", "to_out_deg", "has_twin",
    "in_prev_cands", "conn_prev_nearest",
]


def _heading_and_speed(xy: np.ndarray, t: np.ndarray, w: int = 2):
    T = len(xy)
    i0 = np.clip(np.arange(T) - w, 0, T - 1)
    i1 = np.clip(np.arange(T) + w, 0, T - 1)
    d = xy[i1] - xy[i0]
    move = np.hypot(d[:, 0], d[:, 1])
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = np.arctan2(d[:, 1], d[:, 0])
    return heading, move, move / dt


def _smoothed(xy: np.ndarray, w: int = 2):
    s = pd.DataFrame(xy).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
    return s


class TopologyInfo:
    def __init__(self, trip: Trip):
        arcs = trip.arcs
        n = len(trip.nodes_xy)
        self.out_deg = np.bincount(arcs[:, 0], minlength=n)
        self.in_deg = np.bincount(arcs[:, 1], minlength=n)
        key = arcs[:, 0].astype(np.int64) * (n + 1) + arcs[:, 1]
        rkey = arcs[:, 1].astype(np.int64) * (n + 1) + arcs[:, 0]
        self.has_twin = np.isin(rkey, key)


def build_candidate_table(trip: Trip, point_idx: np.ndarray | None = None,
                          labels: np.ndarray | None = None, label_dist: np.ndarray | None = None,
                          index: SegmentIndex | None = None) -> pd.DataFrame:
    """Builds the candidate/feature table for the selected GPS points of one trip."""
    T = trip.n_points
    if point_idx is None:
        point_idx = np.arange(T)
    index = index or SegmentIndex(trip)
    topo = TopologyInfo(trip)

    xy = trip.track_xy
    heading, move, speed = _heading_and_speed(xy, trip.t)
    sm = _smoothed(xy)
    dt_prev = np.r_[0.0, np.diff(trip.t)]

    # Candidates are generated for the selected points AND their immediately preceding observation,
    # so the contextual features always refer to the previous raw GPS observation (i-1). This keeps
    # the features identical whether we use every point (test / API) or a subsample (training).
    point_idx = np.asarray(point_idx)
    need = np.unique(np.r_[point_idx, point_idx[point_idx > 0] - 1])
    cand_list = index.query(xy[need])
    cand_of = dict(zip(need.tolist(), cand_list))
    cols = {}
    for i in point_idx:
        arcs, d, tt = cand_of[int(i)]
        if i > 0:
            prev_set = cand_of[int(i) - 1][0]
            prev_nearest = prev_set[0]
        else:
            prev_set, prev_nearest = None, None
        n = len(arcs)
        sd, _ = point_segment_distance(sm[i, 0], sm[i, 1], trip.arc_p0[arcs, 0], trip.arc_p0[arcs, 1],
                                       trip.arc_p1[arcs, 0], trip.arc_p1[arcs, 1])
        valid = move[i] >= 3.0
        diff = np.abs((heading[i] - trip.arc_bearing[arcs] + np.pi) % (2 * np.pi) - np.pi)
        dmin = d.min()
        std = d.std() if n > 1 else 1.0
        if prev_set is None:
            in_prev = np.zeros(n)
            conn = np.zeros(n)
        else:
            in_prev = np.isin(arcs, prev_set).astype(float)
            pa = prev_nearest
            conn = ((arcs == pa) | (trip.arcs[arcs, 0] == trip.arcs[pa, 1])).astype(float)
        block = {
            "point_idx": np.full(n, i), "arc_id": arcs,
            "dist": d, "dist_rank": np.arange(n), "dist_minus_min": d - dmin,
            "dist_ratio": d / (dmin + 1.0), "dist_z": (d - d.mean()) / (std + 1e-6), "n_cand": np.full(n, n),
            "smooth_dist": sd, "t_along": tt, "at_endpoint": ((tt <= 0.0) | (tt >= 1.0)).astype(float),
            "heading_cos": np.cos(diff) if valid else np.zeros(n),
            "heading_absdiff": diff if valid else np.full(n, np.pi / 2),
            "heading_valid": np.full(n, float(valid)),
            "speed": np.full(n, speed[i]), "dt_prev": np.full(n, dt_prev[i]),
            "arc_len": trip.arc_len[arcs],
            "from_in_deg": topo.in_deg[trip.arcs[arcs, 0]], "to_out_deg": topo.out_deg[trip.arcs[arcs, 1]],
            "has_twin": topo.has_twin[arcs].astype(float),
            "in_prev_cands": in_prev, "conn_prev_nearest": conn,
        }
        if labels is not None:
            block["label_arc"] = np.full(n, labels[i])
            block["label_dist"] = np.full(n, label_dist[i])
            block["y"] = (arcs == labels[i]).astype(np.int8)
        for k, v in block.items():
            cols.setdefault(k, []).append(np.asarray(v))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df.insert(0, "trip_id", trip.trip_id)
    return df
