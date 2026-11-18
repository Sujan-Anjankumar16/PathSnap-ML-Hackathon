"""Turning the reference route (a sequence of arcs) into one label per GPS point.

The .route file only says WHICH arcs were driven, in order - not which GPS point belongs to
which arc. We find the assignment with a monotone alignment (dynamic programming):

    each GPS point i gets a route position j(i), with j(i) non-decreasing in time,
    minimising  sum_i  dist(point_i, arc route[j(i)]).

Monotonicity is what makes this correct for routes that pass the same street twice:
a point can only be matched to a route arc that comes at or after the previous point's arc.

Recurrence:  C[i, j] = D[i, j] + min_{k <= j} C[i-1, k]
The inner min is a running (cumulative) minimum, so the whole thing is O(T * R).
"""
from __future__ import annotations

import numpy as np

from . import config
from .data import Trip, point_segment_distance


def align_route(trip: Trip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (label_arc_id, route_position, distance_to_label_arc) for every GPS point."""
    R = trip.route
    a, b = trip.arc_p0[R], trip.arc_p1[R]
    P = trip.track_xy
    T, M = len(P), len(R)

    C_prev = None
    back = np.empty((T, M), dtype=np.int32)          # argmin of the running minimum
    for i in range(T):
        d, _ = point_segment_distance(P[i, 0], P[i, 1], a[:, 0], a[:, 1], b[:, 0], b[:, 1])
        d = d.astype(np.float64)
        if C_prev is None:
            C = d
            back[i] = np.arange(M)
        else:
            run_min = np.minimum.accumulate(C_prev)
            # index where the running min was achieved
            is_new_min = np.r_[True, C_prev[1:] < run_min[:-1]]
            arg = np.maximum.accumulate(np.where(is_new_min, np.arange(M), 0))
            C = d + run_min
            back[i] = arg
        C_prev = C

    j = np.empty(T, dtype=np.int64)
    j[-1] = int(np.argmin(C_prev))
    for i in range(T - 1, 0, -1):
        j[i - 1] = back[i, j[i]]

    labels = R[j]
    dist, _ = point_segment_distance(P[:, 0], P[:, 1], trip.arc_p0[labels, 0], trip.arc_p0[labels, 1],
                                     trip.arc_p1[labels, 0], trip.arc_p1[labels, 1])
    return labels, j, dist


def make_point_labels(trip: Trip, repair_margin_m: float = 20.0,
                      max_dist_m: float = config.LABEL_MAX_DIST_M):
    """Monotone alignment + repair for parts of the GPS track that the .route does not cover.

    EDA showed (trips 00000033, 00000054) that the GPS track can pass over the same streets
    again after the reference route has ended; the monotone alignment then pins those points to
    the last route arc. For such points we fall back to the closest route arc whose direction
    agrees with the direction of travel. Points that still have no plausible reference arc are
    marked unreliable and are excluded from training and evaluation.

    Returns a dict with: label (arc id), dist (m), reliable (bool), source ('monotone'|'repaired'|'none').
    """
    labels, j, dist = align_route(trip)
    R = np.unique(trip.route)
    P = trip.track_xy
    T = len(P)
    # direction of travel from neighbouring points
    i0 = np.clip(np.arange(T) - 2, 0, T - 1)
    i1 = np.clip(np.arange(T) + 2, 0, T - 1)
    v = P[i1] - P[i0]
    moving = np.hypot(v[:, 0], v[:, 1]) >= 3.0
    head = np.arctan2(v[:, 1], v[:, 0])

    source = np.array(["monotone"] * T, dtype=object)
    bad = dist > max_dist_m
    for i in np.flatnonzero(bad):
        d, _ = point_segment_distance(P[i, 0], P[i, 1], trip.arc_p0[R, 0], trip.arc_p0[R, 1],
                                      trip.arc_p1[R, 0], trip.arc_p1[R, 1])
        if moving[i]:
            agree = np.cos(head[i] - trip.arc_bearing[R]) > 0.5
            d = np.where(agree, d, np.inf)
        k = int(np.argmin(d))
        if d[k] + repair_margin_m < dist[i]:
            labels[i], dist[i], source[i] = R[k], d[k], "repaired"
    reliable = dist <= max_dist_m
    source[~reliable] = "none"
    return {"label": labels, "dist": dist, "reliable": reliable, "source": source, "route_pos": j}
