"""Candidate generation: which road segments could each GPS point be on?

Road segments can be hundreds of metres long, so a KD-tree on segment end points alone would
miss a long segment that passes right next to a GPS point. We therefore "densify" every segment
near the trajectory into sample points every DENSIFY_STEP_M metres, index those in a KD-tree,
look up nearby samples, and then compute the exact point-to-segment distance.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from . import config
from .data import Trip, point_segment_distance


class SegmentIndex:
    def __init__(self, trip: Trip, margin_m: float = 500.0, step: float = config.DENSIFY_STEP_M):
        P = trip.track_xy
        lo, hi = P.min(axis=0) - margin_m, P.max(axis=0) + margin_m
        a, b = trip.arc_p0, trip.arc_p1
        # keep arcs whose bounding box overlaps the (expanded) trajectory bounding box
        amin, amax = np.minimum(a, b), np.maximum(a, b)
        near = np.all(amax >= lo, axis=1) & np.all(amin <= hi, axis=1)
        self.arc_ids = np.flatnonzero(near)
        L = trip.arc_len[self.arc_ids]
        n = np.maximum(1, np.ceil(L / step).astype(int)) + 1          # samples per arc (incl. both ends)
        rep = np.repeat(self.arc_ids, n)
        frac = np.concatenate([np.linspace(0, 1, k) for k in n]) if len(n) else np.zeros(0)
        self.sample_arc = rep
        self.samples = a[rep] + frac[:, None] * (b[rep] - a[rep])
        self.tree = cKDTree(self.samples)
        self.trip = trip
        self.step = step

    def query(self, xy: np.ndarray, radius: float = config.CANDIDATE_RADIUS_M,
              max_k: int = config.MAX_CANDIDATES, fallback_k: int = config.FALLBACK_K):
        """For each point returns a list of (arc_ids, distances, t_along) sorted by distance."""
        trip = self.trip
        # Any segment within `radius` has a sample within radius + step/2, so this search is exact.
        balls = self.tree.query_ball_point(xy, r=radius + self.step / 2 + 1e-6)
        out = []
        for i in range(len(xy)):
            if len(balls[i]) == 0:                       # nothing close: take nearest samples
                _, nn = self.tree.query(xy[i], k=min(len(self.samples), 64))
                arcs = np.unique(self.sample_arc[np.atleast_1d(nn)])
            else:
                arcs = np.unique(self.sample_arc[balls[i]])
            d, t = point_segment_distance(xy[i, 0], xy[i, 1], trip.arc_p0[arcs, 0], trip.arc_p0[arcs, 1],
                                          trip.arc_p1[arcs, 0], trip.arc_p1[arcs, 1])
            order = np.argsort(d, kind="stable")
            arcs, d, t = arcs[order], d[order], t[order]
            inside = d <= radius
            if inside.sum() == 0:
                sel = slice(0, fallback_k)
            else:
                sel = slice(0, min(int(inside.sum()), max_k))
            out.append((arcs[sel], d[sel], t[sel]))
        return out
