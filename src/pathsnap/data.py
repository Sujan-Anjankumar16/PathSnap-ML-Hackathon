"""Reading raw trip files and turning them into metric (x, y) geometry.

File formats (all whitespace separated, no header):
  .track : lon  lat  t_seconds          one row per GPS observation
  .nodes : lon  lat                     row index = node_id
  .arcs  : from_node  to_node           row index = arc_id (directed road segment)
  .route : arc_id                       reference sequence of arcs driven (not given for evaluator trips)

Coordinates are degrees. We project every trip to a local equirectangular plane (metres)
centred on the trip's own GPS points. Each trip covers only a few km, so the distortion of
this projection is far below GPS noise (<0.1 % over 10 km), and it lets us use ordinary
Euclidean geometry (point-to-segment distance, bearings) in metres.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

EARTH_R = 6_371_000.0


def list_trips(partition: str = "train") -> list[str]:
    root = {"train": config.TRAIN_DIR, "student_test": config.STUDENT_TEST_DIR,
            "evaluator_test": config.EVAL_DIR}[partition]
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _read(path: Path, dtype=float, ndmin=2):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


class LocalProjection:
    """Equirectangular projection around (lon0, lat0) -> metres."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = lon0, lat0
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    def to_xy(self, lon, lat):
        return np.column_stack(((np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky))

    def to_lonlat(self, xy):
        xy = np.asarray(xy)
        return np.column_stack((xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0))


@dataclass
class Trip:
    trip_id: str
    partition: str
    track_ll: np.ndarray          # (T, 2) lon, lat
    t: np.ndarray                 # (T,) seconds
    nodes_ll: np.ndarray          # (N, 2)
    arcs: np.ndarray              # (A, 2) int from,to
    route: np.ndarray | None      # (R,) arc ids, None if hidden
    proj: LocalProjection = field(init=False)
    track_xy: np.ndarray = field(init=False)
    nodes_xy: np.ndarray = field(init=False)

    def __post_init__(self):
        lon0, lat0 = self.track_ll.mean(axis=0)
        self.proj = LocalProjection(lon0, lat0)
        self.track_xy = self.proj.to_xy(self.track_ll[:, 0], self.track_ll[:, 1])
        self.nodes_xy = self.proj.to_xy(self.nodes_ll[:, 0], self.nodes_ll[:, 1])
        a, b = self.nodes_xy[self.arcs[:, 0]], self.nodes_xy[self.arcs[:, 1]]
        self.arc_p0, self.arc_p1 = a, b
        d = b - a
        self.arc_len = np.hypot(d[:, 0], d[:, 1])
        # bearing in radians, 0 = east, counter-clockwise (math convention)
        self.arc_bearing = np.arctan2(d[:, 1], d[:, 0])

    @property
    def n_points(self):
        return len(self.t)


def load_trip(trip_id: str, partition: str = "train", with_route: bool = True) -> Trip:
    root = {"train": config.TRAIN_DIR, "student_test": config.STUDENT_TEST_DIR,
            "evaluator_test": config.EVAL_DIR}[partition]
    d = root / trip_id
    track = _read(d / f"{trip_id}.track")
    nodes = _read(d / f"{trip_id}.nodes")
    arcs = _read(d / f"{trip_id}.arcs", dtype=np.int64)
    route = _read(d / f"{trip_id}.route", dtype=np.int64, ndmin=1) if with_route else None

    # basic cleaning: drop non-finite rows and enforce increasing time
    ok = np.isfinite(track).all(axis=1)
    track = track[ok]
    order = np.argsort(track[:, 2], kind="stable")
    track = track[order]
    keep = np.r_[True, np.diff(track[:, 2]) > 0]      # remove duplicate timestamps
    track = track[keep]
    return Trip(trip_id, partition, track[:, :2], track[:, 2], nodes, arcs, route)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Vectorised distance from points to segments, plus projection fraction t in [0,1]."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = ((px - ax) * dx + (py - ay) * dy) / L2
    t = np.where(L2 > 0, np.clip(t, 0.0, 1.0), 0.0)
    cx, cy = ax + t * dx, ay + t * dy
    return np.hypot(px - cx, py - cy), t
