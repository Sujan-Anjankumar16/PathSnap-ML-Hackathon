"""HMM + Viterbi route reconstruction.

Hidden state  : (GPS observation k, candidate road segment c) - each state keeps its arc_id and the
                projection position t along the segment, so we never lose the candidate -> segment link.
Emission      : log P(candidate | features) from the selected classifier, renormalised over the
                candidates of the observation and raised to a weight w_e:  w_e * log p_norm.
Transition    : Newson & Krumm style. For consecutive observations with great-circle distance g and
                network distance r (drive from the projection on segment a to the projection on segment b
                along directed road segments), log P = -|r - g| / beta. Unreachable pairs get -inf.
Decoding      : Viterbi.  delta_k(b) = e_k(b) + max_a [ delta_{k-1}(a) + T(a, b) ]
Breaks        : if every transition into observation k is impossible (disconnected network, bad GPS)
                or the time gap exceeds HMM_GAP_SECONDS, the chain is cut and a new chain starts at k.
Down-sampling : 1 Hz points that moved < HMM_MIN_MOVE_M are skipped for the decoder (they carry no new
                information and make transitions unstable); every skipped point is then assigned to the
                closest segment on the decoded path between its neighbouring decoded points.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from . import config
from .data import Trip, point_segment_distance


class RoadGraph:
    """Directed road graph with cached, bounded Dijkstra searches."""

    def __init__(self, trip: Trip):
        self.trip = trip
        n = len(trip.nodes_xy)
        order = np.argsort(trip.arcs[:, 0], kind="stable")
        self.out_arcs = order
        self.out_ptr = np.searchsorted(trip.arcs[order, 0], np.arange(n + 1))
        self.cache: dict[int, tuple[float, dict, dict]] = {}

    def _dijkstra(self, src: int, cutoff: float):
        arcs, L = self.trip.arcs, self.trip.arc_len
        dist, pred = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > cutoff:
                continue
            for a in self.out_arcs[self.out_ptr[u]:self.out_ptr[u + 1]]:
                v = arcs[a, 1]
                nd = d + L[a]
                if nd < dist.get(v, np.inf) and nd <= cutoff:
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        return dist, pred

    def from_node(self, src: int, cutoff: float):
        c = self.cache.get(src)
        if c is None or c[0] < cutoff:
            dist, pred = self._dijkstra(src, cutoff)
            c = (cutoff, dist, pred)
            self.cache[src] = c
        return c[1], c[2]

    def path_arcs(self, src: int, dst: int):
        """Arcs of the cached shortest path src -> dst (empty if src == dst)."""
        if src not in self.cache:
            self.from_node(src, 5000.0)
        _, _, pred = self.cache[src]
        out, v = [], dst
        while v != src:
            a = pred.get(v)
            if a is None:
                return None
            out.append(a)
            v = self.trip.arcs[a, 0]
        return out[::-1]


@dataclass
class HMMParams:
    beta: float = config.HMM_BETA
    emission_weight: float = 1.0
    min_move_m: float = config.HMM_MIN_MOVE_M
    gap_seconds: float = config.HMM_GAP_SECONDS
    cutoff_factor: float = 3.0       # search radius = cutoff_factor * gc distance + cutoff_add
    cutoff_add: float = 250.0


def select_hmm_points(trip: Trip, min_move_m: float):
    xy, keep, last = trip.track_xy, [0], 0
    for i in range(1, trip.n_points):
        if np.hypot(*(xy[i] - xy[last])) >= min_move_m or trip.t[i] - trip.t[last] > config.HMM_GAP_SECONDS:
            keep.append(i)
            last = i
    if keep[-1] != trip.n_points - 1:
        keep.append(trip.n_points - 1)
    return np.array(keep)


def viterbi_decode(trip: Trip, cand_table, scores: np.ndarray, params: HMMParams, graph: RoadGraph | None = None):
    """cand_table: DataFrame with point_idx, arc_id, t_along for the decoder points (sorted by point_idx).
    scores: classifier probability per row. Returns dict with matched arc per decoder point and the
    connected list of path arcs."""
    graph = graph or RoadGraph(trip)
    arcs, L = trip.arcs, trip.arc_len
    df = cand_table.assign(score=scores)
    groups = [g for _, g in df.groupby("point_idx", sort=True)]
    pts = np.array([g.point_idx.iat[0] for g in groups])
    xy, t = trip.track_xy, trip.t

    def emis(g):
        p = np.clip(g.score.to_numpy(), 1e-6, None)
        return params.emission_weight * np.log(p / p.sum())

    K = len(groups)
    states = [(g.arc_id.to_numpy(), g.t_along.to_numpy()) for g in groups]
    deltas = [emis(groups[0])]
    back = [np.full(len(deltas[0]), -1)]
    chain_start = [0]
    n_impossible = 0
    for k in range(1, K):
        a_ids, a_t = states[k - 1]
        b_ids, b_t = states[k]
        gc = float(np.hypot(*(xy[pts[k]] - xy[pts[k - 1]])))
        dt = t[pts[k]] - t[pts[k - 1]]
        e = emis(groups[k])
        trans = np.full((len(a_ids), len(b_ids)), -np.inf)
        if dt <= params.gap_seconds:
            cutoff = params.cutoff_factor * gc + params.cutoff_add
            for ia, (a, ta) in enumerate(zip(a_ids, a_t)):
                rest = (1.0 - ta) * L[a]
                dist_map, _ = graph.from_node(int(arcs[a, 1]), cutoff)
                for ib, (b, tb) in enumerate(zip(b_ids, b_t)):
                    if a == b and tb >= ta - 0.05:          # moving forward on the same segment
                        r = max(tb - ta, 0.0) * L[a]
                    else:
                        dn = dist_map.get(int(arcs[b, 0]))
                        if dn is None:
                            continue
                        r = rest + dn + tb * L[b]
                    trans[ia, ib] = -abs(r - gc) / params.beta
        cand = deltas[-1][:, None] + trans
        best = cand.max(axis=0)
        if not np.isfinite(best).any():          # chain broken -> restart from emissions
            n_impossible += int(dt <= params.gap_seconds)
            deltas.append(e)
            back.append(np.full(len(e), -1))
            chain_start.append(k)
            continue
        back.append(cand.argmax(axis=0))
        deltas.append(best + e)

    choice = np.zeros(K, dtype=int)
    bounds = chain_start + [K]
    for s, nxt in zip(bounds[:-1], bounds[1:]):
        end = nxt - 1
        choice[end] = int(np.argmax(deltas[end]))
        for k in range(end, s, -1):
            choice[k - 1] = back[k][choice[k]]
    matched = np.array([states[k][0][choice[k]] for k in range(K)])
    matched_t = np.array([states[k][1][choice[k]] for k in range(K)])
    return {"points": pts, "arcs": matched, "t_along": matched_t, "chain_start": chain_start,
            "n_breaks": len(chain_start) - 1, "n_impossible_breaks": n_impossible}


def assemble_path(trip: Trip, dec: dict, graph: RoadGraph):
    """Connect consecutive matched segments with shortest paths -> list of arcs actually driven,
    plus, for every decoded point, the index in that list where its segment sits."""
    path, pos = [], []
    arcs = trip.arcs
    chain_starts = set(dec["chain_start"])
    for k, a in enumerate(dec["arcs"]):
        if k == 0 or k in chain_starts or not path:
            if not path or path[-1] != a:
                path.append(int(a))
            pos.append(len(path) - 1)
            continue
        prev = path[-1]
        if a == prev and dec["t_along"][k] >= dec["t_along"][k - 1] - 0.05:
            pos.append(len(path) - 1)
            continue
        src, dst = int(arcs[prev, 1]), int(arcs[a, 0])
        mid = [] if src == dst else graph.path_arcs(src, dst)
        if mid is None:
            mid = []
        path.extend(int(m) for m in mid)
        path.append(int(a))
        pos.append(len(path) - 1)
    return np.array(path, dtype=int), np.array(pos)


def assign_all_points(trip: Trip, dec: dict, path: np.ndarray, pos: np.ndarray):
    """Every GPS observation gets the closest path segment between the path positions of the decoded
    points just before and just after it (keeps the assignment consistent with the decoded route)."""
    T = trip.n_points
    out = np.empty(T, dtype=int)
    pts = dec["points"]
    xy = trip.track_xy
    for k in range(len(pts)):
        i0 = pts[k]
        i1 = pts[k + 1] if k + 1 < len(pts) else T
        lo = pos[k]
        hi = pos[k + 1] if k + 1 < len(pts) else pos[k]
        seg = path[lo:max(hi, lo) + 1]
        out[i0] = path[lo]
        if i1 - i0 > 1:
            idx = np.arange(i0 + 1, i1)
            P = xy[idx]
            d = np.stack([point_segment_distance(P[:, 0], P[:, 1], trip.arc_p0[a, 0], trip.arc_p0[a, 1],
                                                 trip.arc_p1[a, 0], trip.arc_p1[a, 1])[0] for a in seg], axis=1)
            # enforce monotone progress along the path between the two decoded points
            choice = np.maximum.accumulate(np.argmin(d, axis=1))
            out[idx] = seg[choice]
    return out


def prepare_trip(trip: Trip, model):
    """Candidates + features + classifier scores for every GPS point (independent of HMM params)."""
    from .features import FEATURE_COLUMNS, build_candidate_table
    from .candidates import SegmentIndex

    table = build_candidate_table(trip, np.arange(trip.n_points), index=SegmentIndex(trip))
    table["score"] = model.predict_proba(table[FEATURE_COLUMNS].to_numpy())[:, 1]
    top = table.loc[table.groupby("point_idx").score.idxmax()].sort_values("point_idx")
    return {"table": table, "classifier_pred": top.arc_id.to_numpy(), "graph": RoadGraph(trip)}


def decode_trip(trip: Trip, prepared: dict, params: HMMParams | None = None):
    params = params or HMMParams()
    table, graph = prepared["table"], prepared["graph"]
    keep = select_hmm_points(trip, params.min_move_m)
    sub = table[table.point_idx.isin(keep)].sort_values(["point_idx", "dist"])
    dec = viterbi_decode(trip, sub[["point_idx", "arc_id", "t_along"]], sub.score.to_numpy(), params, graph)
    path, pos = assemble_path(trip, dec, graph)
    hmm_pred = assign_all_points(trip, dec, path, pos)
    return {"classifier_pred": prepared["classifier_pred"], "hmm_pred": hmm_pred, "path": path,
            "decoder": dec, "n_decoder_points": len(keep)}


def match_trip(trip: Trip, model, params: HMMParams | None = None):
    """Full pipeline for one trip: candidates -> features -> classifier -> Viterbi -> route."""
    return decode_trip(trip, prepare_trip(trip, model), params)
