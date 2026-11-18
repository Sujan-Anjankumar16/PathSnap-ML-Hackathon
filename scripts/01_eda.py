"""Step 1 - Dataset exploration (train + student_test partitions only).

Produces in outputs/eda/:
  inventory.csv            one row per trip: counts, sizes, sampling, gaps, extent, network connectivity
  data_quality_notes.md    anomalies found automatically
  *.png                    distribution plots
  radius_study.csv / .png  candidate miss rate vs search radius (justifies CANDIDATE_RADIUS_M)
and Folium maps in outputs/maps/ (if folium is installed).

Run from the project root:   python scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config
from pathsnap.data import list_trips, load_trip
from pathsnap.labels import align_route, make_point_labels
from pathsnap.candidates import SegmentIndex
from pathsnap import viz

PARTS = ["train", "student_test"]          # evaluator data is never touched here


def trip_stats(trip):
    root = config.TRAIN_DIR if trip.partition == "train" else config.STUDENT_TEST_DIR
    d = root / trip.trip_id
    sizes = {f"size_{ext}_kb": (d / f"{trip.trip_id}.{ext}").stat().st_size / 1024
             for ext in ("track", "nodes", "arcs", "route")}
    raw = np.loadtxt(d / f"{trip.trip_id}.track", ndmin=2)
    dt = np.diff(trip.t)
    step = np.hypot(*np.diff(trip.track_xy, axis=0).T)
    speed = step / np.maximum(dt, 1e-9)

    n_nodes, n_arcs = len(trip.nodes_xy), len(trip.arcs)
    g = coo_matrix((np.ones(n_arcs), (trip.arcs[:, 0], trip.arcs[:, 1])), shape=(n_nodes, n_nodes))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(trip.arcs.ravel(), minlength=n_nodes)
    largest_strong = np.bincount(lab_s).max() / n_nodes

    L = make_point_labels(trip)
    j, ldist = L["route_pos"], L["dist"]
    R = trip.route
    cont = np.mean(trip.arcs[R[:-1], 1] == trip.arcs[R[1:], 0]) if len(R) > 1 else 1.0
    ext = trip.track_xy.max(0) - trip.track_xy.min(0)
    return {
        "partition": trip.partition, "trip_id": trip.trip_id, **sizes,
        "n_gps_raw": len(raw), "n_gps_clean": trip.n_points,
        "n_nonfinite": int((~np.isfinite(raw)).any(axis=1).sum()),
        "n_nonincreasing_t": int((np.diff(raw[:, 2]) <= 0).sum()),
        "duration_min": (trip.t[-1] - trip.t[0]) / 60,
        "frac_dt_1s": float(np.mean(dt == 1)), "median_dt": float(np.median(dt)), "max_dt_s": float(dt.max()),
        "n_gaps_gt_10s": int((dt > 10).sum()), "n_gaps_gt_60s": int((dt > 60).sum()),
        "max_step_m": float(step.max()), "p99_speed_kmh": float(np.percentile(speed, 99) * 3.6),
        "n_jumps_gt_200kmh": int((speed * 3.6 > 200).sum()),
        "lon_center": float(trip.track_ll[:, 0].mean()), "lat_center": float(trip.track_ll[:, 1].mean()),
        "extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000,
        "track_length_km": step.sum() / 1000,
        "n_nodes": n_nodes, "n_arcs": n_arcs,
        "isolated_nodes": int((deg == 0).sum()),
        "weak_components": n_weak, "strong_components": n_strong,
        "largest_strong_comp_frac": float(largest_strong),
        "twoway_arc_frac": float(np.isin(trip.arcs[:, 1] * (n_nodes + 1) + trip.arcs[:, 0],
                                         trip.arcs[:, 0] * (n_nodes + 1) + trip.arcs[:, 1]).mean()),
        "route_len_arcs": len(R), "route_unique_arcs": len(np.unique(R)),
        "route_length_km": trip.arc_len[R].sum() / 1000,
        "route_continuity": float(cont),
        "gps_to_route_median_m": float(np.median(ldist)), "gps_to_route_p95_m": float(np.percentile(ldist, 95)),
        "route_arcs_with_gps": len(np.unique(j)) / len(R),
        "frac_points_reliable_label": float(L["reliable"].mean()),
        "n_points_repaired_label": int((L["source"] == "repaired").sum()),
    }, ldist


def radius_study(trips, radii=(15, 25, 35, 50, 75, 100), stride=10):
    rows = []
    for trip in trips:
        L = make_point_labels(trip)
        labels = L["label"]
        idx = SegmentIndex(trip)
        pts = np.arange(0, trip.n_points, stride)
        pts = pts[L["reliable"][pts]]
        if len(pts) == 0:
            continue
        for r in radii:
            c = idx.query(trip.track_xy[pts], radius=r, max_k=50, fallback_k=0)
            hit = np.array([labels[p] in a for p, (a, _, _) in zip(pts, c)])
            ncand = np.array([len(a) for a, _, _ in c])
            rows.append({"trip_id": trip.trip_id, "radius": r, "miss_rate": 1 - hit.mean(),
                         "mean_cands": ncand.mean(), "n_points": len(pts)})
    return pd.DataFrame(rows)


def main():
    out = config.EDA_DIR
    stats, all_ld = [], []
    trips = {}
    for part in PARTS:
        for tid in list_trips(part):
            trip = load_trip(tid, part)
            trips[(part, tid)] = trip
            s, ld = trip_stats(trip)
            stats.append(s)
            all_ld.append(pd.DataFrame({"partition": part, "trip_id": tid, "gps_to_route_m": ld}))
            print(f"{part:12s} {tid}  pts={s['n_gps_clean']:6d}  arcs={s['n_arcs']:7d}  "
                  f"maxgap={s['max_dt_s']:8.0f}s  gps->route median={s['gps_to_route_median_m']:6.1f} m")
    inv = pd.DataFrame(stats)
    inv.to_csv(out / "inventory.csv", index=False)
    ld = pd.concat(all_ld)

    # ---- partition summary
    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), gps_points=("n_gps_clean", "sum"),
        median_points=("n_gps_clean", "median"), median_duration_min=("duration_min", "median"),
        median_arcs=("n_arcs", "median"), max_arcs=("n_arcs", "max"),
        trips_with_gap_gt_60s=("n_gaps_gt_60s", lambda s: int((s > 0).sum())),
        nonfinite_rows=("n_nonfinite", "sum"))
    summ.to_csv(out / "partition_summary.csv")
    print("\n", summ.to_string())

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for part, c in zip(PARTS, ["tab:blue", "tab:orange"]):
        sub = inv[inv.partition == part]
        ax[0, 0].hist(sub.n_gps_clean, bins=30, alpha=.6, color=c, label=part)
        ax[0, 1].hist(np.log10(sub.n_arcs), bins=30, alpha=.6, color=c, label=part)
        ax[0, 2].hist(sub.duration_min, bins=30, alpha=.6, color=c, label=part)
    ax[0, 0].set(title="GPS points per trip", xlabel="points")
    ax[0, 1].set(title="Road segments per trip (log10)", xlabel="log10(arcs)")
    ax[0, 2].set(title="Trip duration", xlabel="minutes")
    dts = np.concatenate([np.diff(t.t) for t in trips.values()])
    ax[1, 0].hist(np.clip(dts, 0, 10), bins=np.arange(0, 11, .5))
    ax[1, 0].set(title=f"Sampling interval (clipped at 10 s); {np.mean(dts == 1):.1%} exactly 1 s", xlabel="seconds")
    ax[1, 0].set_yscale("log")
    ax[1, 1].hist(np.clip(ld.gps_to_route_m, 0, 100), bins=50)
    ax[1, 1].set(title="GPS point -> aligned reference segment distance", xlabel="metres (clipped at 100)")
    ax[1, 2].scatter(inv.lon_center, inv.lat_center, c=(inv.partition == "train").map({True: "tab:blue", False: "tab:orange"}))
    ax[1, 2].set(title="Trip locations (blue=train, orange=student_test)", xlabel="lon", ylabel="lat")
    for a in ax[0]:
        a.legend()
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=120)
    plt.close(fig)

    # ---- radius study (train only, decides CANDIDATE_RADIUS_M)
    rs = radius_study([t for (p, _), t in trips.items() if p == "train"])
    rs.to_csv(out / "radius_study.csv", index=False)
    agg = rs.groupby("radius").apply(lambda g: pd.Series({
        "miss_rate": np.average(g.miss_rate, weights=g.n_points),
        "mean_cands": np.average(g.mean_cands, weights=g.n_points)}), include_groups=False)
    agg.to_csv(out / "radius_study_summary.csv")
    print("\nCandidate radius study (train):\n", agg.to_string())
    fig, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(agg.index, agg.miss_rate * 100, "o-", color="tab:red")
    a1.set(xlabel="search radius (m)", ylabel="candidate miss rate (%)", title="Choosing the candidate radius (train)")
    a2 = a1.twinx()
    a2.plot(agg.index, agg.mean_cands, "s--", color="tab:blue")
    a2.set_ylabel("mean candidates per point", color="tab:blue")
    fig.tight_layout()
    fig.savefig(out / "radius_study.png", dpi=120)
    plt.close(fig)

    # ---- automatic data-quality notes
    notes = ["# Data-quality notes (generated by scripts/01_eda.py)\n"]
    off = inv[inv.frac_points_reliable_label < 0.9]
    notes.append(f"* The .route file lists the arcs in driving order but does not say which GPS point belongs "
                 f"to which arc. Point labels are created with a monotone dynamic-programming alignment "
                 f"(src/pathsnap/labels.py). Overall {np.average(inv.frac_points_reliable_label, weights=inv.n_gps_clean):.1%} "
                 f"of GPS points get a reference segment within {config.LABEL_MAX_DIST_M:.0f} m.")
    notes.append(f"* {len(off)} trip(s) where > 10 % of GPS points have no reference segment within "
                 f"{config.LABEL_MAX_DIST_M:.0f} m: " + "; ".join(
                     f"{r.trip_id} (GPS track {r.track_length_km:.1f} km vs reference route {r.route_length_km:.1f} km)"
                     for r in off.itertuples()) + ". The reference route covers only part of the GPS track "
                 f"(e.g. repeated loops or driving that the route omits). Such points are excluded from training labels "
                 f"and metrics instead of being forced onto a wrong segment.")
    gaps = inv[inv.n_gaps_gt_60s > 0]
    notes.append(f"* {len(gaps)} trip(s) contain time gaps > 60 s (max {inv.max_dt_s.max():.0f} s): "
                 f"{', '.join(gaps.trip_id)}. The HMM starts a new segment across such gaps.")
    jumps = inv[inv.n_jumps_gt_200kmh > 0]
    notes.append(f"* {len(jumps)} trip(s) have implied speeds > 200 km/h between consecutive points "
                 f"(GPS jumps): {', '.join(jumps.trip_id)}.")
    lowrate = inv[inv.frac_dt_1s < 0.9]
    notes.append(f"* {len(lowrate)} trip(s) with < 90 % of steps at exactly 1 s: {', '.join(lowrate.trip_id)}.")
    notes.append(f"* Non-finite GPS rows across all trips: {int(inv.n_nonfinite.sum())}; "
                 f"non-increasing timestamps: {int(inv.n_nonincreasing_t.sum())}.")
    notes.append(f"* Road networks contain isolated nodes in {int((inv.isolated_nodes > 0).sum())} trips; "
                 f"median number of weakly connected components = {inv.weak_components.median():.0f}.")
    notes.append(f"* Reference routes are continuous (arc[i].to == arc[i+1].from) for "
                 f"{inv.route_continuity.mean():.2%} of consecutive pairs on average; "
                 f"min trip continuity = {inv.route_continuity.min():.2%}.")
    notes.append("* Trips are spread across many countries, each with its own local road network, "
                 "so all distances are computed in a per-trip local metric projection.")
    (out / "data_quality_notes.md").write_text("\n".join(notes) + "\n")
    print("\n".join(notes))

    # ---- static + folium maps for a few trips
    for key in [("train", "00000046"), ("train", "00000033"), ("student_test", "00000052")]:
        if key in trips:
            viz.static_trip_plot(trips[key], out / f"trip_{key[1]}.png")
            viz.folium_trip_map(trips[key], config.MAP_DIR / f"eda_{key[0]}_{key[1]}.html")


if __name__ == "__main__":
    main()
