"""Plotting helpers: static matplotlib figures and interactive Folium maps."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Trip


def _arcs_near_track(trip: Trip, margin_m=150.0, max_arcs=6000):
    P = trip.track_xy
    lo, hi = P.min(0) - margin_m, P.max(0) + margin_m
    a, b = trip.arc_p0, trip.arc_p1
    near = np.all(np.maximum(a, b) >= lo, axis=1) & np.all(np.minimum(a, b) <= hi, axis=1)
    ids = np.flatnonzero(near)
    if len(ids) > max_arcs:                        # keep the ones closest to the track
        from scipy.spatial import cKDTree
        d, _ = cKDTree(P).query((a[ids] + b[ids]) / 2)
        ids = ids[np.argsort(d)[:max_arcs]]
    return ids


def static_trip_plot(trip: Trip, path: Path, pred_arcs=None, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    ids = _arcs_near_track(trip)
    for i in ids:
        ax.plot(*np.c_[trip.arc_p0[i], trip.arc_p1[i]], color="0.8", lw=0.6, zorder=1)
    if trip.route is not None:
        for i in trip.route:
            ax.plot(*np.c_[trip.arc_p0[i], trip.arc_p1[i]], color="tab:green", lw=3, zorder=2)
    if pred_arcs is not None:
        for i in pred_arcs:
            ax.plot(*np.c_[trip.arc_p0[i], trip.arc_p1[i]], color="tab:red", lw=1.5, zorder=3)
    ax.scatter(*trip.track_xy.T, s=3, color="tab:blue", zorder=4)
    ax.set_aspect("equal")
    ax.set(title=title or f"Trip {trip.trip_id} ({trip.partition}): grey=roads, green=reference route, "
                          f"blue=GPS{', red=predicted' if pred_arcs is not None else ''}",
           xlabel="x (m)", ylabel="y (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def folium_trip_map(trip: Trip, path: Path, pred_arcs=None, pred_point_arcs=None, show_reference=True):
    """Interactive map with layer control: road network, GPS points, reference route, predicted route."""
    try:
        import folium
    except ImportError:
        print(f"[viz] folium not installed - skipping {path.name} (pip install folium)")
        return None

    def ll(arc_ids):  # -> list of [[lat, lon], [lat, lon]] segments
        a = trip.nodes_ll[trip.arcs[arc_ids, 0]]
        b = trip.nodes_ll[trip.arcs[arc_ids, 1]]
        return [[[p[1], p[0]], [q[1], q[0]]] for p, q in zip(a, b)]

    center = trip.track_ll.mean(0)
    m = folium.Map(location=[center[1], center[0]], zoom_start=14, tiles="OpenStreetMap")

    roads = folium.FeatureGroup(name="Road network (nearby)", show=True)
    for seg in ll(_arcs_near_track(trip, max_arcs=3000)):
        folium.PolyLine(seg, color="#9e9e9e", weight=1, opacity=0.6).add_to(roads)
    roads.add_to(m)

    if show_reference and trip.route is not None:
        ref = folium.FeatureGroup(name="Reference route (ground truth)", show=True)
        for seg in ll(trip.route):
            folium.PolyLine(seg, color="#2e7d32", weight=6, opacity=0.7).add_to(ref)
        ref.add_to(m)

    if pred_arcs is not None:
        pr = folium.FeatureGroup(name="Predicted route", show=True)
        for seg in ll(np.asarray(pred_arcs)):
            folium.PolyLine(seg, color="#d32f2f", weight=3, opacity=0.9).add_to(pr)
        pr.add_to(m)

    gps = folium.FeatureGroup(name="Raw GPS observations", show=True)
    step = max(1, trip.n_points // 1500)
    for k in range(0, trip.n_points, step):
        lon, lat = trip.track_ll[k]
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.8,
                            tooltip=f"#{k}  t={trip.t[k]:.0f}s").add_to(gps)
    gps.add_to(m)

    s, e = trip.track_ll[0], trip.track_ll[-1]
    folium.Marker([s[1], s[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([e[1], e[0]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white;
                padding: 8px 12px; border: 1px solid #999; border-radius: 6px; font-size: 13px;">
      <b>Trip {tid}</b><br>
      <span style="color:#1565c0">&#9679;</span> Raw GPS<br>
      <span style="color:#9e9e9e">&#9473;</span> Road network<br>
      <span style="color:#2e7d32">&#9473;</span> Reference route<br>
      <span style="color:#d32f2f">&#9473;</span> Predicted route
    </div>""".format(tid=trip.trip_id)
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    lo, hi = trip.track_ll.min(0), trip.track_ll.max(0)
    m.fit_bounds([[lo[1], lo[0]], [hi[1], hi[0]]])
    m.save(str(path))
    return path
