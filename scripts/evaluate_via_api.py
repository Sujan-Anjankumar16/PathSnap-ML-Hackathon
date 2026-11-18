"""Run every trip through the running PathSnap API and save one results table.

Use it on evaluation day after starting the API, from any laptop that can reach it:
    python scripts/evaluate_via_api.py                         # API on this laptop
    python scripts/evaluate_via_api.py http://192.168.43.12:8000   # API on another laptop

It only calls the public endpoints (/health, /trips, /predict), exactly like the UI,
so ground truth never leaves the server. No extra packages are needed.
Output: outputs/results/api_run_<partition>.csv and a summary printed to the screen.
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "results"


def get(path):
    with urllib.request.urlopen(API + path, timeout=60) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    health = get("/health")
    trips = get("/trips")["trips"]
    print(f"Connected to {API}: model {health['model']}, partition {health['partition']}, {len(trips)} trips\n")

    rows = []
    for tid in trips:
        t0 = time.time()
        res = post("/predict", {"trip_id": tid})
        m = res["metrics"]
        row = {
            "trip_id": tid,
            "gps_points": res["n_gps_points"],
            "evaluated_points": m.get("n_evaluated_points"),
            "accuracy_hmm": m.get("point_accuracy_hmm"),
            "accuracy_classifier_only": m.get("point_accuracy_classifier_only"),
            "route_precision": m.get("route_precision"),
            "route_recall": m.get("route_recall"),
            "route_f1": m.get("route_f1"),
            "route_continuity": res["route_continuity"],
            "chain_breaks": res["chain_breaks"],
            "server_time_s": res["timing_s"]["total"],
            "round_trip_s": round(time.time() - t0, 3),
        }
        rows.append(row)
        acc = row["accuracy_hmm"]
        print(f"{tid}: accuracy {acc:.2%}  route F1 {row['route_f1']:.3f}  ({row['round_trip_s']} s)"
              if acc is not None else f"{tid}: no reference route on server")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"api_run_{health['partition']}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["accuracy_hmm"] is not None]
    if scored:
        n = sum(r["evaluated_points"] for r in scored)
        acc = sum(r["accuracy_hmm"] * r["evaluated_points"] for r in scored) / n
        clf = sum(r["accuracy_classifier_only"] * r["evaluated_points"] for r in scored) / n
        f1 = sum(r["route_f1"] for r in scored) / len(scored)
        print(f"\nOverall (point-weighted): HMM {acc:.2%}, classifier only {clf:.2%}; mean route F1 {f1:.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
