# Evaluation-day runbook

Follow these steps in order. Laptop 1 runs the model, Laptop 2 shows the UI to the evaluator.

## Before the evaluator arrives (Laptop 1)

1. Put the evaluator trips in a folder, e.g. `~/Downloads/evaluator_test` (one sub-folder per trip).
2. Connect both laptops to the same network. A phone hotspot is the most reliable option.
3. Find Laptop 1's IP address:
   - macOS: `ipconfig getifaddr en0`
   - Windows: `ipconfig` and read "IPv4 Address"
4. Start the API and keep this window visible, because it shows every request:
   ```bash
   cd ML-HACKATHON
   source .venv/bin/activate
   PATHSNAP_EVAL_DATA=~/Downloads/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   If macOS asks whether to accept incoming connections, choose **Allow**.
5. Check it in a browser on Laptop 1: `http://localhost:8000/health` should show `"n_trips": 10`.

## Laptop 2 (UI)

1. In the `ui` folder: `python3 -m http.server 5500` (Windows: `python -m http.server 5500`).
2. Open `http://localhost:5500`.
3. In the API box type `http://<Laptop-1-IP>:8000`, click **Connect & load trips**.
4. Click a trip: the map shows raw GPS (blue) and the predicted route (red); the panel shows accuracy.

## Optional: all trips at once

From either laptop: `python scripts/evaluate_via_api.py http://<Laptop-1-IP>:8000`
saves `outputs/results/api_run_evaluator_test.csv` with every trip's accuracy.

## If something goes wrong

| Problem | Fix |
|---|---|
| UI says "Could not reach the API" | Same network? Correct IP? API window still running? Try the phone hotspot. |
| Browser shows nothing at `http://0.0.0.0:8000` | Normal. Use `localhost:8000/health` on Laptop 1, or the IP from Laptop 2. |
| `n_trips: 0` | `PATHSNAP_EVAL_DATA` points to the wrong folder. |
| "Address already in use" | An old API is still running: close that window or press Ctrl+C in it. |
| Map is grey | Laptop 2 needs internet for the map tiles. |

## What to say during the demo

- Prediction uses only `.track`, `.nodes`, `.arcs`; the `.route` file is read on the server only after prediction, to compute accuracy.
- Each click is logged on Laptop 1 with the client IP, trip, accuracy and run time.
- Classifier only vs classifier + HMM is shown for every trip.
