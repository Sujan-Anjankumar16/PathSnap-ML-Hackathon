"""Evaluation metrics shared by model comparison, HMM evaluation and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)


def candidate_metrics(y, proba, threshold=0.5):
    """Row-level (candidate) binary classification metrics."""
    pred = (proba >= threshold).astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y, pred, zero_division=0),
        "recall_pos": recall_score(y, pred, zero_division=0),
        "f1_pos": f1_score(y, pred, zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def top1_predictions(df: pd.DataFrame, score_col="score") -> pd.DataFrame:
    """For each GPS point pick the candidate with the highest score."""
    idx = df.groupby(["trip_id", "point_idx"])[score_col].idxmax()
    top = df.loc[idx, ["trip_id", "point_idx", "arc_id", "y"] + (["label_arc"] if "label_arc" in df else [])]
    return top.sort_values(["trip_id", "point_idx"]).reset_index(drop=True)


def top1_accuracy(df: pd.DataFrame, score_col="score") -> float:
    return float(top1_predictions(df, score_col).y.mean())


def candidate_miss_rate(df: pd.DataFrame) -> float:
    return float(1 - df.groupby(["trip_id", "point_idx"]).y.max().mean())


def route_continuity(point_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive matched points whose segments are the same or directly connected
    (arc_i.to == arc_{i+1}.from). 1.0 = a drivable sequence with no jumps."""
    a = np.asarray(point_arcs)
    if len(a) < 2:
        return 1.0
    same = a[1:] == a[:-1]
    conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
    return float(np.mean(same | conn))


def route_overlap(pred_arcs, ref_arcs, arc_len, arcs=None):
    """Sequence-level, length-weighted comparison of the set of driven segments.

    precision = length of predicted segments that are in the reference / predicted length
    recall    = length of reference segments that were predicted / reference length
    A two-way road is two arcs; we count a match only for the correct direction.
    """
    P, R = np.unique(pred_arcs), np.unique(ref_arcs)
    inter = np.intersect1d(P, R)
    lp, lr, li = arc_len[P].sum(), arc_len[R].sum(), arc_len[inter].sum()
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
