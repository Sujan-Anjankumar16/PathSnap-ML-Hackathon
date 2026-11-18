"""A small, transparent Bayesian optimiser (Gaussian-process surrogate + Expected Improvement).

Why our own implementation: it is ~80 lines, has no extra dependency, and every step can be
explained in the viva:
  1. Evaluate a few random configurations (exploration).
  2. Fit a Gaussian process to (normalised hyper-parameters -> objective).
  3. Pick the next configuration that maximises Expected Improvement
        EI(x) = (mu(x) - f_best - xi) * Phi(z) + sigma(x) * phi(z),  z = (mu - f_best - xi) / sigma
     i.e. trade off a high predicted score (exploitation) against high uncertainty (exploration).
  4. Evaluate it, add to the data, repeat.
If Optuna is installed, scripts/04_bayes_opt.py can alternatively use Optuna's TPE sampler.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class Param:
    name: str
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def from_unit(self, u: float):
        if self.log:
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        return int(round(v)) if self.integer else float(v)


def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayes_optimize(objective, space: list[Param], n_trials=30, n_init=6, seed=42, log=print):
    """Maximises objective(params_dict) -> float. Returns list of trial dicts."""
    rng = np.random.default_rng(seed)
    d = len(space)
    U, Y, trials = [], [], []
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.full(d, 0.3), nu=2.5) + WhiteKernel(1e-4)
    for it in range(n_trials):
        if it < n_init:
            u = rng.random(d)
            how = "random"
        else:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed, n_restarts_optimizer=2)
            gp.fit(np.array(U), np.array(Y))
            cand = rng.random((4000, d))
            mu, sd = gp.predict(cand, return_std=True)
            ei = expected_improvement(mu, sd, max(Y))
            u = cand[int(np.argmax(ei))]
            how = "EI"
        params = {p.name: p.from_unit(x) for p, x in zip(space, u)}
        t0 = time.time()
        res = objective(params)
        val = res["objective"] if isinstance(res, dict) else float(res)
        U.append(u)
        Y.append(val)
        rec = {"trial": it, "proposal": how, **params,
               **(res if isinstance(res, dict) else {"objective": val}), "seconds": time.time() - t0,
               "best_so_far": max(Y)}
        trials.append(rec)
        log(f"trial {it:2d} [{how:6s}] objective={val:.5f} best={max(Y):.5f} {params}")
    return trials
