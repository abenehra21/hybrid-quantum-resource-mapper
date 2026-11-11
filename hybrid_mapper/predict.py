"""Phase 2: predict circuit-level resources (two-qubit gates, depth, measurement groups) from
problem-level features, so the Phase 1 rules can decide without building or transpiling a circuit.

Three predictors per target:
  analytic   closed-form count before routing (2*p*|E| for QAOA, reps*pairs for EfficientSU2);
             the baseline ML has to beat. No routing/SWAP overhead, no depth formula.
  power-law  log-linear ridge regression per (family, backend, entanglement) group:
             log y = a + b*log n + c*log reps + d*log analytic. Extrapolates to larger sizes.
  gbt        gradient-boosted trees on all features, log target. Strong inside the training
             range, but trees cannot extrapolate beyond it.
"""
import math
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

from hybrid_mapper.backends import get_backend, load_backends
from hybrid_mapper.dataset import DATA_PATH, LABELS
from hybrid_mapper.profile import TaskProfile
from hybrid_mapper.rules import decide, derive, load_rules

GROUP = ["family", "backend", "entanglement"]
NUMERIC = ["n_qubits", "reps", "n_edges", "n_pauli_terms", "analytic_2q"]
CATEGORIES = {"family": ["maxcut", "vqe", "random"], "entanglement": ["none", "reverse_linear", "circular", "full"]}
FAMILY_CLASS = {"maxcut": ("opt", "qaoa"), "vqe": ("chem", "vqe"), "random": ("samp", "samp")}
EXTRAPOLATION_CUTOFF = 40  # train on n_qubits <= cutoff, test above it


def _log_features(df: pd.DataFrame, target: str) -> np.ndarray:
    ref = df["n_pauli_terms"] if target == "measurement_groups" else df["analytic_2q"]
    return np.column_stack([np.log(df["n_qubits"]), np.log(df["reps"]), np.log1p(ref)])


class Analytic:
    name = "analytic"

    def __init__(self, target):
        self.target = target

    def fit(self, df, y):
        return self

    def predict(self, df):
        if self.target == "two_qubit_gates":
            return df["analytic_2q"].to_numpy(float)
        if self.target == "measurement_groups":
            return df["n_pauli_terms"].to_numpy(float)  # upper bound: no grouping
        return np.full(len(df), np.nan)  # no closed-form depth after routing


class PowerLaw:
    name = "power-law"

    def __init__(self, target):
        self.target, self.models = target, {}

    def fit(self, df, y):
        logy = np.log1p(y)
        self.fallback = Ridge(alpha=1e-3).fit(_log_features(df, self.target), logy)
        for key, idx in df.groupby(GROUP).groups.items():
            rows = df.loc[idx]
            if len(rows) >= 4:
                self.models[key] = Ridge(alpha=1e-3).fit(_log_features(rows, self.target), logy[df.index.get_indexer(idx)])
        return self

    def predict(self, df):
        out = np.empty(len(df))
        X = _log_features(df, self.target)
        for i, key in enumerate(df[GROUP].itertuples(index=False, name=None)):
            out[i] = self.models.get(key, self.fallback).predict(X[i : i + 1])[0]
        return np.expm1(out)


class GBT:
    name = "gbt"

    def __init__(self, target):
        self.target = target

    def _X(self, df):
        cats = {**CATEGORIES, "backend": sorted(load_backends())}
        coded = [df[c].map({v: i for i, v in enumerate(cats[c])}).to_numpy() for c in GROUP]
        return np.column_stack([*coded, df[NUMERIC].to_numpy(float)])

    def fit(self, df, y):
        self.model = HistGradientBoostingRegressor(categorical_features=[0, 1, 2], random_state=0)
        self.model.fit(self._X(df), np.log1p(y))
        return self

    def predict(self, df):
        return np.expm1(self.model.predict(self._X(df)))


MODELS = {m.name: m for m in (Analytic, PowerLaw, GBT)}


def load_dataset(path=DATA_PATH, include_timeouts: bool = False) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m hybrid_mapper dataset` first")
    df = pd.read_csv(path)
    return df if include_timeouts else df[~df.timed_out].reset_index(drop=True)


class ResourcePredictor:
    """One fitted model per label; predict() returns a DataFrame of predicted labels."""

    def __init__(self, model: str = "power-law"):
        self.model = model

    def fit(self, df: pd.DataFrame) -> "ResourcePredictor":
        self.fitted = {}
        for t in LABELS:
            rows = df[df[t].notna()].reset_index(drop=True)
            self.fitted[t] = MODELS[self.model](t).fit(rows, rows[t].to_numpy(float))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        return pd.DataFrame({t: np.clip(m.predict(df), 0, None) for t, m in self.fitted.items()})


def profile_from_features(row: dict, two_qubit_gates, depth, measurement_groups, task_id: str) -> TaskProfile:
    """A full TaskProfile from problem-level features plus (exact or predicted) circuit fields."""
    defaults = load_rules()["workload_defaults"]
    spec = get_backend(row["backend"])
    problem_class, algorithm = FAMILY_CLASS[row["family"]]
    return TaskProfile(
        task_id=task_id, problem_class=problem_class, algorithm=algorithm,
        precision_target=float(defaults["precision_target"]), deadline_min=float(defaults["deadline_min"]),
        n_qubits=int(row["n_qubits"]), depth=int(round(depth)) if not math.isnan(depth) else 0,
        two_qubit_gates=int(round(two_qubit_gates)), sum_abs_coeffs=float(row["sum_abs_coeffs"]),
        measurement_groups=int(round(measurement_groups)) if not math.isnan(measurement_groups) else 0,
        samples_needed=int(defaults["samples_needed"]) if algorithm in ("qaoa", "samp") else 0,
        target_qpu=row["backend"], eplg=float(spec["eplg"]), throughput_per_s=float(spec["throughput_per_s"]),
        queue_wait_min=float(defaults["queue_wait_min"]), shot_budget=float(defaults["shot_budget"]),
    )


# ---------------------------------------------------------------- evaluation

CIRCUIT = ["family", "entanglement", "n_qubits", "reps", "seed"]  # one circuit, compiled for each backend


def splits(df: pd.DataFrame, seed: int = 0) -> dict:
    """Held-out circuits go to test on every backend at once: the same circuit on the other backend shares
    its measurement groups and most of its gate count, so a row-wise split would leak labels into training."""
    groups = df[CIRCUIT].astype(str).agg("-".join, axis=1)
    tr, te = next(GroupShuffleSplit(test_size=0.25, random_state=seed).split(df, groups=groups))
    return {
        "interpolation (random 25% of circuits held out)": (df.iloc[tr], df.iloc[te]),
        f"extrapolation (train n<={EXTRAPOLATION_CUTOFF}, test n>{EXTRAPOLATION_CUTOFF})":
            (df[df.n_qubits <= EXTRAPOLATION_CUTOFF], df[df.n_qubits > EXTRAPOLATION_CUTOFF]),
    }


def mape(y, p) -> float:
    return float(np.mean(np.abs(p - y) / np.maximum(np.abs(y), 1)))


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (accuracy table, decision-agreement table)."""
    acc, agree = [], []
    for split, (train, test) in splits(df).items():
        test = test.reset_index(drop=True)
        exact = [decide(profile_from_features(r, r["two_qubit_gates"], r["depth"], r["measurement_groups"], "x"))
                 for r in test.to_dict("records")]
        for name in MODELS:
            model = ResourcePredictor(name).fit(train)
            t0 = time.perf_counter()
            pred = model.predict(test)
            for t in LABELS:
                mask = test[t].notna().to_numpy()
                y, p = test[t].to_numpy(float)[mask], pred[t].to_numpy()[mask]
                ok = not np.isnan(p).any()
                acc.append({"split": split, "target": t, "model": name, "n_test": int(mask.sum()),
                            "MAPE": mape(y, p) if ok else math.nan, "R2": r2_score(y, p) if ok else math.nan})
            decisions = []
            for r, gates, depth, groups in zip(test.to_dict("records"), pred["two_qubit_gates"], pred["depth"],
                                               pred["measurement_groups"]):
                decisions.append(decide(profile_from_features(r, gates, depth, groups, "x")))
            per_ms = (time.perf_counter() - t0) / len(test) * 1e3
            f_exact = np.array([d.derived["fidelity"] for d in exact])
            f_pred = np.array([d.derived["fidelity"] for d in decisions])
            agree.append({
                "split": split, "model": name, "n_test": len(test),
                "decision agreement": np.mean([a.decision == b.decision for a, b in zip(exact, decisions)]),
                "fired-rule agreement": np.mean([a.fired_rule == b.fired_rule for a, b in zip(exact, decisions)]),
                "median |log10 F error|": float(np.median(np.abs(np.log10(np.maximum(f_pred, 1e-300))
                                                                 - np.log10(np.maximum(f_exact, 1e-300))))),
                "predict+decide ms/task": per_ms,
                "exact transpile ms/task": test["transpile_s"].mean() * 1e3,
            })
    return pd.DataFrame(acc), pd.DataFrame(agree)


# ---------------------------------------------------------------- execution-outcome models

OUTCOME_TARGETS = ["projected_runtime_s", "energy_j_est", "rel_error"]


def outcome_models(log: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out power-law fits (mean absolute error) of hybrid runtime, energy and error from profile fields.

    Execution labels are expensive (each is a simulated hybrid run), so this is data-limited by design;
    the table reports n so it is read accordingly.
    """
    runs = log[log.strategy == "hybrid"].copy()
    runs["rel_error"] = runs["error"] / runs["exact"].abs()
    X = np.column_stack([np.log(runs.n_qubits), np.log1p(runs.two_qubit_gates), np.log1p(runs.depth),
                         (runs.algorithm == "vqe").astype(float)])
    rows = []
    for t in OUTCOME_TARGETS:
        y = runs[t].to_numpy(float)
        if len(y) < 4:
            continue
        logy, preds, base = np.log1p(y), [], []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            preds.append(np.expm1(Ridge(alpha=1.0).fit(X[keep], logy[keep]).predict(X[i : i + 1])[0]))
            base.append(y[keep].mean())
        rows.append({"target": t, "n_runs": len(y), "LOO MAE power-law": float(np.mean(np.abs(np.array(preds) - y))),
                     "LOO MAE mean-baseline": float(np.mean(np.abs(np.array(base) - y)))})
    return pd.DataFrame(rows)
