"""Phase 3: AI task allocation. ML predictions replace or augment the fixed resource-mapping decision.

Allocators compared on the same held-out tasks (the proposal's evaluation table, plus ablations):
  Classical-only, Quantum-only     single-platform baselines
  Rules (exact transpile)          Phase 1 rules on exact circuit fields, R3 never firing (low_entanglement
                                   unknown), as in the original proposal
  Rules + R3 entanglement check    the same, with R3 fed the circuit's entanglement bound (profiler.py)
  Rules + Phase 2 predictions      augment: the same rules fed predicted fields (no transpile)
  Rules, tuned thresholds          augment: rules.yaml thresholds fitted to minimise training regret
  AI: decision tree (depth 4)      replace: an interpretable learned policy, printable as if-then rules
  AI: gradient boosting            replace: the strongest learned policy
  Oracle                           ground truth from environment.py (the label being learned)

The learned allocators see only what is known before execution: problem-level features, operating
conditions, and Phase 2 predicted circuit fields. The entanglement bound, realised queue and readout
effects reach them only through the oracle labels they are trained on.
"""
import math
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

from hybrid_mapper.dataset import DATA_PATH, LABELS, build
from hybrid_mapper.environment import ACTIONS, evaluate_action, flatten, load_environment, oracle
from hybrid_mapper.profiler import workload_entanglement_bound
from hybrid_mapper.predict import ResourcePredictor, load_dataset, profile_from_features
from hybrid_mapper.rules import decide, derive, load_rules

TASKS_PATH = DATA_PATH.parent / "tasks.csv"
CONDITIONS = ["precision_target", "maxcut_ratio_target", "deadline_min", "queue_wait_min", "shot_budget",
              "samples_needed"]
ONE_HOT = {"family": ["maxcut", "vqe", "random"], "backend": ["heron_r3", "nighthawk_r2"],
           "entanglement": ["none", "reverse_linear", "circular", "full"]}
EXTRAPOLATION_CUTOFF = 40


# ---------------------------------------------------------------- task generation

def workload_id(r: dict) -> str:
    return f"{r['family']}-{r['entanglement']}-n{r['n_qubits']}-r{r['reps']}-s{r['seed']}-{r['backend']}"


def build_tasks(path: Path = TASKS_PATH) -> pd.DataFrame:
    """Pair every transpiled workload with sampled operating conditions and label it with the oracle."""
    env = load_environment()
    sc = env["scenarios"]
    rng = np.random.default_rng(sc["seed"])
    res = load_dataset()
    bounds = {}
    for k in res[["family", "n_qubits", "reps", "entanglement", "seed"]].drop_duplicates().itertuples(index=False):
        bounds[tuple(k)] = workload_entanglement_bound(build(k.family, k.n_qubits, k.reps, k.entanglement, k.seed))
    rows = []
    for r in res.to_dict("records"):
        key = (r["family"], r["n_qubits"], r["reps"], r["entanglement"], r["seed"])
        for _ in range(sc["per_workload"]):
            cond = {c: rng.choice(sc[c]).item() for c in CONDITIONS}
            cond["queue_real_min"] = cond["queue_wait_min"] * rng.lognormal(0, env["quantum"]["queue_lognormal_sigma"])
            t = r | cond | {"workload_id": workload_id(r), "entanglement_bound": bounds[key]}
            rows.append(t | flatten(oracle(t, env)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"wrote {len(df)} tasks to {path.parent.name}/{path.name}; oracle labels: "
          + ", ".join(f"{k} {v}" for k, v in df.oracle_label.value_counts().items()))
    return df


def load_tasks(path: Path = TASKS_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m hybrid_mapper tasks` first")
    return pd.read_csv(path)


# ---------------------------------------------------------------- profiles and features

def task_profile(t: dict, gates, depth, groups):
    """Phase 1 TaskProfile for a task, with its operating conditions."""
    p = profile_from_features(t, gates, depth, groups, t.get("workload_id", "task"))
    return replace(p, precision_target=float(t["precision_target"]), deadline_min=float(t["deadline_min"]),
                   queue_wait_min=float(t["queue_wait_min"]), shot_budget=float(t["shot_budget"]),
                   samples_needed=int(t["samples_needed"]) if p.samples_needed else 0)


def profiles(tasks: pd.DataFrame, fields: pd.DataFrame) -> list:
    return [task_profile(t, f["two_qubit_gates"], f["depth"], f["measurement_groups"])
            for t, f in zip(tasks.to_dict("records"), fields.to_dict("records"))]


def features(tasks: pd.DataFrame, pred: pd.DataFrame, profs: list) -> pd.DataFrame:
    """What a learned allocator may see: problem-level inputs, conditions, Phase 2 predictions."""
    tasks, pred = tasks.reset_index(drop=True), pred.reset_index(drop=True)
    derived = pd.DataFrame([derive(p) for p in profs])
    X = pd.DataFrame({
        "n_qubits": tasks.n_qubits, "reps": tasks.reps, "n_edges": tasks.n_edges,
        "n_pauli_terms": tasks.n_pauli_terms, "sum_abs_coeffs": tasks.sum_abs_coeffs,
        "analytic_2q": tasks.analytic_2q,
        "pred_two_qubit_gates": pred.two_qubit_gates, "pred_depth": pred.depth,
        "pred_log10_F": np.log10(np.maximum(derived.fidelity, 1e-300)),
        "pred_log10_shots": np.log10(np.maximum(derived.shots, 1)),
        "pred_qpu_min_over_deadline": (tasks.queue_wait_min + derived.qpu_seconds / 60) / tasks.deadline_min,
        # domain feature: error-free samples the shot budget buys at the predicted fidelity (log10)
        "pred_log10_good_samples_affordable": np.log10(tasks.shot_budget)
                                              + np.log10(np.maximum(derived.fidelity, 1e-300)),
        "log10_precision_target": np.log10(tasks.precision_target),
        "maxcut_ratio_target": tasks.maxcut_ratio_target,
        "deadline_min": tasks.deadline_min, "queue_wait_min": tasks.queue_wait_min,
        "log10_shot_budget": np.log10(tasks.shot_budget), "log10_samples_needed": np.log10(tasks.samples_needed),
    })
    for col, values in ONE_HOT.items():
        for v in values:
            X[f"{col}={v}"] = (tasks[col] == v).astype(float)
    return X


# ---------------------------------------------------------------- allocators

def tune_rules(profs: list, tasks: pd.DataFrame, penalty: float) -> dict:
    """Grid-search the rules.yaml thresholds that minimise mean regret on the training tasks."""
    base = load_rules()
    records = tasks.to_dict("records")
    regret = {a: np.array([evaluate_action(a, t, penalty)["regret"] for t in records]) for a in ACTIONS}
    best = (math.inf, base)
    for mem, f_exp, f_samp in product([16, 64, 256, 512, 1024], [0.003, 0.01, 0.05, 0.2],
                                      [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]):
        cfg = base | {"statevector_memory_gib": mem,
                      "fidelity_min": base["fidelity_min"] | {"expect": f_exp, "sample": f_samp}}
        acts = [decide(p, cfg=cfg).decision for p in profs]
        r = np.mean([regret[a][i] for i, a in enumerate(acts)])
        if r < best[0]:
            best = (r, cfg)
    return best[1]


def mistake_cost(tasks: pd.DataFrame, penalty: float) -> np.ndarray:
    """Per task, the regret of its most expensive wrong action, in units of the failure penalty."""
    records = tasks.to_dict("records")
    return np.array([max(evaluate_action(a, t, penalty)["regret"] for a in ACTIONS if a != t["oracle_label"])
                     for t in records]) / penalty


def fit_learned(X: pd.DataFrame, y: pd.Series, weight: np.ndarray | None = None) -> tuple:
    """The tree is fit for readability (balanced classes); boosting is cost-sensitive (Elkan 2001):
    tasks whose mistakes are expensive weigh more."""
    w = None if weight is None else 1 + np.minimum(weight, 50)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=0)
    gbt = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=0)
    return tree.fit(X, y), gbt.fit(X, y, sample_weight=w)


def metrics(actions, tasks: pd.DataFrame, penalty: float) -> dict:
    actions = np.asarray(actions)
    labels = tasks.oracle_label.to_numpy()
    ev = pd.DataFrame([evaluate_action(a, t, penalty) for a, t in zip(actions, tasks.to_dict("records"))])
    executed = ev[ev.executed]
    return {
        "Decision accuracy (vs oracle)": float(np.mean(actions == labels)),
        "Macro-F1": f1_score(labels, actions, labels=ACTIONS, average="macro", zero_division=0),
        "Success rate": ev.success.mean(),
        "Accuracy (target met when run)": executed.success.mean() if len(executed) else math.nan,
        "Mean regret (cost units)": ev.regret.mean(),
        "Runtime (s, mean)": ev.time_s.mean(),
        "Energy (kJ, mean)": ev.energy_j.mean() / 1e3,
        "QPU time (s, mean)": ev.qpu_s.mean(),
        "Number of shots (mean)": ev.shots.mean(),
        "Unnecessary QPU use": ev.unnecessary_qpu.mean(),
        "Communication overhead (QPU jobs per task)": float(np.mean(actions == "HYBRID")),
    }


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit every allocator on train (Phase 2 predictor included), score all of them on test."""
    penalty = load_environment()["scenarios"]["failure_penalty_s"]
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    predictor = ResourcePredictor("power-law").fit(train.drop_duplicates("workload_id"))
    pred_tr, pred_te = predictor.predict(train), predictor.predict(test)
    prof_tr, prof_te = profiles(train, pred_tr), profiles(test, pred_te)
    X_tr, X_te = features(train, pred_tr, prof_tr), features(test, pred_te, prof_te)
    tuned = tune_rules(prof_tr, train, penalty)
    tree, gbt = fit_learned(X_tr, train.oracle_label, mistake_cost(train, penalty))
    exact_te = profiles(test, test[LABELS])
    max_ebits = load_rules()["low_entanglement_max_ebits"]
    actions = {
        "Classical-only": ["CLASSICAL"] * len(test),
        "Quantum-only": ["HYBRID"] * len(test),
        "Rules (exact transpile)": [decide(p).decision for p in exact_te],
        "Rules + R3 entanglement check": [decide(replace(p, low_entanglement=bool(b <= max_ebits))).decision
                                          for p, b in zip(exact_te, test.entanglement_bound)],
        "Rules + Phase 2 predictions": [decide(p).decision for p in prof_te],
        "Rules, tuned thresholds": [decide(p, cfg=tuned).decision for p in prof_te],
        "AI: decision tree": list(tree.predict(X_te)),
        "AI: gradient boosting": list(gbt.predict(X_te)),
        "Oracle": list(test.oracle_label),
    }
    return {"actions": actions, "metrics": {k: metrics(a, test, penalty) for k, a in actions.items()},
            "tuned": tuned, "tree": tree, "gbt": gbt, "X_test": X_te, "test": test}


def splits(tasks: pd.DataFrame, seed: int = 0) -> dict:
    tr, te = next(GroupShuffleSplit(test_size=0.3, random_state=seed).split(tasks, groups=tasks.workload_id))
    return {"random 30% of workloads held out": (tasks.iloc[tr], tasks.iloc[te]),
            f"extrapolation (train n<={EXTRAPOLATION_CUTOFF}, test n>{EXTRAPOLATION_CUTOFF})":
                (tasks[tasks.n_qubits <= EXTRAPOLATION_CUTOFF], tasks[tasks.n_qubits > EXTRAPOLATION_CUTOFF])}


def stability(tasks: pd.DataFrame, seeds=range(5)) -> pd.DataFrame:
    """Decision accuracy and regret over several random workload splits (mean and standard deviation),
    so a headline number is not one lucky split."""
    rows = []
    for seed in seeds:
        train, test = next(iter(splits(tasks, seed).values()))
        for name, m in evaluate_split(train, test)["metrics"].items():
            rows.append({"allocator": name, "accuracy": m["Decision accuracy (vs oracle)"],
                         "regret": m["Mean regret (cost units)"]})
    agg = pd.DataFrame(rows).groupby("allocator", sort=False).agg(["mean", "std"])
    return pd.DataFrame({"Allocator": agg.index,
                         "Decision accuracy": [f"{m:.3f} ± {s:.3f}" for m, s in agg["accuracy"].to_numpy()],
                         "Mean regret": [f"{m:,.0f} ± {s:,.0f}" for m, s in agg["regret"].to_numpy()]})


def feedback_curve(tasks: pd.DataFrame, fractions=(0.0, 0.05, 0.1, 0.2, 0.3), seed: int = 0) -> pd.DataFrame:
    """Function 6 in miniature: start from the extrapolation split, move a fraction of the large-size
    workloads (all their tasks) into training as "execution feedback", and score on the rest."""
    small = tasks[tasks.n_qubits <= EXTRAPOLATION_CUTOFF]
    large = tasks[tasks.n_qubits > EXTRAPOLATION_CUTOFF]
    ids = large.workload_id.unique()
    rng = np.random.default_rng(seed)
    pool = rng.permutation(ids)
    holdout = set(pool[int(0.3 * len(pool)):])  # fixed test set: 70% of large workloads never fed back
    test = large[large.workload_id.isin(holdout)]
    rows = []
    for f in fractions:
        fed = set(pool[: int(f * len(ids))]) - holdout
        r = evaluate_split(pd.concat([small, large[large.workload_id.isin(fed)]]), test)
        for name in ("Classical-only", "Rules (exact transpile)", "Rules + R3 entanglement check",
                     "AI: decision tree", "AI: gradient boosting"):
            m = r["metrics"][name]
            rows.append({"feedback (share of large workloads)": f, "allocator": name,
                         "accuracy": m["Decision accuracy (vs oracle)"], "regret": m["Mean regret (cost units)"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- explanations

def mechanism(t: dict, action: str) -> str:
    """Why an allocator's action differs from the oracle, in one phrase."""
    label = t["oracle_label"]
    if label == "CLASSICAL":
        method = t["classical_method"]
        return {"mps": "weakly entangled: exact MPS simulation", "heuristic": "classical Max-Cut heuristic is better",
                "statevector": "statevector fits on an HPC node"}.get(method, method)
    if action == "CLASSICAL":  # oracle HYBRID or REDESIGN: the classical route itself fails
        return f"classical fails: {t['classical_reason']}"
    reason = str(t["hybrid_reason"])
    if label == "HYBRID":
        return "low-F sampling rescued by extra shots"
    for key, phrase in [("QAOA ratio", "noisy QAOA misses ratio target"), ("budget", "shots exceed budget"),
                        ("deadline", "queue + QPU time misses deadline"), ("floor", "fidelity below mitigation floor"),
                        ("qubits", "does not fit the QPU")]:
        if key in reason:
            return phrase
    return reason


def disagreements(result: dict, allocator: str) -> pd.DataFrame:
    test, acts = result["test"], result["actions"][allocator]
    rows = [{"allocator says": a, "oracle": t["oracle_label"], "why": mechanism(t, a)}
            for a, t in zip(acts, test.to_dict("records")) if a != t["oracle_label"]]
    if not rows:
        return pd.DataFrame(columns=["allocator says", "oracle", "why", "tasks"])
    return (pd.DataFrame(rows).value_counts().rename("tasks").reset_index()
            .sort_values("tasks", ascending=False))


def tree_path(tree, X_row: pd.DataFrame) -> list[str]:
    """The if-then conditions a single task passes through in the learned tree."""
    t, names = tree.tree_, list(X_row.columns)
    leaf = tree.apply(X_row)[0]
    lines = []
    for node in tree.decision_path(X_row).indices:
        if node == leaf:
            continue
        f, thr, val = t.feature[node], t.threshold[node], X_row.iloc[0, t.feature[node]]
        lines.append(f"{names[f]} = {val:.3g} {'<=' if val <= thr else '>'} {thr:.3g}")
    return lines


def confusion(result: dict, allocator: str) -> pd.DataFrame:
    m = confusion_matrix(result["test"].oracle_label, result["actions"][allocator], labels=ACTIONS)
    return pd.DataFrame(m, index=[f"oracle {a}" for a in ACTIONS], columns=[f"says {a}" for a in ACTIONS])


def importance(result: dict, top: int = 10) -> pd.DataFrame:
    imp = permutation_importance(result["gbt"], result["X_test"], result["test"].oracle_label,
                                 n_repeats=5, random_state=0, scoring="f1_macro")
    order = np.argsort(imp.importances_mean)[::-1][:top]
    return pd.DataFrame({"feature": result["X_test"].columns[order],
                         "macro-F1 drop when shuffled": imp.importances_mean[order]})


# ---------------------------------------------------------------- one task, end to end

def allocate_one(family: str, size: int, reps: int, entanglement: str, seed: int, backend: str,
                 conditions: dict, compare: bool = False) -> str:
    """Rules vs AI for one task (trained on all tasks); with compare, also transpile and ask the oracle."""
    from hybrid_mapper.dataset import label, problem_features

    tasks = load_tasks()
    w = build(family, size, reps, entanglement, seed)
    t = problem_features(family, size, reps, entanglement, seed, workload=w) | {"backend": backend} | conditions
    t["workload_id"] = workload_id(t)
    one = pd.DataFrame([t])
    predictor = ResourcePredictor("power-law").fit(load_dataset())
    pred_all, pred_one = predictor.predict(tasks), predictor.predict(one)
    X_all = features(tasks, pred_all, profiles(tasks, pred_all))
    prof_one = profiles(one, pred_one)
    X_one = features(one, pred_one, prof_one)
    penalty = load_environment()["scenarios"]["failure_penalty_s"]
    tree, gbt = fit_learned(X_all, tasks.oracle_label, mistake_cost(tasks, penalty))
    rules = decide(prof_one[0])
    bound = workload_entanglement_bound(w)
    rules_r3 = decide(replace(prof_one[0], low_entanglement=bound <= load_rules()["low_entanglement_max_ebits"]))
    proba = dict(zip(gbt.classes_, gbt.predict_proba(X_one)[0]))
    p = pred_one.iloc[0]
    lines = [f"TASK  {w.name} on {backend}  |  " + ", ".join(f"{k}={v:g}" for k, v in conditions.items()),
             f"Phase 2 prediction: {p.two_qubit_gates:.0f} two-qubit gates, depth {p.depth:.0f}, "
             f"F={rules.derived['fidelity']:.3g}, shots {rules.derived['shots']:.3g}", "",
             f"Phase 1 rules (on predicted fields):  {rules.decision} via {rules.fired_rule}: {rules.trace[-1]['reason']}",
             f"  + R3 entanglement check ({bound} ebits): {rules_r3.decision} via {rules_r3.fired_rule}",
             f"Phase 3 AI (gradient boosting):       {gbt.predict(X_one)[0]}  "
             + "  ".join(f"P({k})={v:.2f}" for k, v in sorted(proba.items(), key=lambda kv: -kv[1])),
             f"Phase 3 AI (decision tree):           {tree.predict(X_one)[0]}, because:"]
    lines += [f"    {c}" for c in tree_path(tree, X_one)]
    if compare:
        exact = label(w, backend)
        truth = t | exact | {"entanglement_bound": bound,
                             "queue_real_min": conditions["queue_wait_min"]}  # nominal queue for the demo
        o = oracle(truth)
        lines += ["", f"GROUND TRUTH (exact transpile: {exact['two_qubit_gates']} two-qubit gates; "
                      f"entanglement bound {truth['entanglement_bound']} ebits; nominal queue)",
                  f"  classical: {'feasible' if o['classical']['feasible'] else 'infeasible'}  "
                  f"({o['classical']['reason']})",
                  f"  hybrid:    {'feasible' if o['hybrid']['feasible'] else 'infeasible'}  ({o['hybrid']['reason']})",
                  f"  oracle:    {o['label']}"]
    return "\n".join(lines)
