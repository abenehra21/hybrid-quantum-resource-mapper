"""python -m hybrid_mapper {decide,profile,baselines,dataset,predict,tasks,allocate,demo,report}"""
import argparse

from hybrid_mapper.backends import load_backends
from hybrid_mapper.presets import PRESETS
from hybrid_mapper.rules import decide


def print_decision(profile) -> None:
    d = decide(profile)
    print(f"DECISION: {d.decision}  (fired {d.fired_rule})\n")
    print("Derived:")
    print(f"  algorithm kind     {d.derived['kind']}")
    print(f"  est. fidelity F    {d.derived['fidelity']:.4g}")
    print(f"  shots              {d.derived['shots']:.4g}")
    print(f"  QPU seconds        {d.derived['qpu_seconds']:.4g}")
    print(f"  statevector bytes  {d.derived['statevector_bytes']:.4g}\n")
    print("Trace:")
    for t in d.trace:
        print(f"  {t['rule']}  {'FIRE' if t['fired'] else 'pass'}  {t['reason']}")
    if d.split:
        print(f"\nSplit:\n  quantum:   {d.split['quantum']}\n  classical: {d.split['classical']}")
    print(f"\nProfile:\n{profile.to_json()}")


def add_workload_args(p) -> None:
    p.add_argument("--workload", required=True, choices=["maxcut", "vqe", "random"])
    p.add_argument("--size", type=int, required=True, help="qubits (vqe/random) or graph nodes (maxcut)")
    p.add_argument("--backend", default="heron_r3", choices=sorted(load_backends()))
    p.add_argument("--reps", type=int, help="QAOA p / ansatz reps / random-circuit depth (default 1; 10 for random)")
    p.add_argument("--entanglement", default="reverse_linear", choices=["reverse_linear", "circular", "full"],
                   help="VQE ansatz entanglement")
    p.add_argument("--seed", type=int, default=0)


def add_condition_args(p) -> None:
    p.add_argument("--precision", type=float, default=1e-2, help="expectation-value precision target")
    p.add_argument("--target-ratio", type=float, default=0.85, help="Max-Cut approximation-ratio target")
    p.add_argument("--deadline", type=float, default=480, help="minutes")
    p.add_argument("--queue", type=float, default=30, help="nominal QPU queue wait, minutes")
    p.add_argument("--budget", type=float, default=1e7, help="shot budget")
    p.add_argument("--samples", type=float, default=1e5, help="samples needed (sampling tasks)")


def conditions(args) -> dict:
    return {"precision_target": args.precision, "maxcut_ratio_target": args.target_ratio,
            "deadline_min": args.deadline, "queue_wait_min": args.queue, "shot_budget": args.budget,
            "samples_needed": args.samples}


def build_workload(args):
    from hybrid_mapper.dataset import build

    reps = args.reps or (10 if args.workload == "random" else 1)
    entanglement = args.entanglement if args.workload == "vqe" else "none"
    return (args.workload, args.size, reps, entanglement, args.seed), build(args.workload, args.size, reps,
                                                                              entanglement, args.seed)


def predict_cmd(args) -> None:
    import time

    import pandas as pd

    from hybrid_mapper.dataset import problem_features
    from hybrid_mapper.predict import ResourcePredictor, load_dataset, profile_from_features

    predictor = ResourcePredictor(args.model).fit(load_dataset())
    spec, workload = build_workload(args)
    t0 = time.perf_counter()
    row = problem_features(*spec, workload=workload) | {"backend": args.backend}
    pred = predictor.predict(pd.DataFrame([row])).iloc[0]
    profile = profile_from_features(row, pred.two_qubit_gates, pred.depth, pred.measurement_groups,
                                    f"{workload.name}_predicted")
    ms = (time.perf_counter() - t0) * 1e3
    print(f"PREDICTED with {args.model} model in {ms:.1f} ms (no transpile)\n")
    print_decision(profile)
    if args.compare:
        from hybrid_mapper.profiler import profile as exact_profile

        t0 = time.perf_counter()
        exact = exact_profile(workload, args.backend)
        ms = (time.perf_counter() - t0) * 1e3
        d_pred, d_exact = decide(profile), decide(exact)
        print(f"\nCOMPARE with exact transpile ({ms:.0f} ms):")
        for field in ("two_qubit_gates", "depth", "measurement_groups"):
            print(f"  {field:20s} predicted {getattr(profile, field):>8}   exact {getattr(exact, field):>8}")
        print(f"  {'decision':20s} predicted {d_pred.decision} ({d_pred.fired_rule})   "
              f"exact {d_exact.decision} ({d_exact.fired_rule})")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hybrid_mapper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="Phase 1: decide for a named preset")
    d.add_argument("--preset", required=True, choices=sorted(PRESETS))
    add_workload_args(sub.add_parser("profile", help="Phase 1: build, transpile and profile a workload"))
    b = sub.add_parser("baselines", help="Phase 1: run small workloads per strategy, append to logs/decisions.csv")
    b.add_argument("--quick", action="store_true")
    g = sub.add_parser("dataset", help="Phase 2: sweep workloads x backends, write data/resources.csv")
    g.add_argument("--quick", action="store_true")
    p = sub.add_parser("predict", help="Phase 2: predict resources from problem-level features and decide")
    add_workload_args(p)
    p.add_argument("--model", default="power-law", choices=["analytic", "power-law", "gbt"])
    p.add_argument("--compare", action="store_true", help="also transpile and show the exact values")
    sub.add_parser("tasks", help="Phase 3: pair workloads with operating conditions, label with the oracle")
    a = sub.add_parser("allocate", help="Phase 3: rules vs AI allocation for one task (--compare: ground truth)")
    add_workload_args(a)
    add_condition_args(a)
    a.add_argument("--compare", action="store_true", help="also transpile and ask the ground-truth oracle")
    m = sub.add_parser("demo", help="guided walkthrough of Phases 1-3")
    m.add_argument("--phase", type=int, choices=[1, 2, 3], help="run only one phase")
    sub.add_parser("report", help="write results/phase{1,2,3}_*.md")
    args = ap.parse_args(argv)

    if args.cmd == "decide":
        print_decision(PRESETS[args.preset])
    elif args.cmd == "profile":
        from hybrid_mapper.profiler import profile

        print(profile(build_workload(args)[1], args.backend).to_json())
    elif args.cmd == "baselines":
        from hybrid_mapper.baselines import run

        run(quick=args.quick)
    elif args.cmd == "dataset":
        from hybrid_mapper.dataset import generate

        generate(quick=args.quick)
    elif args.cmd == "predict":
        predict_cmd(args)
    elif args.cmd == "tasks":
        from hybrid_mapper.allocate import build_tasks

        build_tasks()
    elif args.cmd == "allocate":
        from hybrid_mapper.allocate import allocate_one

        spec, _ = build_workload(args)
        print(allocate_one(*spec, args.backend, conditions(args), args.compare))
    elif args.cmd == "demo":
        from hybrid_mapper.demo import run as demo

        demo(args.phase)
    else:
        from hybrid_mapper.report import run

        run()
