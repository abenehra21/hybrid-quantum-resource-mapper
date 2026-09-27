"""python -m hybrid_mapper {decide,profile,baselines}"""
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="hybrid_mapper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decide", help="decide for a named preset")
    d.add_argument("--preset", required=True, choices=sorted(PRESETS))
    p = sub.add_parser("profile", help="build, transpile and profile a workload")
    p.add_argument("--workload", required=True, choices=["maxcut", "vqe", "random"])
    p.add_argument("--size", type=int, required=True, help="qubits (vqe/random) or graph nodes (maxcut)")
    p.add_argument("--backend", default="heron_r3", choices=sorted(load_backends()))
    p.add_argument("--reps", type=int, default=1, help="QAOA p / ansatz reps / random-circuit depth x10")
    p.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("baselines", help="run small workloads under each strategy, append to logs/decisions.csv")
    b.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "decide":
        print_decision(PRESETS[args.preset])
    elif args.cmd == "profile":
        from hybrid_mapper.profiler import profile
        from hybrid_mapper.workloads import GENERATORS

        reps = args.reps * 10 if args.workload == "random" else args.reps
        print(profile(GENERATORS[args.workload](args.size, reps, seed=args.seed), args.backend).to_json())
    else:
        from hybrid_mapper.baselines import run

        run(quick=args.quick)
