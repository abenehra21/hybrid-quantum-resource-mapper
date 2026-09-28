import pandas as pd

from hybrid_mapper.report import phase1


def _row(workload, strategy, decision, error, runtime, success):
    return dict(timestamp="t", workload=workload, strategy=strategy, decision=decision, fired_rule="R2",
                error=error, exact=10.0, projected_runtime_s=runtime, energy_j_est=runtime * 300,
                qpu_seconds_est=0.0 if strategy == "classical" else runtime, shots=0, qpu_calls=0, success=success)


def test_oracle_is_cheapest_successful_strategy():
    log = pd.DataFrame([
        _row("a", "classical", "CLASSICAL", 0.0, 1.0, True), _row("a", "hybrid", "CLASSICAL", 0.0, 50.0, True),
        _row("b", "classical", "HYBRID", 5.0, 1.0, False), _row("b", "hybrid", "HYBRID", 0.1, 50.0, True),
    ])
    text = phase1(log)
    assert "**Decision accuracy vs oracle:** 100% of 2 workloads" in text
