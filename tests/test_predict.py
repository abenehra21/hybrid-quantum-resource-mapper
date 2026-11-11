import pandas as pd

from hybrid_mapper.dataset import problem_features
from hybrid_mapper.predict import (EXTRAPOLATION_CUTOFF, ResourcePredictor, load_dataset, mape,
                                   profile_from_features)
from hybrid_mapper.rules import decide


def test_power_law_beats_analytic_baseline_on_larger_unseen_sizes():
    df = load_dataset()
    train, test = df[df.n_qubits <= EXTRAPOLATION_CUTOFF], df[df.n_qubits > EXTRAPOLATION_CUTOFF]
    y = test.two_qubit_gates.to_numpy(float)
    errors = {m: mape(y, ResourcePredictor(m).fit(train).predict(test).two_qubit_gates.to_numpy())
              for m in ("analytic", "power-law")}
    assert errors["power-law"] < errors["analytic"]


def test_predicted_profile_decides_without_transpiling():
    row = problem_features("maxcut", 100, 3) | {"backend": "heron_r3"}
    pred = ResourcePredictor("power-law").fit(load_dataset()).predict(pd.DataFrame([row])).iloc[0]
    assert pred.two_qubit_gates > row["analytic_2q"]  # routing on heavy-hex adds SWAPs
    profile = profile_from_features(row, pred.two_qubit_gates, pred.depth, pred.measurement_groups, "t")
    assert decide(profile).decision in {"CLASSICAL", "HYBRID", "REDESIGN"}


def test_interpolation_split_holds_out_whole_circuits():
    from hybrid_mapper.predict import CIRCUIT, splits

    train, test = next(iter(splits(load_dataset()).values()))
    key = lambda d: set(d[CIRCUIT].itertuples(index=False, name=None))
    assert not key(train) & key(test)  # no circuit in both, even on a different backend
