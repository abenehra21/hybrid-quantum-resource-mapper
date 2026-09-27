"""F3: fixed classical/quantum work split per algorithm, attached to HYBRID decisions."""

SPLITS = {
    "vqe": {
        "quantum": "Prepare parameterised ansatz, estimate Hamiltonian expectation values",
        "classical": "Optimise ansatz parameters; group Pauli terms; error mitigation post-processing",
    },
    "dyn": {
        "quantum": "Run Trotterised time-evolution circuits, estimate observables per time step",
        "classical": "Choose Trotter steps/time grid; extrapolate and mitigate observable estimates",
    },
    "qaoa": {
        "quantum": "Sample bitstrings from the QAOA state at the current angles",
        "classical": "Optimise (gamma, beta); evaluate cut values; keep the best sampled solution",
    },
    "sqd": {
        "quantum": "Sample computational-basis configurations from the ansatz",
        "classical": "Configuration recovery, subspace projection and diagonalisation (the heavy lifting)",
    },
    "samp": {
        "quantum": "Sample bitstrings from the target circuit",
        "classical": "Aggregate samples into the estimator/statistic of interest",
    },
}
