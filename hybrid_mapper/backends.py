"""QPU specs from config/backends.yaml, plus the transpile target each one maps to."""
from functools import cache
from pathlib import Path

import yaml
from qiskit.transpiler import CouplingMap

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@cache
def load_backends(path: Path = CONFIG_DIR / "backends.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def get_backend(name: str) -> dict:
    backends = load_backends()
    if name not in backends:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(backends)}")
    return backends[name]


def transpile_kwargs(name: str) -> dict:
    """Keyword args for qiskit.transpile that target this QPU."""
    target = get_backend(name)["transpile_target"]
    if "fake_backend" in target:
        from qiskit_ibm_runtime import fake_provider

        return {"backend": getattr(fake_provider, target["fake_backend"])()}
    rows, cols = target["grid"]  # topology proxy, see backends.yaml
    return {"coupling_map": CouplingMap.from_grid(rows, cols), "basis_gates": target["basis"]}
