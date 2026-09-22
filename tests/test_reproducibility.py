"""Reproducibility of the synthetic ledger.

The generator is seeded, so the same code must produce a byte-identical ledger on
every run. That guarantee is easy to lose in a way that no ordinary unit test can
see: sampling from a hash-ordered collection - ``rng.choice(list(some_set))`` -
makes the draw depend on ``PYTHONHASHSEED``, and every process gets a different
one. Two consecutive pipeline runs then disagree on the vendor count, the alert
count and the model threshold while every test still passes.

An in-process test cannot catch that, because both runs share one hash seed. The
test below therefore executes the generator in two subprocesses with deliberately
different hash seeds and compares the resulting ledger hashes.

This is not a hypothetical: it is the exact defect that was present here, and it
survived the whole test suite until two pipeline runs were diffed by hand.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Deliberately small, so two subprocesses stay cheap. Large enough that every
#: injection routine has candidates to work with.
N_TRANSACTIONS = 2_000
N_VENDORS = 120
N_EMPLOYEES = 40

#: Runs the generator in a fresh interpreter and prints one hash per artefact.
_GENERATOR_SCRIPT = """
import hashlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, {root!r})

from src.data_generator import (
    GeneratorConfig,
    generate_employees,
    generate_ledger,
    generate_vendors,
    inject_anomalies,
)


def digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


config = GeneratorConfig(
    n_transactions={n_transactions},
    n_vendors={n_vendors},
    n_employees={n_employees},
)
rng = np.random.default_rng(config.seed)

vendors = generate_vendors(config, rng)
employees = generate_employees(config, rng)
ledger = generate_ledger(config, vendors, employees, rng)

print("vendors", digest(vendors))
print("clean_ledger", digest(ledger))

ledger = inject_anomalies(ledger, vendors, config, rng)
print("anomalous_ledger", digest(ledger))
print("vendor_count", ledger["vendor_id"].nunique())
"""


def _run_generator(hash_seed: str) -> dict[str, str]:
    """Run the generator in a fresh interpreter and parse its printed hashes."""
    import os

    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed
    # Keep the subprocess from picking up the project's own conftest side effects.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _GENERATOR_SCRIPT.format(
                root=str(PROJECT_ROOT),
                n_transactions=N_TRANSACTIONS,
                n_vendors=N_VENDORS,
                n_employees=N_EMPLOYEES,
            ),
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=environment,
        timeout=300,
    )

    if result.returncode != 0:
        pytest.fail(
            f"Generator subprocess failed with PYTHONHASHSEED={hash_seed}.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            parsed[parts[0]] = parts[1]
    return parsed


@pytest.fixture(scope="module")
def hash_seed_runs() -> tuple[dict[str, str], dict[str, str]]:
    """Run the generator twice under two different hash seeds."""
    return _run_generator("1"), _run_generator("99991")


class TestGeneratorReproducibility:
    """Two processes, one seed, one ledger."""

    def test_vendor_master_is_identical_across_processes(
        self, hash_seed_runs: tuple[dict[str, str], dict[str, str]]
    ) -> None:
        first, second = hash_seed_runs
        assert first["vendors"] == second["vendors"]

    def test_clean_ledger_is_identical_across_processes(
        self, hash_seed_runs: tuple[dict[str, str], dict[str, str]]
    ) -> None:
        first, second = hash_seed_runs
        assert first["clean_ledger"] == second["clean_ledger"]

    def test_anomaly_injection_is_identical_across_processes(
        self, hash_seed_runs: tuple[dict[str, str], dict[str, str]]
    ) -> None:
        """The regression guard.

        Anomaly injection used to sample its vendor pool from a ``set``, so the
        injected rows - and therefore the vendor count and every downstream
        metric - varied with ``PYTHONHASHSEED``.
        """
        first, second = hash_seed_runs
        assert first["anomalous_ledger"] == second["anomalous_ledger"]

    def test_the_population_reaches_the_same_size(
        self, hash_seed_runs: tuple[dict[str, str], dict[str, str]]
    ) -> None:
        first, second = hash_seed_runs
        assert first["vendor_count"] == second["vendor_count"]
