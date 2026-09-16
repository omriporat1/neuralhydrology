"""Authenticated, source-bound access to one run/period/epoch's NeuralHydrology
evaluation pickle.

RD1-C4-D1 Correction Pass A, finding A3: the fixed-support evaluator used to
accept an externally supplied ``period_results`` *mapping* so a caller that
needs per-basin outcomes could load the ~84.5 MB ``validation_results.p``
once rather than once per basin. That seam was forgeable: any plain ``dict``
satisfied it, so the simulations the evaluator paired against package
observations were not provably the simulations of the run, period and epoch
the caller claimed. The only guard was that *some* results file existed at
the expected path -- it was never proven that the supplied mapping came from
that file.

This module replaces that seam with a typed result that cannot be
fabricated as a plain mapping:

* :class:`AuthenticatedPeriodResults` can only be produced by
  :func:`load_authenticated_period_results`, which is the sole holder of the
  construction token. Constructing one directly -- with a dict, with
  fabricated hashes, or by copying the dataclass -- raises
  :class:`AuthenticatedPeriodResultsError`.
* Every instance carries the authoritative source facts: the run directory,
  period and epoch it was requested for, the exact ``*_results.p`` path that
  was opened, that file's size, and the SHA-256 of its actual bytes.
* The digest is computed **once**, from the same path that is then
  unpickled, at load time. It is never recomputed per basin: re-hashing an
  84.5 MB pickle once per basin would cost ~34 GB of reads per 400-basin
  trial, which is why the object -- not the mapping -- is what gets passed
  around.

A consumer that is handed one of these can therefore state, rather than
assume, which pickle its simulations came from, and can refuse an object
that was authenticated for a different run, period or epoch
(:meth:`AuthenticatedPeriodResults.require_bound_to`).

This module deliberately contains no scientific math and no evaluation
policy. It authenticates a source; what the values mean is the caller's
business.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from .nh_seed_evaluation import load_period_results, period_results_path

__all__ = [
    "AuthenticatedPeriodResultsError",
    "AuthenticatedPeriodResults",
    "load_authenticated_period_results",
    "fixture_only_period_results",
]


class AuthenticatedPeriodResultsError(ValueError):
    """Raised when a period-results source cannot be authenticated, or when
    an authenticated result is used against a run/period/epoch it was not
    authenticated for."""


#: Module-private construction token. Only :func:`load_authenticated_period_results`
#: holds it, so an :class:`AuthenticatedPeriodResults` can only come from an
#: actual authenticated load of an actual file.
_CONSTRUCTION_TOKEN = object()

#: Sentinels stamped onto a :func:`fixture_only_period_results` object so a
#: synthetic test fixture can never be reported as a hashed, file-backed
#: source. Neither is a valid path or a valid SHA-256.
FIXTURE_ONLY_RESULTS_PATH = "<fixture-only: not loaded from a file>"
FIXTURE_ONLY_RESULTS_SHA256 = "<fixture-only: no bytes were hashed>"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AuthenticatedPeriodResults:
    """One run/period/epoch's evaluation results, bound to the actual file
    they were unpickled from.

    Behaves as a read-only mapping over basin id -> that basin's per-frequency
    results (``in``, ``[]``, ``len()``, iteration, ``.get()``) so it can be
    used wherever the raw ``period_results`` mapping used to be, while
    remaining impossible to fabricate.

    ``results_sha256`` is the SHA-256 of the raw bytes of ``results_path``,
    computed once at load time from the same path that was then unpickled.
    """

    run_dir: str
    period: str
    epoch: int
    results_path: str
    results_sha256: str
    results_size_bytes: int
    n_basins: int
    _results: Mapping[str, Any] = field(repr=False, compare=False)
    _token: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CONSTRUCTION_TOKEN:
            raise AuthenticatedPeriodResultsError(
                "AuthenticatedPeriodResults cannot be constructed directly -- it may only be produced by "
                "load_authenticated_period_results(), which proves the results came from a specific "
                "run/period/epoch pickle whose bytes it hashed"
            )
        if not isinstance(self._results, Mapping):
            raise AuthenticatedPeriodResultsError(
                f"period results payload must be a mapping, got {type(self._results).__name__}"
            )
        object.__setattr__(self, "_results", MappingProxyType(dict(self._results)))
        object.__setattr__(self, "_token", None)

    # -- read-only mapping surface ------------------------------------------

    def __contains__(self, basin_id: object) -> bool:
        return basin_id in self._results

    def __getitem__(self, basin_id: str) -> Any:
        return self._results[basin_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def get(self, basin_id: str, default: Any = None) -> Any:
        return self._results.get(basin_id, default)

    def basin_ids(self) -> tuple:
        return tuple(self._results)

    # -- source binding -----------------------------------------------------

    def require_bound_to(self, *, run_dir, period: str, epoch: int) -> None:
        """Fail closed unless this object was authenticated for exactly this
        run directory, period and epoch.

        Call this at every point of use. It is what stops an authenticated
        result for one trial/epoch from being passed into the evaluation of
        another: the object is genuine, but genuine for a different source.
        """
        expected_run_dir = str(Path(run_dir))
        if expected_run_dir != self.run_dir:
            raise AuthenticatedPeriodResultsError(
                f"authenticated period results were loaded from run_dir {self.run_dir!r} but are being used "
                f"for {expected_run_dir!r} -- run-identity contradiction"
            )
        if period != self.period:
            raise AuthenticatedPeriodResultsError(
                f"authenticated period results were loaded for period {self.period!r} but are being used for "
                f"{period!r} -- evaluation-period contradiction"
            )
        if int(epoch) != int(self.epoch):
            raise AuthenticatedPeriodResultsError(
                f"authenticated period results were loaded for epoch {self.epoch!r} but are being used for "
                f"{int(epoch)!r} -- epoch contradiction"
            )

    def source_fields(self) -> dict:
        """Flat, JSON-serializable record of the authenticated source, for
        receipts and result payloads."""
        return {
            "run_dir": self.run_dir,
            "period": self.period,
            "epoch": self.epoch,
            "results_path": self.results_path,
            "results_sha256": self.results_sha256,
            "results_size_bytes": self.results_size_bytes,
            "n_basins": self.n_basins,
        }


def load_authenticated_period_results(
    *,
    run_dir,
    period: str,
    epoch: int,
    expected_sha256: str | None = None,
) -> AuthenticatedPeriodResults:
    """Load one run/period/epoch's evaluation pickle and bind it to its source.

    Resolves the path with the same authority every other consumer uses
    (:func:`nh_seed_evaluation.period_results_path`), requires the file to
    exist, hashes its raw bytes once, then unpickles that same path with
    :func:`nh_seed_evaluation.load_period_results`.

    ``expected_sha256``, when supplied, is required to equal the observed
    digest; this is how a caller that already recorded a trial's validation
    pickle hash proves the file has not changed since.
    """
    results_path = period_results_path(run_dir, period, epoch)
    if not results_path.is_file():
        raise AuthenticatedPeriodResultsError(
            f"period results pickle is absent: {results_path} -- refusing to evaluate a run/epoch that does "
            "not exist on disk"
        )
    observed_sha256 = _sha256_path(results_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise AuthenticatedPeriodResultsError(
            f"{results_path} has sha256 {observed_sha256} but {expected_sha256} was expected -- the "
            "validation pickle is not the product this evaluation was bound to"
        )
    try:
        payload = load_period_results(run_dir, period, epoch)
    except Exception as exc:  # the source is globally unusable, never a basin cell
        raise AuthenticatedPeriodResultsError(
            f"{results_path}: authenticated period results could not be loaded: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AuthenticatedPeriodResultsError(
            f"{results_path}: unpickled period results is a {type(payload).__name__}, not a mapping"
        )
    return AuthenticatedPeriodResults(
        run_dir=str(Path(run_dir)),
        period=str(period),
        epoch=int(epoch),
        results_path=str(results_path),
        results_sha256=observed_sha256,
        results_size_bytes=results_path.stat().st_size,
        n_basins=len(payload),
        _results=payload,
        _token=_CONSTRUCTION_TOKEN,
    )


def fixture_only_period_results(
    *,
    run_dir,
    period: str,
    epoch: int,
    results: Mapping[str, Any],
) -> AuthenticatedPeriodResults:
    """TEST-FIXTURE ONLY. Wrap an in-memory results mapping without a file.

    This exists so the fast synthetic monkeypatch seams in the test suite --
    which substitute lightweight stand-ins for ``load_period_results``,
    ``evaluate_basin_raw_space`` and friends, and therefore never write a
    real ``validation_results.p`` -- can still exercise the real evaluator
    control flow. The returned object reports ``results_path`` and
    ``results_sha256`` as the explicit sentinels below, so a fixture object
    can never be mistaken for, or reported as, an authenticated one.

    It is deliberately NOT importable from any production path: a focused
    test asserts that no module under ``src/baseline`` other than this one
    references this function by name. Production code must call
    :func:`load_authenticated_period_results`, which proves the results came
    from a specific pickle whose bytes it hashed.
    """
    return AuthenticatedPeriodResults(
        run_dir=str(Path(run_dir)),
        period=str(period),
        epoch=int(epoch),
        results_path=FIXTURE_ONLY_RESULTS_PATH,
        results_sha256=FIXTURE_ONLY_RESULTS_SHA256,
        results_size_bytes=-1,
        n_basins=len(results),
        _results=results,
        _token=_CONSTRUCTION_TOKEN,
    )
