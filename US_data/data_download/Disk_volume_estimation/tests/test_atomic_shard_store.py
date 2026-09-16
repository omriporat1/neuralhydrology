"""RD1-C4-D1 section J: generic atomic shard/receipt store.

These tests are deliberately free of any D1 scientific content -- the store
is the reusable mechanism behind the ``nse_kge/``, ``objective_receipts/``,
``observation_provenance/`` and ``q98/`` families, and its atomicity must
hold independently of what a shard happens to contain.
"""
from __future__ import annotations

import json

import pytest

from src.baseline.atomic_shard_store import (
    SHARD_RECEIPT_SCHEMA_NAME,
    SHARD_RECEIPT_SCHEMA_VERSION,
    AtomicShardStore,
    ShardStoreError,
    atomic_write_bytes,
    canonical_json_bytes,
    canonical_json_sha256,
    iter_families,
    sha256_path,
)

IDENTITY = {"trial_id": "t01", "best_epoch": 9, "contract_sha256": "a" * 64}


def _store(tmp_path, family="observation_provenance"):
    return AtomicShardStore(tmp_path / "store", family)


def _publish(store, shard_id="t01", *, identity=None, token="attempt1", payload=b"rows"):
    attempt = store.begin_attempt(shard_id, attempt_token=token)
    (attempt / "cells.json").write_bytes(payload)
    (attempt / "detail" / "part-0.bin").parent.mkdir(parents=True, exist_ok=True)
    (attempt / "detail" / "part-0.bin").write_bytes(b"detail")
    return store.publish(shard_id, attempt_dir=attempt, identity=identity or IDENTITY)


# --- canonical serialisation ------------------------------------------- #


def test_canonical_json_is_key_order_independent():
    assert canonical_json_sha256({"b": 1, "a": 2}) == canonical_json_sha256({"a": 2, "b": 1})


def test_canonical_json_refuses_nan():
    """NaN is not JSON. Letting it through would produce a receipt that a
    strict reader cannot parse back."""
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": float("nan")})


def test_atomic_write_returns_the_hash_of_what_landed(tmp_path):
    target = tmp_path / "nested" / "payload.json"
    digest = atomic_write_bytes(target, b"hello")
    assert target.read_bytes() == b"hello"
    assert digest == sha256_path(target)
    assert not list(tmp_path.glob("**/*.tmp*"))


# --- successful publication -------------------------------------------- #


def test_publication_produces_a_verifiable_shard_and_receipt(tmp_path):
    store = _store(tmp_path)
    receipt = _publish(store)

    assert receipt.schema_name == SHARD_RECEIPT_SCHEMA_NAME
    assert receipt.schema_version == SHARD_RECEIPT_SCHEMA_VERSION
    assert receipt.family == "observation_provenance"
    assert receipt.shard_id == "t01"
    assert receipt.identity == IDENTITY
    assert [component.relative_path for component in receipt.components] == [
        "cells.json",
        "detail/part-0.bin",
    ]
    assert store.verify("t01").content_sha256 == receipt.content_sha256
    assert store.list_completed() == ["t01"]


def test_component_paths_are_posix_and_deterministically_ordered(tmp_path):
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="a")
    for name in ("z.bin", "a.bin", "m/inner.bin"):
        path = attempt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    receipt = store.publish("t01", attempt_dir=attempt, identity=IDENTITY)
    assert [c.relative_path for c in receipt.components] == ["a.bin", "m/inner.bin", "z.bin"]


def test_the_receipt_is_written_after_the_shard_is_in_place(tmp_path):
    """The receipt is the completion marker, so the shard content must
    already be readable at the published path by the time it exists."""
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="a")
    (attempt / "cells.json").write_bytes(b"rows")
    store.publish("t01", attempt_dir=attempt, identity=IDENTITY)

    receipt_payload = json.loads(store.receipt_path("t01").read_text(encoding="utf-8"))
    for component in receipt_payload["components"]:
        assert (store.shard_dir("t01") / component["relative_path"]).is_file()
    assert not attempt.exists()


def test_attempt_directory_is_consumed_not_copied(tmp_path):
    store = _store(tmp_path)
    _publish(store)
    assert list(store.attempts_dir().iterdir()) == []


# --- interrupted work never looks completed ---------------------------- #


def test_an_abandoned_attempt_is_not_a_completed_shard(tmp_path):
    """Simulates a task killed mid-trial: content exists in an attempt
    directory, but reduction must not see it."""
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="job-1")
    (attempt / "cells.json").write_bytes(b"partial rows")

    assert store.completed_receipt("t01") is None
    assert store.list_completed() == []
    assert not store.shard_dir("t01").exists()
    assert not store.receipt_path("t01").exists()
    # and the evidence is preserved, not cleaned up
    assert (attempt / "cells.json").read_bytes() == b"partial rows"


def test_an_injected_failure_during_the_trial_leaves_no_shard_and_no_receipt(tmp_path):
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="job-1")
    (attempt / "cells.json").write_bytes(b"rows so far")
    try:
        (attempt / "detail.bin").write_bytes(b"x")
        raise RuntimeError("basin 231 exploded")
    except RuntimeError:
        pass  # the producer aborts without calling publish()

    assert store.receipt_path("t01").exists() is False
    assert store.list_completed() == []
    with pytest.raises(ShardStoreError, match="no completed shard to verify"):
        store.verify("t01")


def test_a_shard_directory_without_a_receipt_is_not_completed_and_is_not_overwritten(tmp_path):
    """The crash window between the rename and the receipt write."""
    store = _store(tmp_path)
    shard_dir = store.shard_dir("t01")
    shard_dir.mkdir(parents=True)
    (shard_dir / "cells.json").write_bytes(b"orphan")

    assert store.completed_receipt("t01") is None
    assert store.list_completed() == []

    attempt = store.begin_attempt("t01", attempt_token="job-2")
    (attempt / "cells.json").write_bytes(b"rows")
    with pytest.raises(ShardStoreError, match="refusing to overwrite"):
        store.publish("t01", attempt_dir=attempt, identity=IDENTITY)
    assert (shard_dir / "cells.json").read_bytes() == b"orphan"


def test_empty_attempt_is_refused(tmp_path):
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="a")
    with pytest.raises(ShardStoreError, match="empty shard"):
        store.publish("t01", attempt_dir=attempt, identity=IDENTITY)


def test_missing_attempt_directory_is_refused(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ShardStoreError, match="does not exist"):
        store.publish("t01", attempt_dir=tmp_path / "nope", identity=IDENTITY)


# --- never overwrite a completed shard --------------------------------- #


def test_publishing_twice_is_refused(tmp_path):
    store = _store(tmp_path)
    _publish(store, payload=b"first")
    with pytest.raises(ShardStoreError, match="already completed"):
        store.begin_attempt("t01", attempt_token="attempt2")
    assert (store.shard_dir("t01") / "cells.json").read_bytes() == b"first"


def test_concurrent_attempts_get_distinct_directories(tmp_path):
    store = _store(tmp_path)
    first = store.begin_attempt("t01", attempt_token="job-1_0")
    second = store.begin_attempt("t01", attempt_token="job-2_0")
    assert first != second
    with pytest.raises(ShardStoreError, match="attempt tokens must be unique"):
        store.begin_attempt("t01", attempt_token="job-1_0")


def test_a_reused_attempt_token_cannot_silently_share_a_directory(tmp_path):
    store = _store(tmp_path)
    attempt = store.begin_attempt("t01", attempt_token="same")
    (attempt / "cells.json").write_bytes(b"from the first attempt")
    with pytest.raises(ShardStoreError):
        store.begin_attempt("t01", attempt_token="same")
    assert (attempt / "cells.json").read_bytes() == b"from the first attempt"


# --- resume ------------------------------------------------------------- #


def test_reuse_is_allowed_only_on_an_exact_identity_match(tmp_path):
    store = _store(tmp_path)
    published = _publish(store)
    receipt, mismatches = store.reuse_if_identical("t01", dict(IDENTITY))
    assert mismatches == []
    assert receipt is not None
    assert receipt.identity_sha256 == published.identity_sha256


def test_nothing_to_reuse_reports_no_receipt_rather_than_failing(tmp_path):
    store = _store(tmp_path)
    receipt, mismatches = store.reuse_if_identical("t01", dict(IDENTITY))
    assert receipt is None
    assert mismatches == []


def test_a_conflicting_identity_refuses_reuse_and_names_every_disagreeing_field(tmp_path):
    store = _store(tmp_path)
    _publish(store)
    conflicting = dict(IDENTITY)
    conflicting["best_epoch"] = 11
    conflicting["contract_sha256"] = "b" * 64
    conflicting["new_field"] = "added"

    with pytest.raises(ShardStoreError) as excinfo:
        store.reuse_if_identical("t01", conflicting)
    message = str(excinfo.value)
    assert "refusing to reuse, overwrite, or recompute" in message
    for field in ("best_epoch", "contract_sha256", "new_field"):
        assert field in message
    # trial_id agrees and must not be reported as a mismatch
    assert '"field": "trial_id"' not in message


def test_a_conflicting_identity_leaves_the_existing_shard_untouched(tmp_path):
    store = _store(tmp_path)
    _publish(store, payload=b"original rows")
    before = store.verify("t01")
    with pytest.raises(ShardStoreError):
        store.reuse_if_identical("t01", {"trial_id": "t01", "best_epoch": 11})
    after = store.verify("t01")
    assert after.content_sha256 == before.content_sha256
    assert (store.shard_dir("t01") / "cells.json").read_bytes() == b"original rows"


# --- damage detection ---------------------------------------------------- #


def test_verify_detects_a_modified_component(tmp_path):
    store = _store(tmp_path)
    _publish(store, payload=b"rows")
    (store.shard_dir("t01") / "cells.json").write_bytes(b"ROWS")  # same length
    with pytest.raises(ShardStoreError, match="has sha256"):
        store.verify("t01")


def test_a_truncated_component_is_damaged_not_absent(tmp_path):
    store = _store(tmp_path)
    _publish(store)
    (store.shard_dir("t01") / "cells.json").write_bytes(b"")
    with pytest.raises(ShardStoreError, match="damaged, not merely incomplete"):
        store.completed_receipt("t01")


def test_a_deleted_component_is_damaged_not_absent(tmp_path):
    store = _store(tmp_path)
    _publish(store)
    (store.shard_dir("t01") / "cells.json").unlink()
    with pytest.raises(ShardStoreError, match="damaged, not merely incomplete"):
        store.completed_receipt("t01")


def test_an_unparseable_receipt_is_never_treated_as_no_receipt(tmp_path):
    store = _store(tmp_path)
    _publish(store)
    store.receipt_path("t01").write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ShardStoreError, match="unreadable shard receipt"):
        store.read_receipt("t01")


def test_a_receipt_naming_another_shard_is_refused(tmp_path):
    store = _store(tmp_path)
    receipt = _publish(store)
    payload = receipt.as_dict()
    payload["shard_id"] = "t02"
    store.receipt_path("t01").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ShardStoreError, match="is for observation_provenance/t02"):
        store.read_receipt("t01")


def test_a_receipt_from_another_schema_version_is_refused(tmp_path):
    store = _store(tmp_path)
    receipt = _publish(store)
    payload = receipt.as_dict()
    payload["schema_version"] = SHARD_RECEIPT_SCHEMA_VERSION + 1
    store.receipt_path("t01").write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ShardStoreError, match="receipt schema version"):
        store.read_receipt("t01")


# --- family scoping (the extension point) ------------------------------- #


def test_families_are_isolated_from_one_another(tmp_path):
    root = tmp_path / "store"
    observation = AtomicShardStore(root, "observation_provenance")
    objectives = AtomicShardStore(root, "objective_receipts")
    _publish(observation, "t01")
    _publish(objectives, "t01", identity={"kind": "objective"})

    assert observation.list_completed() == ["t01"]
    assert objectives.list_completed() == ["t01"]
    assert observation.verify("t01").family == "observation_provenance"
    assert objectives.verify("t01").family == "objective_receipts"
    assert set(iter_families(root)) == {"observation_provenance", "objective_receipts"}


def test_generic_families_share_the_same_atomicity_guarantees(tmp_path):
    """The mechanism is scientific-content free: the same publish/verify/
    reuse behaviour must hold for the NSE/KGE and Q98 families, which are
    documented extension points and are NOT implemented by this task."""
    root = tmp_path / "store"
    for family in ("nse_kge", "objective_receipts", "observation_provenance", "q98"):
        store = AtomicShardStore(root, family)
        _publish(store, "shard", identity={"family": family})
        assert store.verify("shard").family == family
        with pytest.raises(ShardStoreError, match="already completed"):
            store.begin_attempt("shard", attempt_token="second")


@pytest.mark.parametrize("family", ["", "with/slash", ".hidden", "with\\backslash"])
def test_invalid_family_names_are_refused(tmp_path, family):
    with pytest.raises(ShardStoreError, match="invalid shard family name"):
        AtomicShardStore(tmp_path, family)


@pytest.mark.parametrize("shard_id", ["", "..", "a/b"])
def test_invalid_shard_ids_are_refused(tmp_path, shard_id):
    store = _store(tmp_path)
    with pytest.raises(ShardStoreError, match="invalid shard id"):
        store.shard_dir(shard_id)
