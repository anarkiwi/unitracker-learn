"""Shared fixtures: a stand-in lifter, so the corpus's forms are built without a tracker format."""

from types import SimpleNamespace

import pytest

import optimiser_docs as D
from unitracker_learn import corpus


def _in_process(fn, tasks, jobs=None):  # pylint: disable=unused-argument
    """``pysidtracker.jobs.imap`` without the spawned workers, which the patches do not reach."""
    for index, args in enumerate(tasks):
        try:
            yield index, fn(*args), None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            yield index, None, exc


@pytest.fixture(name="lifter")
def _lifter(monkeypatch):
    """A lifter of any tune to ``D.HOIST``, verifying clean; its calls are recorded."""
    calls = SimpleNamespace(lifted=[], verified=[], fail=None)

    def lift(raw, subtune=0, frames=None):  # pylint: disable=unused-argument
        calls.lifted.append(subtune)
        return D.HOIST

    def verify(raw, subtune=0, frames=0):  # pylint: disable=unused-argument
        calls.verified.append((subtune, frames))
        if calls.fail:
            raise RuntimeError(calls.fail)
        return []

    module = SimpleNamespace(lift=lift, verify=verify)
    monkeypatch.setattr(corpus.cli, "lifter", lambda raw: ("stub", module))
    monkeypatch.setattr(corpus.cli, "lift", lift)
    monkeypatch.setattr(corpus.cli, "verify", verify)
    monkeypatch.setattr(corpus, "imap", _in_process)
    corpus.versions.cache_clear()
    yield calls
    corpus.versions.cache_clear()
