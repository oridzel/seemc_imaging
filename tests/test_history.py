from __future__ import annotations

import numpy as np

from seemc_imaging import Sample, simulate_trajectory

from synthetic_material import write_synthetic_database


def _sample(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    return Sample("Synthetic", db_path=db_path)


def _physical_signature(result):
    emissions = [
        (e.energy, e.uz, e.is_cascade, e.generation, e.birth_depth,
         e.xyz, e.uvw)
        for e in result.emissions
    ]
    return (
        result.tey,
        result.sey_cascade,
        result.bse_cascade,
        result.sey_50ev,
        result.bse_50ev,
        emissions,
        dict(result.diagnostics),
    )


def test_history_is_rng_transparent(tmp_path):
    sample = _sample(tmp_path)
    plain = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(1234), history=False
    )
    traced = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(1234),
        history=True, trajectory_id=17,
    )

    assert _physical_signature(plain) == _physical_signature(traced)
    assert plain.history is None
    assert traced.history.trajectory_id == 17


def test_history_ancestry_links_counts_and_fates(tmp_path):
    sample = _sample(tmp_path)
    result = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(3),
        history=True, trajectory_id=3,
    )
    history = result.history

    assert [r.electron_id for r in history.electrons] == list(
        range(len(history.electrons))
    )
    assert [e.event_id for e in history.events] == list(range(len(history.events)))
    assert history.electrons[0].parent_id is None
    assert history.electrons[0].root_primary_id == 0
    assert all(record.fate is not None for record in history.electrons)

    by_event = {event.event_id: event for event in history.events}
    child_sources = {
        event.child_id: event
        for event in history.events
        if event.child_id is not None
    }
    for record in history.electrons[1:]:
        assert record.parent_id < record.electron_id
        assert record.root_primary_id == 0
        assert record.electron_id in child_sources
        assert child_sources[record.electron_id].electron_id == record.parent_id
        ancestry = history.ancestry(record.electron_id)
        assert ancestry[0].electron_id == 0
        assert ancestry[-1].electron_id == record.electron_id

    elastic = [e for e in history.events if e.kind == "elastic"]
    inelastic = [e for e in history.events if e.kind == "inelastic"]
    emissions = [e for e in history.events if e.kind == "emission"]
    assert len(elastic) == result.diagnostics["elastic_events"]
    assert len(inelastic) == result.diagnostics["inelastic_events"]
    assert len(emissions) == result.tey

    for event in inelastic:
        assert np.isclose(
            event.energy_before - event.energy_after,
            event.energy_loss,
            rtol=0.0,
            atol=1e-12,
        )
        assert event.mechanism in {
            "binary", "plasmon", "binary_pauli_fallback", "binary_dropped"
        }

    for record in history.electrons:
        for event_id in (
            record.first_beam_reversal_event_id,
            record.first_surface_return_event_id,
        ):
            if event_id is not None:
                assert by_event[event_id].electron_id == record.electron_id
                assert by_event[event_id].kind in {"elastic", "inelastic"}

    emitted_ids = {e.electron_id for e in result.emissions}
    assert emitted_ids == {
        record.electron_id
        for record in history.electrons
        if record.fate == "emitted"
    }


def test_history_serializes_to_plain_python_rows(tmp_path):
    sample = _sample(tmp_path)
    result = simulate_trajectory(
        sample, 500.0, 0.35, np.random.default_rng(42), history=True
    )
    payload = result.history.to_dict()
    assert payload["incident_energy"] == 500.0
    assert payload["events"][0]["kind"] == "primary_launch"
    assert payload["electrons"][0]["birth_mechanism"] == "incident_primary"

