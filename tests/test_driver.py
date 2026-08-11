from __future__ import annotations

from seemc_imaging import SEEMC
from seemc_imaging.transport import _init_worker, _worker_task

from synthetic_material import write_synthetic_database


def test_driver_retains_histories_by_energy_and_trajectory(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    model = SEEMC(
        [100.0, 500.0], "Synthetic", 0.2, 3,
        db_path=db_path, seed=91, history=True,
    ).run_simulation(use_parallel=False, progress=False, verbose=False)

    assert [len(group) for group in model.histories] == [3, 3]
    assert [h.trajectory_id for h in model.histories[0]] == [0, 1, 2]
    assert [h.incident_energy for h in model.histories[1]] == [500.0] * 3


def test_parallel_worker_payload_carries_history(tmp_path):
    db_path = write_synthetic_database(tmp_path / "synthetic.pkl")
    _init_worker("Synthetic", db_path, None, 0.2, False, True)
    payload = _worker_task((500.0, [91, 0, 2], 2))

    history = payload[-2]
    diagnostics = payload[-1]
    assert history.trajectory_id == 2
    assert history.incident_energy == 500.0
    assert len(history.electrons) >= 1
    assert diagnostics["elastic_events"] == sum(
        event.kind == "elastic" for event in history.events
    )
