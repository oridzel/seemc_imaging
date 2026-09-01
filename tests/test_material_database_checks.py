from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

from seemc_imaging import Sample

from synthetic_material import write_synthetic_database

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import check_material_database as checker  # noqa: E402


@pytest.fixture
def good_database(tmp_path):
    path = tmp_path / "good.pkl"
    write_synthetic_database(path)
    with open(path, "rb") as stream:
        return path, pickle.load(stream)


def _write(tmp_path, name, material):
    path = tmp_path / f"{name}.pkl"
    with open(path, "wb") as stream:
        pickle.dump(material, stream)
    return path


def _mutate(material, **changes):
    data = {k: (v.copy() if hasattr(v, "copy") else v)
            for k, v in material.items()}
    data.update(changes)
    return data


def test_duplicate_q_point_names_the_key_and_the_index(tmp_path, good_database):
    _, material = good_database
    q = material["q"].copy()
    q[100] = q[99]
    path = _write(tmp_path, "dup", _mutate(material, q=q))

    with pytest.raises(ValueError) as error:
        Sample("Synthetic", db_path=path)
    message = str(error.value)
    assert "material_data['q']" in message
    assert "strictly increasing" in message
    assert "99" in message
    assert "repeated" in message


def test_unsorted_q_grid_says_so(tmp_path, good_database):
    _, material = good_database
    path = _write(tmp_path, "rev", _mutate(
        material,
        q=material["q"][::-1].copy(),
        elf_se=material["elf_se"][:, ::-1].copy(),
        elf_pl=material["elf_pl"][:, ::-1].copy(),
    ))
    with pytest.raises(ValueError, match="not sorted ascending"):
        Sample("Synthetic", db_path=path)


def test_duplicate_omega_point_is_attributed_to_omega(tmp_path, good_database):
    _, material = good_database
    omega = material["omega"].copy()
    omega[50] = omega[49]
    path = _write(tmp_path, "omega", _mutate(material, omega=omega))
    with pytest.raises(ValueError, match=r"material_data\['omega'\]"):
        Sample("Synthetic", db_path=path)


def test_checker_reports_each_failure_mode(good_database):
    _, material = good_database

    q = material["q"].copy()
    q[100] = q[99]
    findings = checker.check(_mutate(material, q=q), verbose=False)
    assert any("repeated" in item and item.startswith("q:") for item in findings)

    findings = checker.check(
        _mutate(material, q=material["q"][::-1].copy()), verbose=False
    )
    assert any("decreasing" in item for item in findings)

    assert checker.check(material, verbose=False) == []


def test_checker_flags_a_grid_and_array_length_mismatch(good_database):
    _, material = good_database
    truncated = _mutate(material, elf_se=material["elf_se"][:, :-3].copy())
    findings = checker.check(truncated, verbose=False)
    assert any("elf_se" in item and "axis 1" in item for item in findings)


def test_repair_restores_a_descending_table_exactly(tmp_path, good_database):
    """A table stored in descending q is the same physics, stored backwards."""
    _, material = good_database
    descending = _mutate(
        material,
        q=material["q"][::-1].copy(),
        elf_se=material["elf_se"][:, ::-1].copy(),
        elf_pl=material["elf_pl"][:, ::-1].copy(),
    )
    repaired, actions = checker.repair(descending, verbose=False)

    assert any("sorted ascending" in line for line in actions)
    assert np.array_equal(repaired["q"], material["q"])
    assert np.array_equal(repaired["elf_se"], material["elf_se"])
    assert np.array_equal(repaired["elf_pl"], material["elf_pl"])

    # And the repaired database actually loads.
    path = _write(tmp_path, "repaired", repaired)
    Sample("Synthetic", db_path=path)


def test_repair_drops_duplicates_and_the_matching_elf_columns(
        tmp_path, good_database):
    _, material = good_database
    q = material["q"]
    overlapped = _mutate(
        material,
        q=np.concatenate([q[:120], q[100:]]),
        elf_se=np.concatenate(
            [material["elf_se"][:, :120], material["elf_se"][:, 100:]], axis=1),
        elf_pl=np.concatenate(
            [material["elf_pl"][:, :120], material["elf_pl"][:, 100:]], axis=1),
    )
    repaired, _ = checker.repair(overlapped, verbose=False)

    assert np.array_equal(repaired["q"], q)
    assert repaired["elf_se"].shape == material["elf_se"].shape
    # Every surviving column still belongs to its own q point.
    assert np.array_equal(repaired["elf_se"], material["elf_se"])

    path = _write(tmp_path, "deduped", repaired)
    Sample("Synthetic", db_path=path)


def test_repair_leaves_a_clean_database_untouched(good_database):
    _, material = good_database
    repaired, actions = checker.repair(material, verbose=False)
    assert actions == []
    for key in ("q", "omega", "energy", "decs_theta"):
        assert np.array_equal(repaired[key], material[key])
