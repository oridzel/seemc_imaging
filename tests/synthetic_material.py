"""Small deterministic material database used only by the regression tests."""

from __future__ import annotations

import pickle

import numpy as np


def write_synthetic_database(path):
    energy = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
                       500.0, 1000.0, 2000.0])
    theta = np.linspace(0.0, np.pi, 181)
    angular = np.exp(2.5 * np.cos(theta)) + 0.02
    decs = np.repeat(angular[:, None], energy.size, axis=1)

    losses = np.linspace(0.5, 80.0, 161)
    loss_pdf_se = np.exp(-losses / 14.0)
    loss_pdf_pl = np.exp(-0.5 * ((losses - 18.0) / 5.0) ** 2)

    diimfp_se = np.empty((losses.size, 2, energy.size))
    diimfp_pl = np.empty_like(diimfp_se)
    for j in range(energy.size):
        diimfp_se[:, 0, j] = losses
        diimfp_se[:, 1, j] = loss_pdf_se
        diimfp_pl[:, 0, j] = losses
        diimfp_pl[:, 1, j] = loss_pdf_pl

    omega = np.linspace(0.5, 100.0, 160)
    q = np.geomspace(1.0e-4, 50.0, 180)
    # Smooth and strictly positive maps keep the test focused on transport and
    # provenance rather than on interpolation edge cases.
    omega_shape = np.exp(-omega[:, None] / 45.0) + 0.05
    q_shape = 1.0 / (1.0 + (q[None, :] / 2.0) ** 2)
    elf_se = omega_shape * q_shape
    elf_pl = (np.exp(-0.5 * ((omega[:, None] - 18.0) / 8.0) ** 2) + 0.02) \
        * (1.0 / (1.0 + (q[None, :] / 0.8) ** 4))

    # Rates are intentionally moderate: cascades occur often enough to test
    # ancestry, while trajectories stay short enough for a fast unit test.
    imfp = np.full(energy.size, 18.0)
    emfp = np.full(energy.size, 9.0)
    inv_total = 1.0 / imfp

    material = {
        "name": "Synthetic",
        "is_metal": True,
        "energy": energy,
        "e_fermi": 7.0,
        "work_function": 4.5,
        "e_vb": 7.0,
        "atomic_number": 29,
        "imfp": imfp,
        "emfp": emfp,
        "inv_imfp_se": 0.72 * inv_total,
        "inv_imfp_pl": 0.28 * inv_total,
        "decs_theta": theta,
        "decs": decs,
        "diimfp_se": diimfp_se,
        "diimfp_pl": diimfp_pl,
        "omega": omega,
        "q": q,
        "elf_se": elf_se,
        "elf_pl": elf_pl,
    }
    with open(path, "wb") as stream:
        pickle.dump(material, stream)
    return path
