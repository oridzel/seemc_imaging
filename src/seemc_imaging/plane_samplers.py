"""Generate JMONSEL-style planar emission sampler libraries with SEEMC.

The exported polar angle is measured from the beam-back direction (opposite
the incident vacuum ray), not from the local sample normal.  At incidence
``alpha`` the outward half-space therefore occupies a subset of the nominal
``0 .. 90 + alpha`` polar range.  The polar inverse CDF alone does not encode
that subset; downstream code must sample or clip azimuth so that the final
direction still has positive dot product with the sample outward normal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from .geometry import Plane
from .transport import MCConfig, SEEMC
from .plane_sampler_joint_export import (
    direction_angles,
    write_joint_angle_samplers,
)


JMONSEL_ENERGIES_EV = (
    75.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0, 300.0, 350.0,
    400.0, 500.0, 800.0, 1000.0, 1500.0, 2000.0, 3500.0, 5000.0,
    8000.0, 10000.0,
)

DEFAULT_INCIDENCE_ANGLES_DEG = (
    0.0, 15.0, 30.0, 45.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    83.0, 85.0, 87.0, 89.0,
)

SEY_FILENAME = "SEYFromPlane_SEVaccum_t0nmCuFPA.csv"
BSEY_FILENAME = "BSEYFromPlane_SEVaccum_t0nmCuFPA.csv"
SE_ENERGY_FILENAME = "SEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"
BSE_ENERGY_FILENAME = "BSEeEFromPlaneSampler_SEVaccum_t0nmCuFPA.csv"
SE_THETA_FILENAME = "SEThetaFromPlaneSampler_uncoatedCuFPA.csv"
BSE_THETA_FILENAME = "BSEThetaFromPlaneSampler_uncoatedCuFPA.csv"

CHECKPOINT_SCHEMA = "seemc-plane-sampler-case-v2"
PACKAGE_VERSION = "0.7.5"


def cosine_probability_grid(size: int = 513) -> np.ndarray:
    """Return a common inverse-CDF grid with extra resolution in both tails."""
    size = int(size)
    if size < 3:
        raise ValueError("probability-grid size must be at least 3")
    u = np.linspace(0.0, 1.0, size)
    return 0.5 * (1.0 - np.cos(np.pi * u))


def plane_directions(incidence_angle_deg: float):
    """Return ``(vacuum_direction, outward_normal, beam_back)`` for Plane()."""
    angle = float(incidence_angle_deg)
    if not math.isfinite(angle) or angle < 0.0 or angle >= 90.0:
        raise ValueError("incidence angle must be finite and in [0, 90) degrees")
    alpha = math.radians(angle)
    outward = np.array((0.0, 0.0, -1.0), dtype=float)
    vacuum = np.array((math.sin(alpha), 0.0, math.cos(alpha)), dtype=float)
    beam_back = -vacuum
    return tuple(vacuum), tuple(outward), tuple(beam_back)


def deterministic_case_seed(base_seed: int, incidence_angle_deg: float,
                            energy_ev: float) -> int:
    """Derive a stable case seed without depending on loop order or workers."""
    angle_key = int(round(float(incidence_angle_deg) * 1_000_000.0))
    energy_key = int(round(float(energy_ev) * 1_000.0))
    sequence = np.random.SeedSequence(
        [int(base_seed), angle_key & 0xFFFFFFFF, energy_key & 0xFFFFFFFF]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _sem_from_counts(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    if values.size == 0:
        raise ValueError("cannot calculate a standard error from zero primaries")
    variance = max(float(np.mean(values * values) - np.mean(values) ** 2), 0.0)
    return math.sqrt(variance / values.size)


@dataclass
class PlaneSamplerCase:
    """Raw emissions and yield statistics for one angle-energy simulation."""

    incidence_angle_deg: float
    incident_energy_ev: float
    n_primaries: int
    case_seed: int
    energy_cutoff_ev: float
    se_energy_ev: np.ndarray
    bse_energy_ev: np.ndarray
    se_theta_deg: np.ndarray
    bse_theta_deg: np.ndarray

    # Event-level joint direction data.  Each row stays paired with the
    # corresponding emitted energy and primary ID.
    se_phi_deg: np.ndarray
    bse_phi_deg: np.ndarray
    se_direction_xyz: np.ndarray
    bse_direction_xyz: np.ndarray
    se_mu_beam_back: np.ndarray
    bse_mu_beam_back: np.ndarray
    se_mu_toward_normal: np.ndarray
    bse_mu_toward_normal: np.ndarray
    se_mu_side: np.ndarray
    bse_mu_side: np.ndarray
    se_emission_mechanism: np.ndarray
    bse_emission_mechanism: np.ndarray
    se_barrier_reflection_probability: np.ndarray
    bse_barrier_reflection_probability: np.ndarray

    se_primary_id: np.ndarray
    bse_primary_id: np.ndarray
    se_counts_per_primary: np.ndarray
    bse_counts_per_primary: np.ndarray

    @property
    def sey(self) -> float:
        return float(self.se_energy_ev.size / self.n_primaries)

    @property
    def bsey(self) -> float:
        return float(self.bse_energy_ev.size / self.n_primaries)

    @property
    def tey(self) -> float:
        return self.sey + self.bsey

    @property
    def incoming_barrier_reflections(self) -> int:
        """Number of primaries reflected directly at the vacuum->solid barrier."""
        return int(np.count_nonzero(
            self.bse_emission_mechanism == "incoming_barrier_reflection"
        ))

    @property
    def incoming_barrier_reflection_fraction(self) -> float:
        return float(self.incoming_barrier_reflections / self.n_primaries)

    @property
    def sey_sem(self) -> float:
        return _sem_from_counts(self.se_counts_per_primary)

    @property
    def bsey_sem(self) -> float:
        return _sem_from_counts(self.bse_counts_per_primary)

    def validate(self, tolerance: float = 1.0e-9) -> None:
        if self.n_primaries < 1:
            raise ValueError("n_primaries must be positive")
        if self.incident_energy_ev <= self.energy_cutoff_ev:
            raise ValueError("incident energy must exceed the SE/BSE cutoff")
        pairs = (
            (
                self.se_energy_ev, self.se_theta_deg, self.se_phi_deg,
                self.se_direction_xyz, self.se_mu_beam_back,
                self.se_mu_toward_normal, self.se_mu_side,
                self.se_emission_mechanism,
                self.se_barrier_reflection_probability,
                self.se_primary_id, "SE",
            ),
            (
                self.bse_energy_ev, self.bse_theta_deg, self.bse_phi_deg,
                self.bse_direction_xyz, self.bse_mu_beam_back,
                self.bse_mu_toward_normal, self.bse_mu_side,
                self.bse_emission_mechanism,
                self.bse_barrier_reflection_probability,
                self.bse_primary_id, "BSE",
            ),
        )
        for (
            energies, theta, phi, directions, mu_b, mu_t, mu_s,
            mechanism, barrier_r, primary_ids, label
        ) in pairs:
            one_d = (
                energies, theta, phi, mu_b, mu_t, mu_s,
                mechanism, barrier_r, primary_ids,
            )
            if any(array.ndim != 1 for array in one_d):
                raise ValueError(f"{label} scalar raw arrays must be one-dimensional")
            if directions.ndim != 2 or directions.shape[1] != 3:
                raise ValueError(f"{label} direction array must have shape (N, 3)")
            n = energies.size
            if any(array.size != n for array in one_d[1:]) or directions.shape[0] != n:
                raise ValueError(f"{label} joint emission-array lengths differ")
            numeric = (energies, theta, phi, mu_b, mu_t, mu_s, directions)
            if any(not np.all(np.isfinite(array)) for array in numeric):
                raise ValueError(f"{label} raw arrays contain non-finite values")
            # barrier_r is NaN for ordinary transport escapes and finite only
            # for direct incoming-barrier reflections.
            finite_r = np.isfinite(barrier_r)
            if np.any(finite_r & ((barrier_r < 0.0) | (barrier_r > 1.0))):
                raise ValueError(f"{label} barrier reflection probability is outside [0,1]")
            if primary_ids.size and (
                int(primary_ids.min()) < 0
                or int(primary_ids.max()) >= self.n_primaries
            ):
                raise ValueError(f"{label} primary IDs are outside the case range")

            if n:
                norms = np.linalg.norm(directions, axis=1)
                if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-10):
                    raise ValueError(f"{label} emitted directions are not unit vectors")
                local_norm = np.sqrt(mu_b * mu_b + mu_t * mu_t + mu_s * mu_s)
                if not np.allclose(local_norm, 1.0, rtol=0.0, atol=2.0e-10):
                    raise ValueError(f"{label} local direction cosines are inconsistent")
        if self.se_counts_per_primary.shape != (self.n_primaries,):
            raise ValueError("SE per-primary counts have the wrong length")
        if self.bse_counts_per_primary.shape != (self.n_primaries,):
            raise ValueError("BSE per-primary counts have the wrong length")
        if int(self.se_counts_per_primary.sum()) != self.se_energy_ev.size:
            raise ValueError("SE per-primary counts do not match raw emissions")
        if int(self.bse_counts_per_primary.sum()) != self.bse_energy_ev.size:
            raise ValueError("BSE per-primary counts do not match raw emissions")
        if self.se_energy_ev.size and (
            float(self.se_energy_ev.min()) < -tolerance
            or float(self.se_energy_ev.max()) >= self.energy_cutoff_ev + tolerance
        ):
            raise ValueError("SE energies violate the < cutoff definition")
        if self.bse_energy_ev.size and (
            float(self.bse_energy_ev.min()) < self.energy_cutoff_ev - tolerance
            or float(self.bse_energy_ev.max()) > self.incident_energy_ev + tolerance
        ):
            raise ValueError("BSE energies violate the >= cutoff definition")
        theta_max = 90.0 + self.incidence_angle_deg
        all_theta = np.concatenate((self.se_theta_deg, self.bse_theta_deg))
        if all_theta.size and (
            float(all_theta.min()) < -tolerance
            or float(all_theta.max()) > theta_max + tolerance
        ):
            raise ValueError(
                "beam-relative polar angle lies outside the outward support "
                f"[0, {theta_max:g}] degrees"
            )

        # Reconstruct outward cosine directly from the beam-relative local
        # components.  For incidence alpha:
        #   n_out = cos(alpha)*beam_back + sin(alpha)*toward_normal.
        alpha = math.radians(self.incidence_angle_deg)
        ca, sa = math.cos(alpha), math.sin(alpha)
        for mu_b, mu_t, label in (
            (self.se_mu_beam_back, self.se_mu_toward_normal, "SE"),
            (self.bse_mu_beam_back, self.bse_mu_toward_normal, "BSE"),
        ):
            outward_cos = ca * mu_b + sa * mu_t
            if outward_cos.size and np.any(outward_cos <= -tolerance):
                raise ValueError(
                    f"{label} joint directions contain electrons pointing into the sample"
                )

        # Direct incoming-barrier reflections have an exact planar fingerprint:
        # Eout=E0, theta=2*alpha, phi=0.  Enforce it so a future coordinate
        # convention change cannot silently corrupt the sharp specular lobe.
        if np.any(self.se_emission_mechanism == "incoming_barrier_reflection"):
            raise ValueError(
                "incoming-barrier reflected primaries must be BSE for this "
                "sampler because incident_energy_ev exceeds the 50 eV cutoff"
            )
        reflected = (
            self.bse_emission_mechanism == "incoming_barrier_reflection"
        )
        if np.any(reflected):
            expected_theta = 2.0 * self.incidence_angle_deg
            if not np.allclose(
                self.bse_energy_ev[reflected],
                self.incident_energy_ev,
                rtol=0.0,
                atol=max(tolerance, 1.0e-10),
            ):
                raise ValueError(
                    "incoming-barrier reflected primary did not retain E0"
                )
            if not np.allclose(
                self.bse_theta_deg[reflected],
                expected_theta,
                rtol=0.0,
                atol=2.0e-9,
            ):
                raise ValueError(
                    "incoming-barrier reflected primary is not at theta=2*alpha"
                )
            if not np.allclose(
                self.bse_phi_deg[reflected],
                0.0,
                rtol=0.0,
                atol=2.0e-9,
            ):
                raise ValueError(
                    "incoming-barrier reflected primary is not at phi=0"
                )
            if not np.all(np.isfinite(
                self.bse_barrier_reflection_probability[reflected]
            )):
                raise ValueError(
                    "incoming-barrier reflected primary lacks its reflection probability"
                )


def _config_json(config: MCConfig) -> str:
    return json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))


def _config_sha256(config: MCConfig) -> str:
    return hashlib.sha256(_config_json(config).encode("utf-8")).hexdigest()


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_plane_sampler_case(
    database_path,
    material: str,
    incident_energy_ev: float,
    incidence_angle_deg: float,
    n_primaries: int,
    *,
    config: Optional[MCConfig] = None,
    case_seed: int = 12345,
    workers: int = 1,
    progress: bool = True,
) -> PlaneSamplerCase:
    """Run one planar case and retain the raw emissions needed for a sampler."""
    config = config or MCConfig()
    config.validate()
    if not config.collect_spectra:
        raise ValueError("sampler generation requires config.collect_spectra=True")
    n_primaries = int(n_primaries)
    if n_primaries < 1:
        raise ValueError("n_primaries must be positive")
    energy = float(incident_energy_ev)
    angle = float(incidence_angle_deg)
    if energy <= config.bse_cutoff_ev:
        raise ValueError("incident energy must exceed config.bse_cutoff_ev")
    vacuum, outward, beam_back = plane_directions(angle)

    model = SEEMC(
        [energy], material, math.radians(angle), n_primaries,
        db_path=str(database_path), config=config, seed=int(case_seed),
        history=False, geometry=Plane(), vacuum_direction=vacuum,
        surface_normal=outward,
    ).run_simulation(
        use_parallel=int(workers) > 1,
        workers=int(workers),
        progress=progress,
        verbose=False,
    )

    emissions = model.emissions[0]
    if emissions:
        emission_energy = np.asarray(
            [item.energy for item in emissions], dtype=float
        )
        emission_direction = np.asarray(
            [item.uvw for item in emissions], dtype=float
        )
        primary_id = np.asarray(
            [item.root_primary_id for item in emissions], dtype=np.int64
        )
        (
            emission_direction,
            theta_deg,
            phi_deg,
            mu_beam_back,
            mu_toward_normal,
            mu_side,
            outward_cosine,
        ) = direction_angles(
            emission_direction,
            vacuum_incident_direction=vacuum,
            surface_normal_out=outward,
        )
        if np.any(outward_cosine <= -1.0e-12):
            raise ValueError(
                "transport returned an electron directed into the sample"
            )
        emission_mechanism = np.asarray(
            [
                str(getattr(item, "emission_mechanism", "transport_escape"))
                for item in emissions
            ]
        )
        barrier_reflection_probability = np.asarray(
            [
                np.nan
                if getattr(item, "barrier_reflection_probability", None) is None
                else float(item.barrier_reflection_probability)
                for item in emissions
            ],
            dtype=float,
        )
    else:
        emission_energy = np.empty(0, dtype=float)
        emission_direction = np.empty((0, 3), dtype=float)
        theta_deg = np.empty(0, dtype=float)
        phi_deg = np.empty(0, dtype=float)
        mu_beam_back = np.empty(0, dtype=float)
        mu_toward_normal = np.empty(0, dtype=float)
        mu_side = np.empty(0, dtype=float)
        outward_cosine = np.empty(0, dtype=float)
        emission_mechanism = np.empty(0, dtype="<U1")
        barrier_reflection_probability = np.empty(0, dtype=float)
        primary_id = np.empty(0, dtype=np.int64)

    is_se = emission_energy < config.bse_cutoff_ev
    se_ids = primary_id[is_se]
    bse_ids = primary_id[~is_se]
    if primary_id.size and (primary_id.min() < 0 or primary_id.max() >= n_primaries):
        raise RuntimeError(
            "emissions lack valid per-primary IDs; use seemc-imaging 0.7.4 or newer"
        )
    case = PlaneSamplerCase(
        incidence_angle_deg=angle,
        incident_energy_ev=energy,
        n_primaries=n_primaries,
        case_seed=int(case_seed),
        energy_cutoff_ev=float(config.bse_cutoff_ev),
        se_energy_ev=emission_energy[is_se],
        bse_energy_ev=emission_energy[~is_se],
        se_theta_deg=theta_deg[is_se],
        bse_theta_deg=theta_deg[~is_se],
        se_phi_deg=phi_deg[is_se],
        bse_phi_deg=phi_deg[~is_se],
        se_direction_xyz=emission_direction[is_se],
        bse_direction_xyz=emission_direction[~is_se],
        se_mu_beam_back=mu_beam_back[is_se],
        bse_mu_beam_back=mu_beam_back[~is_se],
        se_mu_toward_normal=mu_toward_normal[is_se],
        bse_mu_toward_normal=mu_toward_normal[~is_se],
        se_mu_side=mu_side[is_se],
        bse_mu_side=mu_side[~is_se],
        se_emission_mechanism=emission_mechanism[is_se],
        bse_emission_mechanism=emission_mechanism[~is_se],
        se_barrier_reflection_probability=barrier_reflection_probability[is_se],
        bse_barrier_reflection_probability=barrier_reflection_probability[~is_se],
        se_primary_id=se_ids,
        bse_primary_id=bse_ids,
        se_counts_per_primary=np.bincount(se_ids, minlength=n_primaries),
        bse_counts_per_primary=np.bincount(bse_ids, minlength=n_primaries),
    )
    case.validate()
    if not math.isclose(case.sey, float(model.sey_50ev[0]), abs_tol=1.0e-14):
        raise RuntimeError("raw SE emissions disagree with the SEEMC yield")
    if not math.isclose(case.bsey, float(model.bse_50ev[0]), abs_tol=1.0e-14):
        raise RuntimeError("raw BSE emissions disagree with the SEEMC yield")
    return case


def _number_token(value: float) -> str:
    token = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return token.replace("-", "m").replace(".", "p")


def angle_directory_name(angle_deg: float) -> str:
    return f"alpha_{_number_token(angle_deg)}deg"


def checkpoint_path(output_directory, angle_deg: float,
                    energy_ev: float) -> Path:
    return (
        Path(output_directory) / angle_directory_name(angle_deg) / "raw"
        / f"E_{_number_token(energy_ev)}eV.npz"
    )


def save_case_checkpoint(path, case: PlaneSamplerCase, *, material: str,
                         config: MCConfig,
                         database_sha256: Optional[str] = None) -> Path:
    """Atomically save one raw case without pickle-dependent object arrays."""
    case.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "material": str(material),
        "config_json": _config_json(config),
        "database_sha256": database_sha256,
        "direction_convention": (
            "beam_back=-incident vacuum direction; phi=0 toward the outward "
            "sample normal in the incidence plane"
        ),
        "stores_joint_energy_direction": True,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            incidence_angle_deg=np.asarray(case.incidence_angle_deg),
            incident_energy_ev=np.asarray(case.incident_energy_ev),
            n_primaries=np.asarray(case.n_primaries, dtype=np.int64),
            case_seed=np.asarray(case.case_seed, dtype=np.uint64),
            energy_cutoff_ev=np.asarray(case.energy_cutoff_ev),
            se_energy_ev=case.se_energy_ev,
            bse_energy_ev=case.bse_energy_ev,
            se_theta_deg=case.se_theta_deg,
            bse_theta_deg=case.bse_theta_deg,
            se_phi_deg=case.se_phi_deg,
            bse_phi_deg=case.bse_phi_deg,
            se_direction_xyz=case.se_direction_xyz,
            bse_direction_xyz=case.bse_direction_xyz,
            se_mu_beam_back=case.se_mu_beam_back,
            bse_mu_beam_back=case.bse_mu_beam_back,
            se_mu_toward_normal=case.se_mu_toward_normal,
            bse_mu_toward_normal=case.bse_mu_toward_normal,
            se_mu_side=case.se_mu_side,
            bse_mu_side=case.bse_mu_side,
            se_emission_mechanism=case.se_emission_mechanism,
            bse_emission_mechanism=case.bse_emission_mechanism,
            se_barrier_reflection_probability=case.se_barrier_reflection_probability,
            bse_barrier_reflection_probability=case.bse_barrier_reflection_probability,
            se_primary_id=case.se_primary_id,
            bse_primary_id=case.bse_primary_id,
            se_counts_per_primary=case.se_counts_per_primary,
            bse_counts_per_primary=case.bse_counts_per_primary,
        )
    os.replace(temporary, path)
    return path


def load_case_checkpoint(path, *, material: Optional[str] = None,
                         config: Optional[MCConfig] = None,
                         database_sha256: Optional[str] = None,
                         expected_angle_deg: Optional[float] = None,
                         expected_energy_ev: Optional[float] = None,
                         expected_n_primaries: Optional[int] = None,
                         expected_case_seed: Optional[int] = None,
                         ) -> PlaneSamplerCase:
    """Load and strictly validate a resumable raw case."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unsupported checkpoint schema in {path}; the joint-direction "
                "sampler requires v2 checkpoints. Re-run this case with "
                "overwrite=True (or use a fresh output directory)."
            )
        if material is not None and metadata.get("material") != str(material):
            raise ValueError(f"checkpoint material mismatch in {path}")
        if config is not None and metadata.get("config_json") != _config_json(config):
            raise ValueError(f"checkpoint MCConfig mismatch in {path}")
        if (database_sha256 is not None
                and metadata.get("database_sha256") != database_sha256):
            raise ValueError(f"checkpoint material-database mismatch in {path}")
        case = PlaneSamplerCase(
            incidence_angle_deg=float(archive["incidence_angle_deg"].item()),
            incident_energy_ev=float(archive["incident_energy_ev"].item()),
            n_primaries=int(archive["n_primaries"].item()),
            case_seed=int(archive["case_seed"].item()),
            energy_cutoff_ev=float(archive["energy_cutoff_ev"].item()),
            se_energy_ev=archive["se_energy_ev"].astype(float),
            bse_energy_ev=archive["bse_energy_ev"].astype(float),
            se_theta_deg=archive["se_theta_deg"].astype(float),
            bse_theta_deg=archive["bse_theta_deg"].astype(float),
            se_phi_deg=archive["se_phi_deg"].astype(float),
            bse_phi_deg=archive["bse_phi_deg"].astype(float),
            se_direction_xyz=archive["se_direction_xyz"].astype(float),
            bse_direction_xyz=archive["bse_direction_xyz"].astype(float),
            se_mu_beam_back=archive["se_mu_beam_back"].astype(float),
            bse_mu_beam_back=archive["bse_mu_beam_back"].astype(float),
            se_mu_toward_normal=archive["se_mu_toward_normal"].astype(float),
            bse_mu_toward_normal=archive["bse_mu_toward_normal"].astype(float),
            se_mu_side=archive["se_mu_side"].astype(float),
            bse_mu_side=archive["bse_mu_side"].astype(float),
            se_emission_mechanism=archive["se_emission_mechanism"].astype(str),
            bse_emission_mechanism=archive["bse_emission_mechanism"].astype(str),
            se_barrier_reflection_probability=archive[
                "se_barrier_reflection_probability"
            ].astype(float),
            bse_barrier_reflection_probability=archive[
                "bse_barrier_reflection_probability"
            ].astype(float),
            se_primary_id=archive["se_primary_id"].astype(np.int64),
            bse_primary_id=archive["bse_primary_id"].astype(np.int64),
            se_counts_per_primary=archive["se_counts_per_primary"].astype(np.int64),
            bse_counts_per_primary=archive["bse_counts_per_primary"].astype(np.int64),
        )
    case.validate()
    expected = (
        (expected_angle_deg, case.incidence_angle_deg, "incidence angle"),
        (expected_energy_ev, case.incident_energy_ev, "incident energy"),
        (expected_n_primaries, case.n_primaries, "primary count"),
        (expected_case_seed, case.case_seed, "case seed"),
    )
    for wanted, actual, label in expected:
        if wanted is not None and not math.isclose(
            float(wanted), float(actual), rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"checkpoint {label} mismatch in {path}")
    return case


def _inverse_cdf(samples: np.ndarray, probabilities: np.ndarray,
                 lower: float, upper: float) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError(
            "cannot construct an inverse CDF from zero emissions; increase "
            "the number of primaries for this angle-energy case"
        )
    quantiles = np.quantile(samples, probabilities, method="linear")
    quantiles = np.clip(quantiles, lower, upper)
    quantiles[0] = lower
    quantiles[-1] = upper
    return np.maximum.accumulate(quantiles)


def _atomic_csv(path: Path, description_lines: Sequence[str], header,
                rows: Iterable[Sequence[object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for line in description_lines:
            writer.writerow([line])
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def _format_number(value: float) -> str:
    return format(float(value), ".17g")


def export_angle_tables(directory, cases: Sequence[PlaneSamplerCase],
                        *, material: str, probabilities: np.ndarray) -> None:
    """Write the six legacy-compatible CSV tables for one incidence angle."""
    if not cases:
        raise ValueError("at least one case is required")
    cases = sorted(cases, key=lambda item: item.incident_energy_ev)
    angle = cases[0].incidence_angle_deg
    cutoff = cases[0].energy_cutoff_ev
    for case in cases:
        case.validate()
        if not math.isclose(case.incidence_angle_deg, angle, abs_tol=1.0e-12):
            raise ValueError("all cases in one export must have the same angle")
        if not math.isclose(case.energy_cutoff_ev, cutoff, abs_tol=1.0e-12):
            raise ValueError("all cases in one export must have the same cutoff")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    angle_text = _format_number(angle)

    yield_description = (
        f"{{label}} vs. energy for {material} substrate tilted "
        f"{angle_text} degrees."
    )
    _atomic_csv(
        directory / SEY_FILENAME,
        [yield_description.format(label="SEY")],
        ("beamE (eV)", "SEY"),
        ((_format_number(case.incident_energy_ev), _format_number(case.sey))
         for case in cases),
    )
    _atomic_csv(
        directory / BSEY_FILENAME,
        [yield_description.format(label="BSEY")],
        ("beamE (eV)", "BSEY"),
        ((_format_number(case.incident_energy_ev), _format_number(case.bsey))
         for case in cases),
    )

    second_line_energy = (
        "eE = sampler[T,r] with r uniform on [0,1]; Interpolate this "
        "tabulation to obtain sampler."
    )
    second_line_theta = (
        "theta = sampler[T,r] with r uniform on [0,1]; Interpolate this "
        "tabulation to obtain sampler."
    )

    def distribution_rows(kind: str, value: str):
        for case in cases:
            if kind == "SE":
                samples = case.se_energy_ev if value == "energy" else case.se_theta_deg
                lower = 0.0
                upper = cutoff if value == "energy" else 90.0 + angle
            else:
                samples = case.bse_energy_ev if value == "energy" else case.bse_theta_deg
                lower = cutoff if value == "energy" else 0.0
                upper = case.incident_energy_ev if value == "energy" else 90.0 + angle
            quantiles = _inverse_cdf(samples, probabilities, lower, upper)
            for probability, quantile in zip(probabilities, quantiles):
                yield (
                    _format_number(case.incident_energy_ev),
                    _format_number(probability),
                    _format_number(quantile),
                )

    for kind, filename in (("SE", SE_ENERGY_FILENAME), ("BSE", BSE_ENERGY_FILENAME)):
        _atomic_csv(
            directory / filename,
            [f"Sampler for emission energy (eE) of {kind} from a planar "
             f"{material} substrate tilted {angle_text} degrees.",
             second_line_energy],
            ("beamE (eV)", "r", "eE (eV)"),
            distribution_rows(kind, "energy"),
        )
    for kind, filename in (("SE", SE_THETA_FILENAME), ("BSE", BSE_THETA_FILENAME)):
        _atomic_csv(
            directory / filename,
            [f"Sampler for beam-relative polar angle of {kind} from a planar "
             f"{material} substrate tilted {angle_text} degrees.",
             second_line_theta],
            ("beamE (eV)", "r", "theta (deg)"),
            distribution_rows(kind, "theta"),
        )

    readme = directory / "readme.txt"
    temporary = readme.with_suffix(".txt.tmp")
    temporary.write_text(
        f"Planar {material} emission samplers at {angle_text} degrees incidence.\n"
        f"SE means emitted energy < {cutoff:g} eV; BSE means emitted energy "
        f">= {cutoff:g} eV.\n"
        "Theta is measured from the beam-back direction (opposite the incident "
        "vacuum ray), not from the sample normal.\n"
        f"Physical polar support is 0 to {90.0 + angle:g} degrees.\n"
        "The legacy theta CSV is a marginal distribution only. For oblique "
        "incidence use the event-level SE/BSE JointFromPlaneSampler NPZ files, "
        "which preserve E, theta, phi, and direction correlations.\n",
        encoding="utf-8",
    )
    os.replace(temporary, readme)


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_manifest(path: Path, cases: Sequence[PlaneSamplerCase]) -> None:
    _atomic_csv(
        path,
        [],
        (
            "incidence_angle_deg", "incident_energy_ev", "n_primaries",
            "case_seed", "se_emissions", "bse_emissions",
            "incoming_barrier_reflections",
            "incoming_barrier_reflection_fraction",
            "sey", "sey_sem", "bsey", "bsey_sem", "tey",
        ),
        (
            (
                _format_number(case.incidence_angle_deg),
                _format_number(case.incident_energy_ev),
                case.n_primaries,
                case.case_seed,
                case.se_energy_ev.size,
                case.bse_energy_ev.size,
                case.incoming_barrier_reflections,
                _format_number(case.incoming_barrier_reflection_fraction),
                _format_number(case.sey),
                _format_number(case.sey_sem),
                _format_number(case.bsey),
                _format_number(case.bsey_sem),
                _format_number(case.tey),
            )
            for case in sorted(
                cases, key=lambda item: (
                    item.incidence_angle_deg, item.incident_energy_ev
                )
            )
        ),
    )


def generate_plane_sampler_library(
    database_path,
    output_directory,
    *,
    material: str = "Cu",
    energies_ev: Sequence[float] = JMONSEL_ENERGIES_EV,
    incidence_angles_deg: Sequence[float] = DEFAULT_INCIDENCE_ANGLES_DEG,
    n_primaries: int = 20_000,
    probability_count: int = 513,
    config: Optional[MCConfig] = None,
    base_seed: int = 20260816,
    workers: int = 1,
    resume: bool = False,
    overwrite: bool = False,
    progress: bool = True,
    status: Optional[Callable[[str], None]] = print,
) -> list[PlaneSamplerCase]:
    """Run, checkpoint, validate, and export a complete planar library."""
    config = config or MCConfig()
    config.collect_spectra = True
    config.validate()
    energies = tuple(sorted({float(value) for value in energies_ev}))
    angles = tuple(sorted({float(value) for value in incidence_angles_deg}))
    if not energies or not angles:
        raise ValueError("energy and angle grids must both be non-empty")
    for angle in angles:
        plane_directions(angle)
    if any(energy <= config.bse_cutoff_ev for energy in energies):
        raise ValueError("every energy must exceed config.bse_cutoff_ev")
    n_primaries = int(n_primaries)
    workers = int(workers)
    if n_primaries < 1 or workers < 1:
        raise ValueError("n_primaries and workers must be positive")
    probabilities = cosine_probability_grid(probability_count)
    output = Path(output_directory)
    existing_checkpoints = list(output.glob("alpha_*deg/raw/E_*eV.npz")) \
        if output.exists() else []
    if existing_checkpoints and not (resume or overwrite):
        raise FileExistsError(
            f"{output} already contains raw checkpoints; use resume=True or "
            "overwrite=True"
        )
    output.mkdir(parents=True, exist_ok=True)
    database_digest = _file_sha256(database_path)

    cases = []
    for angle in angles:
        angle_cases = []
        for energy in energies:
            seed = deterministic_case_seed(base_seed, angle, energy)
            raw_path = checkpoint_path(output, angle, energy)
            if resume and raw_path.exists():
                if status is not None:
                    status(f"resume alpha={angle:g} deg, E={energy:g} eV")
                case = load_case_checkpoint(
                    raw_path, material=material, config=config,
                    database_sha256=database_digest,
                    expected_angle_deg=angle, expected_energy_ev=energy,
                    expected_n_primaries=n_primaries,
                    expected_case_seed=seed,
                )
            else:
                if raw_path.exists() and not overwrite:
                    raise FileExistsError(raw_path)
                if status is not None:
                    status(
                        f"run alpha={angle:g} deg, E={energy:g} eV, "
                        f"N={n_primaries}, seed={seed}"
                    )
                case = run_plane_sampler_case(
                    database_path, material, energy, angle, n_primaries,
                    config=config, case_seed=seed, workers=workers,
                    progress=progress,
                )
                save_case_checkpoint(
                    raw_path, case, material=material, config=config,
                    database_sha256=database_digest,
                )
            cases.append(case)
            angle_cases.append(case)
        angle_output = output / angle_directory_name(angle)
        export_angle_tables(
            angle_output, angle_cases,
            material=material, probabilities=probabilities,
        )

        # Aggregate the event-level v2 checkpoints into the two files consumed
        # directly by the RFA joint sampler.  No marginalisation or new random
        # azimuth is introduced here.
        angle_case_paths = [
            checkpoint_path(output, angle, case.incident_energy_ev)
            for case in angle_cases
        ]
        joint_paths = write_joint_angle_samplers(
            angle_case_paths, angle_output
        )
        if status is not None:
            status(
                f"joint alpha={angle:g} deg: "
                f"SE={joint_paths['SE'].name}, "
                f"BSE={joint_paths['BSE'].name}"
            )

    _write_manifest(output / "sampler_manifest.csv", cases)
    generation = {
        "schema": "seemc-plane-sampler-library-v1",
        "seemc_imaging_version": PACKAGE_VERSION,
        "material": material,
        "database_path_as_run": str(database_path),
        "database_sha256": database_digest,
        "energies_ev": list(energies),
        "incidence_angles_deg": list(angles),
        "n_primaries_per_case": n_primaries,
        "probability_grid": {
            "kind": "cosine-clustered",
            "count": int(probability_count),
            "formula": "r=0.5*(1-cos(pi*u)), u=linspace(0,1,count)",
        },
        "seed": {
            "base_seed": int(base_seed),
            "case_derivation": (
                "SeedSequence([base_seed, round(angle_deg*1e6), "
                "round(energy_ev*1e3)])"
            ),
            "trajectory_derivation": "SeedSequence([case_seed, 0, trajectory_id])",
        },
        "classification": {
            "se": f"emission_energy_ev < {config.bse_cutoff_ev:g}",
            "bse": f"emission_energy_ev >= {config.bse_cutoff_ev:g}",
        },
        "angle_convention": {
            "polar_axis": "beam-back direction (-incident vacuum direction)",
            "support_deg": "0 .. 90 + incidence_angle_deg",
            "phi_zero": (
                "incidence-plane direction from beam-back axis toward "
                "the sample outward normal"
            ),
            "phi_handedness": "right-handed about the beam-back axis",
            "raw_joint_direction": (
                "each emitted electron retains paired E, theta, phi, and "
                "direction cosines; no uniform-azimuth reconstruction"
            ),
        },
        "joint_sampler_files": {
            "SE": "SEJointFromPlaneSampler_uncoatedCuFPA.npz",
            "BSE": "BSEJointFromPlaneSampler_uncoatedCuFPA.npz",
            "schema": "seemc-joint-emission-v1",
        },
        "incoming_barrier_reflection": bool(
            getattr(config, "incoming_barrier_reflection", False)
        ),
        "mc_config": asdict(config),
        "mc_config_sha256": _config_sha256(config),
    }
    _atomic_json(output / "sampler_generation.json", generation)
    return cases


__all__ = [
    "BSE_ENERGY_FILENAME",
    "BSE_THETA_FILENAME",
    "BSEY_FILENAME",
    "DEFAULT_INCIDENCE_ANGLES_DEG",
    "JMONSEL_ENERGIES_EV",
    "PlaneSamplerCase",
    "SE_ENERGY_FILENAME",
    "SE_THETA_FILENAME",
    "SEY_FILENAME",
    "angle_directory_name",
    "checkpoint_path",
    "cosine_probability_grid",
    "deterministic_case_seed",
    "export_angle_tables",
    "generate_plane_sampler_library",
    "load_case_checkpoint",
    "plane_directions",
    "run_plane_sampler_case",
    "save_case_checkpoint",
]
