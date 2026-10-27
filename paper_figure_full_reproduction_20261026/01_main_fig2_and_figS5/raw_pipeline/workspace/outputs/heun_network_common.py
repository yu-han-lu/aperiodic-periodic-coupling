"""Shared full-network Heun primitives for the deletion reruns.

The predictor and corrector use the same registered forcing sample.  All 110
threshold-linear drives are recomputed independently at both Heun stages.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=False)
def threshold_network_heun_batch_fixedpoint(
    inputs: np.ndarray,
    local_ee: np.ndarray,
    local_ei: np.ndarray,
    local_ie: np.ndarray,
    local_ii: np.ndarray,
    edge_target: np.ndarray,
    edge_source: np.ndarray,
    edge_e: np.ndarray,
    edge_i: np.ndarray,
    background_e: np.ndarray,
    background_i: np.ndarray,
    initial_e: np.ndarray,
    initial_i: np.ndarray,
    dt_ms: float,
    tau_e_ms: float,
    tau_i_ms: float,
    burn_steps: int,
    output_indices: np.ndarray,
    input_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the complete threshold-linear network with explicit Heun."""

    n_trials, n_steps = inputs.shape
    n_area = len(local_ee)
    n_outputs = len(output_indices)
    output = np.empty(
        (n_trials, n_outputs, n_steps - burn_steps), dtype=np.float32
    )
    rectification_stage1 = np.zeros(n_trials, dtype=np.int64)
    rectification_stage2 = np.zeros(n_trials, dtype=np.int64)

    for trial in prange(n_trials):
        rate_e = initial_e.copy()
        rate_i = initial_i.copy()
        predictor_e = np.empty(n_area)
        predictor_i = np.empty(n_area)
        k1_e = np.empty(n_area)
        k1_i = np.empty(n_area)
        long_e = np.empty(n_area)
        long_i = np.empty(n_area)
        out_pos = 0
        rect1 = 0
        rect2 = 0

        for step in range(n_steps):
            long_e[:] = 0.0
            long_i[:] = 0.0
            for edge in range(len(edge_target)):
                target = edge_target[edge]
                source = edge_source[edge]
                source_rate = rate_e[source]
                long_e[target] += edge_e[edge] * source_rate
                long_i[target] += edge_i[edge] * source_rate

            for area in range(n_area):
                drive_e = (
                    local_ee[area] * rate_e[area]
                    + local_ei[area] * rate_i[area]
                    + long_e[area]
                    + background_e[area]
                )
                if area == input_idx:
                    drive_e += inputs[trial, step]
                drive_i = (
                    local_ie[area] * rate_e[area]
                    + local_ii[area] * rate_i[area]
                    + long_i[area]
                    + background_i[area]
                )
                if drive_e < 0.0:
                    drive_e = 0.0
                    rect1 += 1
                if drive_i < 0.0:
                    drive_i = 0.0
                    rect1 += 1
                k1_e[area] = (-rate_e[area] + drive_e) / tau_e_ms
                k1_i[area] = (-rate_i[area] + drive_i) / tau_i_ms
                predictor_e[area] = rate_e[area] + dt_ms * k1_e[area]
                predictor_i[area] = rate_i[area] + dt_ms * k1_i[area]

            long_e[:] = 0.0
            long_i[:] = 0.0
            for edge in range(len(edge_target)):
                target = edge_target[edge]
                source = edge_source[edge]
                source_rate = predictor_e[source]
                long_e[target] += edge_e[edge] * source_rate
                long_i[target] += edge_i[edge] * source_rate

            for area in range(n_area):
                drive_e = (
                    local_ee[area] * predictor_e[area]
                    + local_ei[area] * predictor_i[area]
                    + long_e[area]
                    + background_e[area]
                )
                if area == input_idx:
                    drive_e += inputs[trial, step]
                drive_i = (
                    local_ie[area] * predictor_e[area]
                    + local_ii[area] * predictor_i[area]
                    + long_i[area]
                    + background_i[area]
                )
                if drive_e < 0.0:
                    drive_e = 0.0
                    rect2 += 1
                if drive_i < 0.0:
                    drive_i = 0.0
                    rect2 += 1
                k2_e = (-predictor_e[area] + drive_e) / tau_e_ms
                k2_i = (-predictor_i[area] + drive_i) / tau_i_ms
                rate_e[area] += 0.5 * dt_ms * (k1_e[area] + k2_e)
                rate_i[area] += 0.5 * dt_ms * (k1_i[area] + k2_i)

            if step >= burn_steps:
                for output_pos in range(n_outputs):
                    output[trial, output_pos, out_pos] = (
                        rate_e[output_indices[output_pos]]
                        - initial_e[output_indices[output_pos]]
                    )
                out_pos += 1

        rectification_stage1[trial] = rect1
        rectification_stage2[trial] = rect2

    return output, rectification_stage1, rectification_stage2


def heun_recurrence(
    jacobian: np.ndarray,
    input_idx: int,
    tau_e_ms: float,
    dt_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return A and b for the active-branch explicit-Heun recurrence."""

    eye = np.eye(len(jacobian))
    forcing = np.zeros(len(jacobian))
    forcing[input_idx] = 1.0 / tau_e_ms
    a = eye + dt_ms * jacobian + 0.5 * dt_ms**2 * (jacobian @ jacobian)
    b = dt_ms * forcing + 0.5 * dt_ms**2 * (jacobian @ forcing)
    return a, b


def heun_discrete_transfer_all(
    jacobian: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    n_steps: int,
    tau_e_ms: float,
    dt_ms: float,
) -> np.ndarray:
    """Frequency response of the exact active-branch Heun recurrence."""

    a, b = heun_recurrence(jacobian, input_idx, tau_e_ms, dt_ms)
    eigenvalues, vectors = np.linalg.eig(a)
    coefficients = np.linalg.solve(vectors, b)
    readout = vectors[np.asarray(output_indices), :]
    result = np.empty((len(output_indices), n_steps // 2 + 1), complex)
    block = 8192
    for start in range(0, result.shape[1], block):
        stop = min(start + block, result.shape[1])
        theta = 2.0 * np.pi * np.arange(start, stop) / n_steps
        denominator = 1.0 / (
            1.0 - np.exp(-1j * theta[:, None]) * eigenvalues[None, :]
        )
        result[:, start:stop] = np.einsum(
            "fk,ok,k->of", denominator, readout, coefficients, optimize=True
        )
    return result
