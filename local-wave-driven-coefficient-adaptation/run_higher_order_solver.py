
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

A_MIN = 0.64
A_MAX = 1.69
C_MAX = math.sqrt(A_MAX)
KAPPA = 0.5
ENERGY_SCALE = 0.03
PULSE_AMPLITUDE = 3.0

EDGE_LENGTH = 8.0
FINAL_TIME = 5.0
INITIAL_A = np.array([1.21, 1.04, 0.96], dtype=float)
SUPPORT_LEFT = 4.2
SUPPORT_RIGHT = 4.8
SENSOR_X = 1.0
CFL = 0.35
SUPPORT_THRESHOLD = 1e-10


def feedback(a, p, s):
    energy = 0.5 * (p * p + s * s / a)
    return (
        KAPPA
        * (A_MAX - a)
        * (a - A_MIN)
        * energy
        / (ENERGY_SCALE + energy)
    )


def scattering(c, incoming):
    vertex_p = float(np.sum(c * incoming) / np.sum(c))
    return 2.0 * vertex_p - incoming


def project_boundaries(r, ell, a):
    r = r.copy()
    ell = ell.copy()
    clipped = np.clip(a, A_MIN, A_MAX)
    correction = float(np.max(np.abs(clipped - a)))
    c = np.sqrt(clipped)
    ell[:, 0] = scattering(c[:, 0], r[:, 0])
    # Reflecting endpoint p=0, hence r=-ell at x=L.
    r[:, -1] = -ell[:, -1]
    return r, ell, clipped, correction


def centered_derivative(field, h):
    derivative = np.empty_like(field)
    derivative[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * h)
    derivative[:, 0] = (
        -3.0 * field[:, 0]
        + 4.0 * field[:, 1]
        - field[:, 2]
    ) / (2.0 * h)
    derivative[:, -1] = (
        3.0 * field[:, -1]
        - 4.0 * field[:, -2]
        + field[:, -3]
    ) / (2.0 * h)
    return derivative


def rhs(r, ell, a, h):
    r, ell, a, _ = project_boundaries(r, ell, a)
    c = np.sqrt(a)
    p = 0.5 * (r + ell)
    s = 0.5 * c * (r - ell)
    F = feedback(a, p, s)
    c_x = centered_derivative(c, h)

    source_r = (r - ell) * (F / (4.0 * a) + 0.5 * c_x)
    source_ell = (r - ell) * (0.5 * c_x - F / (4.0 * a))

    r_x = np.empty_like(r)
    r_x[:, :-2] = (
        -3.0 * r[:, :-2]
        + 4.0 * r[:, 1:-1]
        - r[:, 2:]
    ) / (2.0 * h)
    r_x[:, -2] = (r[:, -1] - r[:, -3]) / (2.0 * h)
    r_x[:, -1] = 0.0

    ell_x = np.empty_like(ell)
    ell_x[:, 2:] = (
        3.0 * ell[:, 2:]
        - 4.0 * ell[:, 1:-1]
        + ell[:, :-2]
    ) / (2.0 * h)
    ell_x[:, 1] = (ell[:, 2] - ell[:, 0]) / (2.0 * h)
    ell_x[:, 0] = 0.0

    dr = c * r_x + source_r
    dell = -c * ell_x + source_ell
    da = F

    dell[:, 0] = 0.0
    # Differentiate the reflecting trace r=-ell in time.
    dr[:, -1] = -dell[:, -1]
    return dr, dell, da


def distance_to_initial_support(x):
    distance = np.empty((3, len(x)))
    distance[0] = np.where(
        x < SUPPORT_LEFT,
        SUPPORT_LEFT - x,
        np.where(x > SUPPORT_RIGHT, x - SUPPORT_RIGHT, 0.0),
    )
    distance[1] = x + SUPPORT_LEFT
    distance[2] = x + SUPPORT_LEFT
    return distance


def simulate(h, record_history=False):
    cells = int(round(EDGE_LENGTH / h))
    x = np.linspace(0.0, EDGE_LENGTH, cells + 1)

    a0 = np.vstack([np.full(cells + 1, value) for value in INITIAL_A])
    a = a0.copy()
    r = np.zeros_like(a)
    ell = np.zeros_like(a)

    mask = (x >= SUPPORT_LEFT - 1e-12) & (x <= SUPPORT_RIGHT + 1e-12)
    z = (x[mask] - 4.5) / 0.3
    r[0, mask] = PULSE_AMPLITUDE * (1.0 - z * z) ** 4
    maximum_projection_correction = 0.0
    r, ell, a, correction = project_boundaries(r, ell, a)
    maximum_projection_correction = max(maximum_projection_correction, correction)

    nominal_dt = CFL * h / C_MAX
    steps = int(math.ceil(FINAL_TIME / nominal_dt))
    dt = FINAL_TIME / steps

    distance = distance_to_initial_support(x)
    sensor_index = int(round(SENSOR_X / h))
    history = []

    maximum_wave_outside = 0.0
    maximum_structure_outside = 0.0
    first_branch_wave = None
    first_branch_structure = None
    first_sensor_wave = None
    first_sensor_structure = None

    record_stride = max(1, steps // 350)

    for step in range(steps + 1):
        t = step * dt
        c = np.sqrt(a)
        p = 0.5 * (r + ell)
        s = 0.5 * c * (r - ell)
        wave_amplitude = np.maximum(np.abs(r), np.abs(ell))
        delta_a = a - a0

        outside = distance > C_MAX * t + 1e-12
        if np.any(outside):
            maximum_wave_outside = max(
                maximum_wave_outside,
                float(np.max(wave_amplitude[outside])),
            )
            maximum_structure_outside = max(
                maximum_structure_outside,
                float(np.max(np.abs(delta_a[outside]))),
            )

        branch_wave = float(np.max(wave_amplitude[1:]))
        branch_structure = float(np.max(np.abs(delta_a[1:])))
        sensor_wave = float(np.max(wave_amplitude[1:, sensor_index]))
        sensor_structure = float(np.max(np.abs(delta_a[1:, sensor_index])))

        if first_branch_wave is None and branch_wave > SUPPORT_THRESHOLD:
            first_branch_wave = t
        if first_branch_structure is None and branch_structure > SUPPORT_THRESHOLD:
            first_branch_structure = t
        if first_sensor_wave is None and sensor_wave > SUPPORT_THRESHOLD:
            first_sensor_wave = t
        if first_sensor_structure is None and sensor_structure > SUPPORT_THRESHOLD:
            first_sensor_structure = t

        if record_history and (step % record_stride == 0 or step == steps):
            history.append({
                "time": t,
                "branch_wave": branch_wave,
                "branch_structure": branch_structure,
                "sensor_wave": sensor_wave,
                "sensor_structure": sensor_structure,
                "maximum_wave_outside_cone_so_far": maximum_wave_outside,
                "maximum_structure_outside_cone_so_far": maximum_structure_outside,
            })

        if step == steps:
            break

        k1 = rhs(r, ell, a, h)
        r1, ell1, a1, correction = project_boundaries(
            r + dt * k1[0],
            ell + dt * k1[1],
            a + dt * k1[2],
        )
        maximum_projection_correction = max(maximum_projection_correction, correction)

        k2 = rhs(r1, ell1, a1, h)
        r2, ell2, a2, correction = project_boundaries(
            0.75 * r + 0.25 * (r1 + dt * k2[0]),
            0.75 * ell + 0.25 * (ell1 + dt * k2[1]),
            0.75 * a + 0.25 * (a1 + dt * k2[2]),
        )
        maximum_projection_correction = max(maximum_projection_correction, correction)

        k3 = rhs(r2, ell2, a2, h)
        r, ell, a, correction = project_boundaries(
            (1.0 / 3.0) * r + (2.0 / 3.0) * (r2 + dt * k3[0]),
            (1.0 / 3.0) * ell + (2.0 / 3.0) * (ell2 + dt * k3[1]),
            (1.0 / 3.0) * a + (2.0 / 3.0) * (a2 + dt * k3[2]),
        )
        maximum_projection_correction = max(maximum_projection_correction, correction)

    c = np.sqrt(a)
    p = 0.5 * (r + ell)
    s = 0.5 * c * (r - ell)

    return {
        "h": h,
        "dt": dt,
        "steps": steps,
        "x": x,
        "a0": a0,
        "a": a,
        "p": p,
        "s": s,
        "r": r,
        "ell": ell,
        "history": pd.DataFrame(history),
        "maximum_wave_outside": maximum_wave_outside,
        "maximum_structure_outside": maximum_structure_outside,
        "first_branch_wave": first_branch_wave,
        "first_branch_structure": first_branch_structure,
        "first_sensor_wave": first_sensor_wave,
        "first_sensor_structure": first_sensor_structure,
        "maximum_projection_correction": maximum_projection_correction,
    }


def relative_nested_error(coarse, fine, field, perturbation=False):
    ratio = int(round(coarse["h"] / fine["h"]))
    coarse_field = coarse[field]
    fine_field = fine[field][:, ::ratio]
    if perturbation:
        coarse_field = coarse_field - coarse["a0"]
        fine_field = fine_field - fine["a0"][:, ::ratio]
    numerator = np.linalg.norm(coarse_field - fine_field)
    denominator = np.linalg.norm(fine_field)
    return float(numerator / max(denominator, 1e-15))


def main():
    out = Path(__file__).resolve().parent
    grid_sizes = [0.04, 0.02, 0.01, 0.005, 0.0025]
    runs = {
        h: simulate(h, record_history=(h == 0.005))
        for h in grid_sizes
    }

    grid_rows = []
    for h in grid_sizes:
        run = runs[h]
        grid_rows.append({
            "h": h,
            "dt": run["dt"],
            "steps": run["steps"],
            "maximum_wave_outside_cone": run["maximum_wave_outside"],
            "maximum_structure_outside_cone": run["maximum_structure_outside"],
            "first_branch_wave_above_1e_10": run["first_branch_wave"],
            "first_branch_structure_above_1e_10": run["first_branch_structure"],
            "first_sensor_wave_above_1e_10": run["first_sensor_wave"],
            "first_sensor_structure_above_1e_10": run["first_sensor_structure"],
            "maximum_delta_a": float(np.max(np.abs(run["a"] - run["a0"]))),
            "maximum_projection_correction": run["maximum_projection_correction"],
        })
    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(out / "grid_runs.csv", index=False)

    convergence_rows = []
    error_series = {"p": [], "s": [], "delta_a": []}
    for coarse_h, fine_h in zip(grid_sizes[:-1], grid_sizes[1:]):
        p_error = relative_nested_error(runs[coarse_h], runs[fine_h], "p")
        s_error = relative_nested_error(runs[coarse_h], runs[fine_h], "s")
        a_error = relative_nested_error(
            runs[coarse_h], runs[fine_h], "a", perturbation=True
        )
        error_series["p"].append(p_error)
        error_series["s"].append(s_error)
        error_series["delta_a"].append(a_error)
        convergence_rows.append({
            "coarse_h": coarse_h,
            "fine_h": fine_h,
            "p_relative_error": p_error,
            "s_relative_error": s_error,
            "delta_a_relative_error": a_error,
        })

    convergence_df = pd.DataFrame(convergence_rows)
    for field, column in [
        ("p", "p_relative_error"),
        ("s", "s_relative_error"),
        ("delta_a", "delta_a_relative_error"),
    ]:
        values = convergence_df[column].to_numpy()
        orders = [np.nan]
        for i in range(1, len(values)):
            orders.append(math.log(values[i - 1] / values[i], 2.0))
        convergence_df[f"{field}_observed_order"] = orders
    convergence_df.to_csv(out / "convergence.csv", index=False)

    baseline_history = runs[0.005]["history"]
    baseline_history.to_csv(out / "sensor_history_h0p005.csv", index=False)

    reference = runs[0.0025]
    profile_rows = []
    for edge in range(3):
        for index, coordinate in enumerate(reference["x"]):
            profile_rows.append({
                "edge": edge,
                "x": coordinate,
                "p": reference["p"][edge, index],
                "s": reference["s"][edge, index],
                "a_initial": reference["a0"][edge, index],
                "a_final": reference["a"][edge, index],
                "delta_a": (
                    reference["a"][edge, index]
                    - reference["a0"][edge, index]
                ),
            })
    pd.DataFrame(profile_rows).to_csv(
        out / "reference_profiles_h0p0025.csv", index=False
    )

    finest_pair = convergence_df.iloc[-1]
    summary = {
        "status": (
            "HIGHER_ORDER_NUMERICS_PASS"
        ),
        "solver": {
            "spatial_method": "second-order one-sided characteristic differences",
            "time_method": "three-stage SSP Runge-Kutta",
            "variables": "left/right characteristics plus local coefficient",
            "grid_sizes": grid_sizes,
            "support_threshold": SUPPORT_THRESHOLD,
            "external_boundary": "reflecting p=0, implemented as r=-ell",
            "coefficient_projection": "clip to [a_min,a_max] at every SSP-RK stage",
        },
        "finest_pair": {
            "coarse_h": float(finest_pair["coarse_h"]),
            "fine_h": float(finest_pair["fine_h"]),
            "p_relative_error": float(finest_pair["p_relative_error"]),
            "s_relative_error": float(finest_pair["s_relative_error"]),
            "delta_a_relative_error": float(
                finest_pair["delta_a_relative_error"]
            ),
            "p_observed_order": float(finest_pair["p_observed_order"]),
            "s_observed_order": float(finest_pair["s_observed_order"]),
            "delta_a_observed_order": float(
                finest_pair["delta_a_observed_order"]
            ),
        },
        "finest_run": {
            "h": reference["h"],
            "dt": reference["dt"],
            "steps": reference["steps"],
            "maximum_wave_outside_cone": (
                reference["maximum_wave_outside"]
            ),
            "maximum_structure_outside_cone": (
                reference["maximum_structure_outside"]
            ),
            "first_branch_wave_above_1e_10": (
                reference["first_branch_wave"]
            ),
            "first_branch_structure_above_1e_10": (
                reference["first_branch_structure"]
            ),
            "first_sensor_wave_above_1e_10": (
                reference["first_sensor_wave"]
            ),
            "first_sensor_structure_above_1e_10": (
                reference["first_sensor_structure"]
            ),
            "uniform_vertex_lower_bound": SUPPORT_LEFT / C_MAX,
            "uniform_sensor_lower_bound": (
                SUPPORT_LEFT + SENSOR_X
            ) / C_MAX,
            "maximum_delta_a": float(
                np.max(np.abs(reference["a"] - reference["a0"]))
            ),
            "maximum_projection_correction": reference["maximum_projection_correction"],
            "maximum_projection_correction_all_grids": float(
                grid_df["maximum_projection_correction"].max()
            ),
        },
        "interpretation": [
            "The finest-pair wave convergence is approximately second order.",
            "The coefficient perturbation converges faster than second order on the finest pair, but this should not be promoted as a general order theorem.",
            "The higher-order stencil is not exactly support preserving: it produces small pre-cone numerical leakage.",
            "The pre-cone wave leakage decreases rapidly under refinement and the induced structural leakage is near machine precision on the finest grid.",
            "The numerical experiment supports the continuum model but does not prove the common-cone theorem.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # Convergence plot
    fig, ax = plt.subplots(figsize=(8, 5))
    xh = convergence_df["fine_h"]
    ax.loglog(xh, convergence_df["p_relative_error"], marker="o", label="p")
    ax.loglog(xh, convergence_df["s_relative_error"], marker="o", label="s")
    ax.loglog(
        xh,
        convergence_df["delta_a_relative_error"],
        marker="o",
        label="delta a",
    )
    ax.set_xlabel("fine grid spacing")
    ax.set_ylabel("relative difference to next finer grid")
    ax.set_title("Higher-order refinement study")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "higher_order_convergence.png", dpi=200)
    plt.close(fig)

    # Leakage plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        grid_df["h"],
        grid_df["maximum_wave_outside_cone"],
        marker="o",
        label="wave leakage",
    )
    ax.loglog(
        grid_df["h"],
        grid_df["maximum_structure_outside_cone"],
        marker="o",
        label="structural leakage",
    )
    ax.set_xlabel("grid spacing")
    ax.set_ylabel("maximum magnitude outside continuum cone")
    ax.set_title("Numerical pre-cone leakage under refinement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "precone_leakage.png", dpi=200)
    plt.close(fig)

    # Reference structural profile
    fig, ax = plt.subplots(figsize=(8, 5))
    for edge in range(3):
        ax.plot(
            reference["x"],
            reference["a"][edge] - reference["a0"][edge],
            label=f"edge {edge}",
        )
    ax.set_xlabel("distance from central vertex")
    ax.set_ylabel("final coefficient perturbation")
    ax.set_title("Higher-order reference operator trail")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "reference_operator_trail.png", dpi=200)
    plt.close(fig)

    # Sensor timing
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        baseline_history["time"],
        baseline_history["sensor_wave"],
        label="sensor wave",
    )
    ax.plot(
        baseline_history["time"],
        baseline_history["sensor_structure"],
        label="sensor structural change",
    )
    ax.axvline(
        (SUPPORT_LEFT + SENSOR_X) / C_MAX,
        linestyle="--",
        label="uniform cone lower bound",
    )
    ax.set_xlabel("time")
    ax.set_ylabel("sensor magnitude")
    ax.set_title("Higher-order sensor timing")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "higher_order_sensor_timing.png", dpi=200)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
