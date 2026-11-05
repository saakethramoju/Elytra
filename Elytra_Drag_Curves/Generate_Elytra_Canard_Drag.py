"""Generate Elytra's canard drag-increment tables and optional RocketPy combined decks.

Implements the CANARD DRAG EQUATIONS from Vehicle-Sizer/Flight/canards.py
(https://github.com/gt-space/Vehicle-Sizer/blob/main/Flight/canards.py),
using its live shock/expansion solver instead of the canardmodel.h5 interpolation
for Mach > 2.8. This avoids any need to copy canardmodel.h5 into Elytra.

The canards are a symmetric counter-rotating pair. Positive `deflection` is the
magnitude of the differential command; the pair moves +delta and -delta.

Note: Vehicle-Sizer's canard drag model is 2D. For RocketPy's full 3D drag,
we approximate its 2D body incidence by hypot(alpha,beta), clipped to 15 deg.
This is a DRAG-ONLY approximation, not a 3D canard lift/roll/yaw model.

Existing base drag tables MUST NOT already include canard drag.
"""

from __future__ import annotations

import argparse
import csv

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ellipe

# Exact Elytra canard values from flight_pressure_fed_regulator 2.yaml
OMLD_M = 0.31115
ROOT_CHORD_M = 0.25
CANARD_COUNT = 2
CD0 = 0.2
OSWALD = 0.4
DEFLECTION_DRAG_MULTIPLIER = 2.0

# This is also the main RASAero aero-model validated body-incidence limit.
MAX_BODY_INCIDENCE_DEG = 15.0
MAX_CANARD_DEFLECTION_DEG = 10.0
CANARD_LIFT_GRID_MAX_DEG = 25.0

GAMMA = 1.4
MACH_GRID = np.round(np.arange(0.1, 10.00001, 0.1), 10)
ANGLE_GRID_DEG = np.arange(-25.0, 25.00001, 0.25)


def shock_turn(mach, beta, gamma):
    return np.arctan(
        2 / np.tan(beta) * (mach**2 * np.sin(beta)**2 - 1)
        / (mach**2 * (gamma + np.cos(2 * beta)) + 2)
    )


def pm_angle(mach, gamma):
    x = np.sqrt(np.maximum(mach**2 - 1, 0))
    return (
        np.sqrt((gamma + 1) / (gamma - 1))
        * np.arctan(np.sqrt((gamma - 1) / (gamma + 1)) * x)
        - np.arctan(x)
    )


def shock_expansion_cn(mach, angle):
    """Vectorized copy of Vehicle-Sizer's reference shock-expansion solver."""
    mach, angle = np.broadcast_arrays(np.asarray(mach, float), np.asarray(angle, float))
    gamma = GAMMA
    abs_angle = np.abs(angle)
    mach2 = mach**2
    mach_angle = np.arcsin(1 / mach)
    sin_beta2 = (
        (gamma + 1) * mach2 / 4 - 1
        + np.sqrt((gamma + 1) * ((gamma + 1) * mach2**2 / 16 + (gamma - 1) * mach2 / 2 + 1))
    ) / (gamma * mach2)
    beta_max = np.arcsin(np.sqrt(np.clip(sin_beta2, 0, 1)))
    theta_max = shock_turn(mach, beta_max, gamma)
    valid = (abs_angle <= theta_max) & (mach > 1)
    low = mach_angle.copy()
    high = beta_max.copy()
    for _ in range(45):
        middle = (low + high) / 2
        lower_side = shock_turn(mach, middle, gamma) < abs_angle
        low = np.where(lower_side, middle, low)
        high = np.where(lower_side, high, middle)
    beta = (low + high) / 2
    compression = 1 + 2 * gamma / (gamma + 1) * (mach2 * np.sin(beta)**2 - 1)
    target_nu = pm_angle(mach, gamma) + abs_angle
    low = mach.copy()
    high = np.full(mach.shape, 1000.0)
    valid &= target_nu < pm_angle(high, gamma)
    for _ in range(50):
        middle = (low + high) / 2
        lower_side = pm_angle(middle, gamma) < target_nu
        low = np.where(lower_side, middle, low)
        high = np.where(lower_side, middle, high)
    expanded_mach = (low + high) / 2
    expansion = (
        (1 + (gamma - 1) * mach2 / 2) / (1 + (gamma - 1) * expanded_mach**2 / 2)
    ) ** (gamma / (gamma - 1))
    cn = np.sign(angle) * (compression - expansion) / (0.5 * gamma * mach2)
    cn = np.where(abs_angle < 1e-12, 0, cn)
    return np.where(valid, cn, np.nan)


class ElytraCanardDrag:
    def __init__(
        self,
        *,
        body_diameter_m=OMLD_M,
        root_chord_m=ROOT_CHORD_M,
        canard_count=CANARD_COUNT,
        cd0=CD0,
        oswald=OSWALD,
        deflection_drag_multiplier=DEFLECTION_DRAG_MULTIPLIER,
        model_path=None,
    ):
        diameter = float(body_diameter_m)
        root = float(root_chord_m)
        self.canard_count = int(canard_count)
        self.cd0 = float(cd0)
        self.oswald = float(oswald)
        self.deflection_drag_multiplier = float(deflection_drag_multiplier)
        if diameter <= 0 or root <= 0 or self.canard_count <= 0 or self.oswald <= 0:
            raise ValueError("Canard geometry and Oswald efficiency must be positive")
        radius = diameter / 2
        span = root / np.tan(np.deg2rad(74.0))
        self.area = 0.5 * root * span
        self.aspect_ratio = 2 * span**2 / self.area
        attached = 1 + radius / (span + radius)
        gap = max(0.0, 1 - 0.020 / span)
        self.body_factor = 1 + (attached - 1) * gap
        self.reference_area = np.pi * radius**2
        self.area_factor = 0.5 * self.canard_count * self.area / self.reference_area

        self._shock = None
        if model_path is not None:
            with h5py.File(model_path, "r") as table:
                self._shock = RegularGridInterpolator(
                    (table["mach"][:], table["angle_rad"][:]),
                    table["cn"][:],
                    bounds_error=False,
                    fill_value=np.nan,
                )

        mach, angle = np.meshgrid(MACH_GRID, np.deg2rad(ANGLE_GRID_DEG), indexing="ij")
        kp = np.interp(mach, [0.60, 1.20, 2.00, 2.36, 2.80], [1.539, 1.838, 1.484, 1.325, 1.210])
        kv = np.interp(mach, [0.60, 1.20, 2.00, 2.36, 2.80], [2.869, 1.766, 0.722, 0.709, 0.521])
        lift = kp * np.sin(angle) * np.cos(angle)**2 + kv * np.cos(angle) * np.sin(angle) * np.abs(np.sin(angle))
        high = mach > 2.8
        if np.any(high):
            mh, ah = mach[high], angle[high]
            if self._shock is None:
                hypersonic = shock_expansion_cn(mh, ah)
            else:
                hypersonic = self._shock((mh, np.abs(ah))) * np.sign(ah)
                outside = ~np.isfinite(hypersonic)
                if np.any(outside):
                    hypersonic[outside] = shock_expansion_cn(
                        mh[outside], ah[outside]
                    )
            beta = np.sqrt(mh**2 - 1)
            m = beta / np.tan(np.deg2rad(74.0))
            elliptic_e = ellipe(np.maximum(0, 1 - m**2))
            wing_factor = np.where(
                m < 1, np.pi * beta * self.aspect_ratio / (8 * elliptic_e), 1.0
            )
            hypersonic *= wing_factor * np.cos(ah)
            blend = np.clip((mh - 2.8) / 1.2, 0, 1)
            lift[high] = (1 - blend) * lift[high] + blend * hypersonic
        if not np.all(np.isfinite(lift)):
            raise ValueError("Canard solver was used outside the valid shock/expansion domain")
        self._lift = RegularGridInterpolator(
            (MACH_GRID, np.deg2rad(ANGLE_GRID_DEG)), lift,
            bounds_error=False, fill_value=None,
        )

    def delta_cd(self, mach, body_alpha_rad, deflection_rad=0.0):
        """Return drag coefficient ADDITION on Elytra body reference area."""
        mach, body_alpha_rad, deflection_rad = np.broadcast_arrays(
            np.asarray(mach, float), np.asarray(body_alpha_rad, float),
            np.asarray(deflection_rad, float),
        )
        original_shape = mach.shape
        mach = mach.ravel()
        body_alpha_rad = body_alpha_rad.ravel()
        deflection_rad = deflection_rad.ravel()
        if np.any(np.abs(body_alpha_rad) > np.deg2rad(MAX_BODY_INCIDENCE_DEG) + 1e-9):
            raise ValueError("2D canard body incidence exceeds 15 deg")
        if np.any(np.abs(deflection_rad) > np.deg2rad(MAX_CANARD_DEFLECTION_DEG) + 1e-9):
            raise ValueError("Differential canard deflection exceeds 10 deg")
        mach = np.clip(mach, MACH_GRID[0], MACH_GRID[-1])
        inc = np.linspace(0, 1, 9)[:, None] * np.abs(body_alpha_rad)[None, :]
        commands = np.broadcast_to(deflection_rad[None, :], inc.shape)
        angles = np.concatenate((commands + inc, commands - inc, inc, -inc))
        m = np.broadcast_to(mach[None, :], angles.shape)
        lift = self.body_factor * self._lift((m, angles))
        subsonic = lift**2 / (np.pi * self.oswald * self.aspect_ratio)
        supersonic = np.abs(lift * np.tan(angles))
        blend = np.clip((m - 0.8) / 0.4, 0, 1)
        coefficients = self.cd0 + (1 - blend) * subsonic + blend * supersonic
        raw = np.max(coefficients[:9] + coefficients[9:18], axis=0)
        neutral = np.max(coefficients[18:27] + coefficients[27:], axis=0)
        cd = self.area_factor * (
            neutral + self.deflection_drag_multiplier * (raw - neutral)
        )
        return cd.reshape(original_shape)

    def neutral_3d_cd(self, mach, alpha, beta):
        # Body incidence is a *3D approximation* from the sizer's planar alpha.
        body_angle = np.arccos(
            np.clip(np.cos(alpha) * np.cos(beta), -1.0, 1.0)
        )
        body_angle = np.minimum(body_angle, np.deg2rad(MAX_BODY_INCIDENCE_DEG))
        return self.delta_cd(mach, body_angle, 0.0)


def write_neutral_3d(model, directory):
    grid = np.deg2rad(np.arange(-15.0, 15.0001, 1.0))
    path = f"{directory}/Canard_Drag_Neutral.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("alpha", "beta", "mach", "cd"))
        aa, bb, mm = np.meshgrid(grid, grid, MACH_GRID, indexing="ij")
        cd = model.neutral_3d_cd(mm, aa, bb)
        writer.writerows(zip(aa.ravel(), bb.ravel(), mm.ravel(), cd.ravel()))
    return path


def write_deflection_table(model, directory):
    """Diagnostic/future controller table. NOT directly a RocketPy drag input."""
    path = f"{directory}/Canard_Drag_Deflection_Sweep.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mach", "body_alpha_deg", "canard_deflection_deg", "cd"))
        mm, aa, dd = np.meshgrid(
            MACH_GRID,
            np.arange(0., 15.001, 1.),
            np.arange(0., 10.001, 1.),
            indexing="ij",
        )
        cd = model.delta_cd(mm, np.deg2rad(aa), np.deg2rad(dd))
        writer.writerows(zip(mm.ravel(), aa.ravel(), dd.ravel(), cd.ravel()))
    return path


def combine_with_base(model, source, destination):
    """Add the sizer-derived canard Cd to any RocketPy CSV with alpha/mach columns.

    Accepts alpha,beta,mach[,other_rocketpy_independent_vars],cd; assumes
    alpha/beta in radians and the LAST column is the baseline coefficient.
    Other independent input columns are left intact. Existing base deck MUST
    not include canards, or the contribution is double-counted.
    """
    with open(source, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if not {"alpha", "mach"}.issubset(header):
            raise ValueError(f"{source} must have named 'alpha' and 'mach' columns")
        if header[-1] in {"alpha", "beta", "mach", "reynolds", "pitch_rate", "yaw_rate", "roll_rate"}:
            raise ValueError(f"{source}: last column must be Cd")
        rows = list(reader)
    alpha = np.array([float(row[header.index("alpha")]) for row in rows])
    beta = np.array([float(row[header.index("beta")]) for row in rows]) if "beta" in header else np.zeros(len(rows))
    mach = np.array([float(row[header.index("mach")]) for row in rows])
    base_cd = np.array([float(row[-1]) for row in rows])
    increment = model.neutral_3d_cd(mach, alpha, beta)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row, combined_cd in zip(rows, base_cd + increment):
            row[-1] = f"{combined_cd:.10g}"
            writer.writerow(row)
    print(f"Combined {len(rows):,} baseline Cd rows with neutral canards: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="Elytra_Drag_Curves")
    parser.add_argument("--power-on", help="existing NON-canard power-on RocketPy drag CSV")
    parser.add_argument("--power-off", help="existing NON-canard power-off RocketPy drag CSV")
    args = parser.parse_args()
    model = ElytraCanardDrag()
    neutral = write_neutral_3d(model, args.out)
    sweep = write_deflection_table(model, args.out)
    print(f"Neutral RocketPy canard increment: {neutral}")
    print(f"Canard deflection sensitivity sweep: {sweep}")
    for flag, source, dest in (
        ("power-on", args.power_on, "Drag_Power_On_With_Canards.csv"),
        ("power-off", args.power_off, "Drag_Power_Off_With_Canards.csv"),
    ):
        if source:
            combine_with_base(model, source, f"{args.out}/{dest}")
    zero = model.delta_cd(1.0, 0.0, 0.0)
    print(f"Canard neutral Cd increment at Mach 1 / 0 deg body incidence: {zero:.7f}")


if __name__ == "__main__":
    main()
