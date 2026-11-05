"""Build RocketPy drag CSVs from the same aerodynamic decks as Vehicle-Sizer."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.constants import inch


ELYTRA_DIRECTORY = str(Path(__file__).resolve().parents[1])
DRAG_DIRECTORY = str(Path(__file__).resolve().parent)

# Allow both direct script execution and an import from Elytra.py.
if ELYTRA_DIRECTORY not in sys.path:
    sys.path.insert(0, ELYTRA_DIRECTORY)

from ELYTRA_DATA import (  # noqa: E402
    BOATTAIL_AFT_DIAMETER,
    BOATTAIL_LENGTH,
    CANARD_ROOT_CHORD,
    FIN_COUNT,
    FIN_ROOT_CHORD,
    FIN_SPAN,
    FIN_SWEEP_LENGTH,
    FIN_THICKNESS,
    FIN_TIP_CHORD,
    NOSECONE_LENGTH,
    NOSECONE_TYPE,
    NOZZLE_EXIT_AREA,
    VEHICLE_DIAMETER,
    VEHICLE_LENGTH,
)
from Elytra_Drag_Curves.Generate_Elytra_Canard_Drag import (  # noqa: E402
    ElytraCanardDrag,
)
from Elytra_Drag_Curves.dragmodel import DragModel  # noqa: E402


ANGLE_LIMIT_DEG = 15.0
ANGLE_STEP_DEG = 1.0
DEFAULT_FINISH = "30um"
DEFAULT_DRAG_MULTIPLIER = 1.1


def _vehicle_candidate() -> dict[str, float]:
    """Translate ELYTRA_DATA's SI geometry to Vehicle-Sizer's inch inputs."""
    chord_difference = FIN_ROOT_CHORD - FIN_TIP_CHORD
    if chord_difference <= 0.0:
        raise ValueError("FIN_ROOT_CHORD must be greater than FIN_TIP_CHORD")
    return {
        "omld": VEHICLE_DIAMETER / inch,
        "length": VEHICLE_LENGTH / inch,
        "fineness": NOSECONE_LENGTH / VEHICLE_DIAMETER,
        "exit": 2.0 * np.sqrt(NOZZLE_EXIT_AREA / np.pi) / inch,
        "boattail_aft": BOATTAIL_AFT_DIAMETER / inch,
        "boattail_length": BOATTAIL_LENGTH / inch,
        "span": FIN_SPAN / inch,
        "root": FIN_ROOT_CHORD / inch,
        "tip": FIN_TIP_CHORD / inch,
        "sweep_fraction": FIN_SWEEP_LENGTH / chord_difference,
        "thickness": FIN_THICKNESS / inch,
    }


def _nose_name() -> str:
    name = NOSECONE_TYPE.lower().replace("_", "").replace("-", "").replace(" ", "")
    aliases = {
        "vonkarman": "vonkarman",
        "ogive": "ogive",
        "conical": "conical",
        "cone": "conical",
    }
    try:
        return aliases[name]
    except KeyError as error:
        raise ValueError(f"Unsupported Vehicle-Sizer nose shape: {NOSECONE_TYPE!r}") from error


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    *,
    drag_model_filename: str = f"{DRAG_DIRECTORY}/dragmodel.h5",
    canard_model_filename: str = f"{DRAG_DIRECTORY}/canardmodel.h5",
    output_directory: str = DRAG_DIRECTORY,
    finish: str = DEFAULT_FINISH,
    drag_multiplier: float = DEFAULT_DRAG_MULTIPLIER,
) -> dict[str, str]:
    """Regenerate the powered and unpowered RocketPy drag tables.

    Vehicle geometry is never extrapolated. ``DragModel.check`` applies the
    same bounds enforcement used by Vehicle-Sizer and raises if any geometry
    value lies outside the uploaded deck.
    """
    drag_model_path = drag_model_filename
    canard_model_path = canard_model_filename
    output_path = output_directory

    if FIN_COUNT != 4:
        raise ValueError(
            f"The Vehicle-Sizer RASAero deck assumes four fins; FIN_COUNT is {FIN_COUNT:g}"
        )
    if not np.isfinite(drag_multiplier) or drag_multiplier < 0.0:
        raise ValueError("drag_multiplier must be finite and nonnegative")

    model = DragModel(drag_model_path)
    candidate = _vehicle_candidate()
    model.check(candidate)

    # RocketPy's multidimensional CSV format uses radians for alpha and beta.
    # Use Vehicle-Sizer's Mach knots, with Mach zero held at its Mach-0.1
    # coefficient exactly as Vehicle-Sizer does while dynamic pressure tends to zero.
    angles = np.deg2rad(
        np.arange(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG + ANGLE_STEP_DEG / 2, ANGLE_STEP_DEG)
    )
    mach_values = np.concatenate(([0.0], np.asarray(model.mach, dtype=float)))
    alpha, beta, mach = np.meshgrid(angles, angles, mach_values, indexing="ij")

    # This is the angle between the rocket axis and the air-relative velocity.
    # Only flight incidence is limited to the deck's documented 15-degree range;
    # geometry itself is strictly bounds-checked and never extrapolated.
    incidence_rad = np.arccos(
        np.clip(np.cos(alpha) * np.cos(beta), -1.0, 1.0)
    )
    incidence_deg = np.minimum(np.rad2deg(incidence_rad), ANGLE_LIMIT_DEG)
    unique_incidence, incidence_index = np.unique(
        np.round(incidence_deg.ravel(), 10), return_inverse=True
    )

    canards = ElytraCanardDrag(
        body_diameter_m=VEHICLE_DIAMETER,
        root_chord_m=CANARD_ROOT_CHORD,
        model_path=canard_model_path,
    )
    neutral_canard_cd = canards.neutral_3d_cd(
        mach.ravel(), alpha.ravel(), beta.ravel()
    )

    os.makedirs(output_path, exist_ok=True)
    result: dict[str, str] = {}
    clamped_mach = np.maximum(mach_values, float(model.mach[0]))
    mach_index = np.tile(np.arange(len(mach_values)), len(angles) ** 2)
    nose = _nose_name()

    for power_on, filename in (
        (False, "Drag_Power_Off.csv"),
        (True, "Drag_Power_On.csv"),
    ):
        # DragModel.cd performs the same PCHIP interpolation in alpha and linear
        # interpolation in Mach used by Vehicle-Sizer's flight model.
        curves = np.asarray(
            [
                model.cd(
                    candidate,
                    mach=clamped_mach,
                    alpha=float(angle),
                    nose=nose,
                    finish=finish,
                    power_on=power_on,
                )
                for angle in unique_incidence
            ]
        )
        base_cd = curves[incidence_index, mach_index]
        cd = drag_multiplier * base_cd + neutral_canard_cd

        if not np.all(np.isfinite(cd)) or np.any(cd <= 0.0):
            raise ValueError(f"Generated invalid coefficients for {filename}")

        frame = pd.DataFrame(
            {
                "alpha": alpha.ravel(),
                "beta": beta.ravel(),
                "mach": mach.ravel(),
                "cd": cd,
            }
        )
        destination = f"{output_path}/{filename}"
        temporary = f"{destination}.tmp"
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
        result[filename] = destination
        print(
            f"{filename}: {len(frame)} rows, Cd {cd.min():.6f} to {cd.max():.6f}"
        )

    metadata = {
        "vehicle_candidate_inches": candidate,
        "nose": nose,
        "finish": finish,
        "drag_multiplier": drag_multiplier,
        "fin_count": FIN_COUNT,
        "neutral_canards": {
            "count": canards.canard_count,
            "root_chord_m": CANARD_ROOT_CHORD,
            "body_diameter_m": VEHICLE_DIAMETER,
            "deflection_deg": 0.0,
        },
        "drag_model": {
            "filename": drag_model_path.rsplit("/", 1)[-1],
            "sha256": _sha256(drag_model_path),
            "info": model.info,
        },
        "canard_model": {
            "filename": canard_model_path.rsplit("/", 1)[-1],
            "sha256": _sha256(canard_model_path),
        },
        "incidence_limit_deg": ANGLE_LIMIT_DEG,
        "incidence_step_deg": ANGLE_STEP_DEG,
    }
    metadata_path = f"{output_path}/Drag_Curve_Metadata.json"
    temporary_metadata = f"{metadata_path}.tmp"
    with open(temporary_metadata, "w", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_path)
    result["Drag_Curve_Metadata.json"] = metadata_path

    return result


if __name__ == "__main__":
    build()
