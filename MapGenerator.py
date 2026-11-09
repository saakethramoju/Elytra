"""Generate the chamber and fluid-property maps used by Elytra_Prop_System.py.

Each enabled map is written to a separate group in FILENAME. Regenerating one
map replaces only that group; other groups in the HDF5 file are preserved.
"""

from pathlib import Path
from scipy.constants import psi
import fullplot as fplt
from thermoprop import Fluid, Propellant, Reactants, Equilibrium
from ELYTRA_DATA import (
    LOX_INITIAL_PROPELLANT_TEMPERATURE,
    FUEL_INITIAL_PROPELLANT_TEMPERATURE,
)


# ---- Maps to Generate ---- #
COMBUSTION_MAP = True
OXYGEN_MAP = True
FUEL_MAP = True
PRESSURANT_MAP = True

# ---- Output File and Group Names ---- #
FILENAME = str(Path(__file__).resolve().parent / "VEHICLE_MAPS.h5")

COMBUSTION_GROUP = "chamber"
OXYGEN_GROUP = "oxygen"
FUEL_GROUP = "fuel"
PRESSURANT_GROUP = "pressurant"

# ---- Fluids ---- #
PRESSURANT = "GN2"
FUEL = "RP-1"
OX = "LOX"

# ---- Combustion Map Settings ---- #
CHAMBER_PRESSURE_MIN = 50.0 * psi
CHAMBER_PRESSURE_MAX = 500.0 * psi
CHAMBER_PRESSURE_POINTS = 46

MIXTURE_RATIO_MIN = 0.5
MIXTURE_RATIO_MAX = 4.0
MIXTURE_RATIO_POINTS = 36

LOX_TEMPERATURE = LOX_INITIAL_PROPELLANT_TEMPERATURE
FUEL_TEMPERATURE = FUEL_INITIAL_PROPELLANT_TEMPERATURE

# ---- Pressurant (GN2) Map Settings ---- #
PRESSURANT_PRESSURE_MIN = 10.0 * psi
PRESSURANT_PRESSURE_MAX = 5000.0 * psi
PRESSURANT_PRESSURE_POINTS = 101
PRESSURANT_TEMPERATURE_MIN = 175.0
PRESSURANT_TEMPERATURE_MAX = 325.0
PRESSURANT_TEMPERATURE_POINTS = 61

# ---- Oxygen (LOX) Map Settings ---- #
OXYGEN_PRESSURE_MIN = 5.0 * psi
OXYGEN_PRESSURE_MAX = 1000.0 * psi
OXYGEN_PRESSURE_POINTS = 121
OXYGEN_TEMPERATURE_MIN = 85.0
OXYGEN_TEMPERATURE_MAX = 115.0
OXYGEN_TEMPERATURE_POINTS = 31

# ---- Fuel (RP-1) Map Settings ---- #
FUEL_PRESSURE_MIN = 5.0 * psi
FUEL_PRESSURE_MAX = 1000.0 * psi
FUEL_PRESSURE_POINTS = 81
FUEL_TEMPERATURE_MIN = 270.0
FUEL_TEMPERATURE_MAX = 330.0
FUEL_TEMPERATURE_POINTS = 31


# ---- Property Evaluation ---- #
ChamberFuel = Propellant(FUEL, temperature=FUEL_TEMPERATURE)
ChamberOx = Propellant(OX, temperature=LOX_TEMPERATURE)


def ChamberProperties(chamber_pressure, mixture_ratio):
    reactants = Reactants(
        fuels=ChamberFuel,
        oxidizers=ChamberOx,
        mixture_ratio=mixture_ratio,
    )
    gas = Equilibrium(reactants=reactants, pressure=chamber_pressure)
    return {
        "temperature": gas.temperature,
        "gamma": gas.gamma,
        "gamma_s": gas.gamma_s,
        "gas_constant": gas.gas_constant,
        "density": gas.density,
        "enthalpy": gas.enthalpy,
        "internal_energy": gas.internal_energy,
    }


def PressurantProperties(pressure, temperature):
    gas = Fluid(PRESSURANT, pressure=pressure, temperature=temperature)
    return {
        "density": gas.density,
        "enthalpy": gas.enthalpy,
        "internal_energy": gas.internal_energy,
        "gas_constant": gas.gas_constant,
        "specific_heat_ratio": gas.specific_heat_ratio,
        "gamma": gas.gamma,
    }


def OxygenProperties(pressure, temperature):
    oxygen = Fluid(OX, pressure=pressure, temperature=temperature)
    return {"density": oxygen.density}


def FuelProperties(pressure, temperature):
    fuel = Propellant(FUEL, pressure=pressure, temperature=temperature)
    return {"density": fuel.density}


# ---- Generate Selected Maps ---- #
def GenerateMap(group, axes, outputs, evaluate):
    print(f"Generating {FILENAME}/{group}...", flush=True)
    fplt.generate_map(
        FILENAME,
        group=group,
        axes=axes,
        outputs=outputs,
        evaluate=evaluate,
        overwrite=True,
        raise_errors=True,
    )


if __name__ == "__main__":
    enabled = [
        group for use_map, group in (
            (COMBUSTION_MAP, COMBUSTION_GROUP),
            (OXYGEN_MAP, OXYGEN_GROUP),
            (FUEL_MAP, FUEL_GROUP),
            (PRESSURANT_MAP, PRESSURANT_GROUP),
        ) if use_map
    ]
    if len(enabled) != len(set(enabled)):
        raise ValueError("Each enabled map must have a unique HDF5 group name.")

    if COMBUSTION_MAP:
        GenerateMap(
            COMBUSTION_GROUP,
            axes=[
                fplt.Axis.linear("chamber_pressure", start=CHAMBER_PRESSURE_MIN, stop=CHAMBER_PRESSURE_MAX, count=CHAMBER_PRESSURE_POINTS, units="Pa"),
                fplt.Axis.linear("mixture_ratio", start=MIXTURE_RATIO_MIN, stop=MIXTURE_RATIO_MAX, count=MIXTURE_RATIO_POINTS),
            ],
            outputs=["temperature", "gamma", "gamma_s", "gas_constant", "density", "enthalpy", "internal_energy"],
            evaluate=ChamberProperties,
        )

    if OXYGEN_MAP:
        GenerateMap(
            OXYGEN_GROUP,
            axes=[
                fplt.Axis.log("pressure", start=OXYGEN_PRESSURE_MIN, stop=OXYGEN_PRESSURE_MAX, count=OXYGEN_PRESSURE_POINTS, units="Pa"),
                fplt.Axis.linear("temperature", start=OXYGEN_TEMPERATURE_MIN, stop=OXYGEN_TEMPERATURE_MAX, count=OXYGEN_TEMPERATURE_POINTS, units="K"),
            ],
            outputs=["density"],
            evaluate=OxygenProperties,
        )

    if FUEL_MAP:
        GenerateMap(
            FUEL_GROUP,
            axes=[
                fplt.Axis.log("pressure", start=FUEL_PRESSURE_MIN, stop=FUEL_PRESSURE_MAX, count=FUEL_PRESSURE_POINTS, units="Pa"),
                fplt.Axis.linear("temperature", start=FUEL_TEMPERATURE_MIN, stop=FUEL_TEMPERATURE_MAX, count=FUEL_TEMPERATURE_POINTS, units="K"),
            ],
            outputs=["density"],
            evaluate=FuelProperties,
        )

    if PRESSURANT_MAP:
        GenerateMap(
            PRESSURANT_GROUP,
            axes=[
                fplt.Axis.log("pressure", start=PRESSURANT_PRESSURE_MIN, stop=PRESSURANT_PRESSURE_MAX, count=PRESSURANT_PRESSURE_POINTS, units="Pa"),
                fplt.Axis.linear("temperature", start=PRESSURANT_TEMPERATURE_MIN, stop=PRESSURANT_TEMPERATURE_MAX, count=PRESSURANT_TEMPERATURE_POINTS, units="K"),
            ],
            outputs=["density", "enthalpy", "internal_energy", "gas_constant", "specific_heat_ratio", "gamma"],
            evaluate=PressurantProperties,
        )
