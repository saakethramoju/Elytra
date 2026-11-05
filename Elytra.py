from pathlib import Path
import numpy as np
import thermoprop
import ussa1976
import fullplot as fplt
from scipy.constants import liter
from rocketpy import (
    Environment,
    CylindricalTank,
    Fluid,
    MassBasedTank,
    LiquidMotor,
    Rocket,
    Flight
)
from Elytra_Prop_System import ElytraPropSystem
from Elytra_Chamber_Map import ElytraChamberMap
from Elytra_Export import ExportRocketPyCurves, ExportFlightResults
from ELYTRA_DATA import *
from Elytra_Canted_Flight import CantedThrustFlight
from Elytra_Drag_Curves.Build_Elytra_Drag import build as BuildElytraDragCurves


ELYTRA_DIRECTORY = str(Path(__file__).resolve().parent)



# ---- Flight Sim Settings ---- #
RUN_FLIGHT                  = True
MAX_TIMESTEP                = 0.1
MAX_TIME                    = 600
VERBOSE                     = True
TERMINATE_ON_APOGEE         = True
ROTATING_EARTH              = False
WIND_ACTIVE                 = True
WIND_U_FILENAME             = f"{ELYTRA_DIRECTORY}/Wind_Profiles/Sizer_Wind_U.csv"
WIND_V_FILENAME             = f"{ELYTRA_DIRECTORY}/Wind_Profiles/Sizer_Wind_V.csv"
THRUST_MISALIGNMENT_ACTIVE  = False
EXPORT_RESULTS              = True
RESULTS_FILENAME            = f"{ELYTRA_DIRECTORY}/Elytra_6DOF_Results.h5"


# ---- Aerodynamic Settings ---- #
GENERATE_DRAG_CURVES        = True
DRAG_CURVE_DIRECTORY        = f"{ELYTRA_DIRECTORY}/Elytra_Drag_Curves"
DRAG_MODEL_FILENAME         = f"{DRAG_CURVE_DIRECTORY}/dragmodel.h5"
CANARD_MODEL_FILENAME       = f"{DRAG_CURVE_DIRECTORY}/canardmodel.h5"
DRAG_FINISH                 = "30um"
DRAG_MULTIPLIER             = 1.1
CD_POWER_ON                 = f"{DRAG_CURVE_DIRECTORY}/Drag_Power_On.csv"
CD_POWER_OFF                = f"{DRAG_CURVE_DIRECTORY}/Drag_Power_Off.csv"


# ---- Prop System Settings ---- #
PROP_GENERATE_CHAMBER_MAP   = False
PROP_CHAMBER_MAP_FILENAME   = f"{ELYTRA_DIRECTORY}/Elytra_Chamber_Map.h5"
PROP_SOLVE_PROP_SYSTEM      = False
PROP_FILENAME               = f"{ELYTRA_DIRECTORY}/Elytra_Prop_System.h5"
PROP_GENERATE_CURVES        = True
PROP_CURVE_DIRECTORY        = f"{ELYTRA_DIRECTORY}/Elytra_Prop_Curves"
PROP_DT                     = 0.1
PROP_T_FINAL                = 60
PROP_BANG_BANG              = True
PROP_SWITCH_VALVES          = True
PROP_TANK_EMPTY_VOLUME      = 0.1 * liter
PROP_VERBOSE                = True
PROP_STATISTICS             = True


LOX_THROTTLE_TIME = 5.0
FUEL_THROTTLE_TIME = 5.0
LOX_THROTTLE = 0.5
FUEL_THROTTLE = 0.5

PROP_LOX_TANK_PRESSURE_TARGET = fplt.Trace(
    x=[0.0, LOX_THROTTLE_TIME],
    y=[LOX_TANK_PRESSURE, LOX_TANK_PRESSURE * LOX_THROTTLE],
    name="LOX Tank Pressure Target",
    role="command",
    attrs={"method": "previous"},
)

PROP_FUEL_TANK_PRESSURE_TARGET = fplt.Trace(
    x=[0.0, FUEL_THROTTLE_TIME],
    y=[FUEL_TANK_PRESSURE, FUEL_TANK_PRESSURE * FUEL_THROTTLE],
    name="Fuel Tank Pressure Target",
    role="command",
    attrs={"method": "previous"},
)



# ---- Aerodynamic Curves ---- #
if GENERATE_DRAG_CURVES:

    print("Generating Vehicle-Sizer aerodynamic curves...")

    BuildElytraDragCurves(
        drag_model_filename=DRAG_MODEL_FILENAME,
        canard_model_filename=CANARD_MODEL_FILENAME,
        output_directory=DRAG_CURVE_DIRECTORY,
        finish=DRAG_FINISH,
        drag_multiplier=DRAG_MULTIPLIER,
    )




# ---- Prop System ---- #
if PROP_GENERATE_CHAMBER_MAP or not Path(PROP_CHAMBER_MAP_FILENAME).is_file():

    print("Generating chamber Pc/MR map...")

    ElytraChamberMap(
        filename=PROP_CHAMBER_MAP_FILENAME,
        fuel_temperature=FUEL_INITIAL_PROPELLANT_TEMPERATURE,
        lox_temperature=LOX_INITIAL_PROPELLANT_TEMPERATURE,
    )




if PROP_SOLVE_PROP_SYSTEM or not Path(PROP_FILENAME).is_file():

    print("Solving propulsion system...")

    ElytraPropSystem(
        dt=PROP_DT,
        t_final=PROP_T_FINAL,
        filename=PROP_FILENAME,
        chamber_map_filename=PROP_CHAMBER_MAP_FILENAME,
        lox_tank_pressure_target=PROP_LOX_TANK_PRESSURE_TARGET,
        fuel_tank_pressure_target=PROP_FUEL_TANK_PRESSURE_TARGET,
        bang_bang=PROP_BANG_BANG,
        switch_valves=PROP_SWITCH_VALVES,
        tank_empty_volume=PROP_TANK_EMPTY_VOLUME,
        verbose=PROP_VERBOSE,
        statistics=PROP_STATISTICS,
    )




if PROP_GENERATE_CURVES:

    print("Generating propulsion system curves...")

    ExportRocketPyCurves(
        filename=PROP_FILENAME,
        output_directory=PROP_CURVE_DIRECTORY,
        tank_empty_volume=PROP_TANK_EMPTY_VOLUME,
    )




# ---- Launch Conditions ---- #
LaunchConditions = Environment(
    date=(int(YEAR), int(MONTH), int(DAY), int(HOUR)),
    latitude=LATITUDE,
    longitude=LONGITUDE,
    elevation=LAUNCH_ELEVATION,
    timezone=TIMEZONE,
    datum="WGS84",
)

if not ROTATING_EARTH:
    LaunchConditions.earth_rotation_vector = [0.0, 0.0, 0.0]


'''
if WIND_ACTIVE:
    LaunchConditions.set_atmospheric_model(
        type="custom_atmosphere",
        wind_u=WIND_U_FILENAME,
        wind_v=WIND_V_FILENAME,
    )
'''


# ---- Atmospheric Model ---- #
Atmosphere = ussa1976.compute(
    z=np.arange(0, 200001, 500),
    variables=["p", "t"],
)

LaunchConditions.set_atmospheric_model(
    type="custom_atmosphere",
    pressure=np.column_stack((Atmosphere.z.values, Atmosphere.p.values)),
    temperature=np.column_stack((Atmosphere.z.values, Atmosphere.t.values)),
    wind_u=WIND_U_FILENAME if WIND_ACTIVE else 0.0,
    wind_v=WIND_V_FILENAME if WIND_ACTIVE else 0.0,
)


# ---- Motor Definition ---- #
BURN_TIME = np.loadtxt(f"{PROP_CURVE_DIRECTORY}/Thrust.csv", delimiter=",", skiprows=1)[-1, 0]

def GN2Density(T, P):
    GN2 = thermoprop.Fluid(
        fluid="Nitrogen",
        temperature=T,
        pressure=P,
    )
    return GN2.density

def LOXDensity(T, P):
    LOX = thermoprop.Fluid(
        fluid="LOX",
        temperature=LOX_INITIAL_PROPELLANT_TEMPERATURE,
        pressure=P,
    )
    return LOX.density


def FuelDensity(T, P):
    RP1 = thermoprop.Propellant(
        "RP-1",
        temperature=FUEL_INITIAL_PROPELLANT_TEMPERATURE,
        pressure=P,
    )
    return RP1.density

# Adjusted tank heights to make volume and radius align
COPV_HEIGHT = COPV_VOLUME * (1 + 1e-8)/ (np.pi * COPV_RADIUS**2)
LOX_TANK_HEIGHT = LOX_TANK_VOLUME  * (1 + 2e-6) / (np.pi * LOX_TANK_RADIUS**2)
FUEL_TANK_HEIGHT = FUEL_TANK_VOLUME * (1 + 1e-8) / (np.pi * FUEL_TANK_RADIUS**2)

COPVTankGeometry = CylindricalTank(
    radius_function=COPV_RADIUS,
    height=COPV_HEIGHT,
    spherical_caps=False,
)

LOXTankGeometry = CylindricalTank(
    radius_function=LOX_TANK_RADIUS,
    height=LOX_TANK_HEIGHT,
    spherical_caps=False,
)

FuelTankGeometry = CylindricalTank(
    radius_function=FUEL_TANK_RADIUS,
    height=FUEL_TANK_HEIGHT,
    spherical_caps=False,
)

Pressurant = Fluid(
    name="GN2",
    density=GN2Density,
)

LOXPropellant = Fluid(
    name="LOX",
    density=LOXDensity,
)

FuelPropellant = Fluid(
    name="RP-1",
    density=FuelDensity,
)

COPVTank = MassBasedTank(
    name="COPV",
    geometry=COPVTankGeometry,
    flux_time=BURN_TIME,
    liquid=Fluid("Plutonium", density=1.0),
    gas=Pressurant,
    liquid_mass=0.0,
    gas_mass=f"{PROP_CURVE_DIRECTORY}/Copv_Gas_Mass.csv",
    temperature=f"{PROP_CURVE_DIRECTORY}/Copv_Temperature.csv",
    pressure=f"{PROP_CURVE_DIRECTORY}/Copv_Pressure.csv",
)

LOXTank = MassBasedTank(
    name="LOX Tank",
    geometry=LOXTankGeometry,
    flux_time=BURN_TIME,
    liquid=LOXPropellant,
    gas=Pressurant,
    liquid_mass=f"{PROP_CURVE_DIRECTORY}/Lox_Liquid_Mass.csv",
    gas_mass=f"{PROP_CURVE_DIRECTORY}/Lox_Ullage_Gas_Mass.csv",
    temperature=f"{PROP_CURVE_DIRECTORY}/Lox_Ullage_Temperature.csv",
    pressure=f"{PROP_CURVE_DIRECTORY}/Lox_Tank_Pressure.csv",
    discretize=None,
)

FuelTank = MassBasedTank(
    name="Fuel Tank",
    geometry=FuelTankGeometry,
    flux_time=BURN_TIME,
    liquid=FuelPropellant,
    gas=Pressurant,
    liquid_mass=f"{PROP_CURVE_DIRECTORY}/Fuel_Liquid_Mass.csv",
    gas_mass=f"{PROP_CURVE_DIRECTORY}/Fuel_Ullage_Gas_Mass.csv",
    temperature=f"{PROP_CURVE_DIRECTORY}/Fuel_Ullage_Temperature.csv",
    pressure=f"{PROP_CURVE_DIRECTORY}/Fuel_Tank_Pressure.csv",
    discretize=None,
)


ElytraEngine = LiquidMotor(
    thrust_source=f"{PROP_CURVE_DIRECTORY}/Thrust.csv",
    dry_mass=ENGINE_MASS,
    dry_inertia=(
        ENGINE_I_XX,
        ENGINE_I_YY,
        ENGINE_I_ZZ,
    ),
    nozzle_radius=np.sqrt(NOZZLE_EXIT_AREA / np.pi),
    center_of_dry_mass_position=ENGINE_CG_Z - NOZZLE_EXIT_Z,
    nozzle_position=0.0,
    burn_time=BURN_TIME,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

ElytraEngine.add_tank(
    tank=LOXTank,
    position=LOX_TANK_Z - NOZZLE_EXIT_Z,
)

ElytraEngine.add_tank(
    tank=FuelTank,
    position=FUEL_TANK_Z - NOZZLE_EXIT_Z,
)

ElytraEngine.add_tank(
    tank=COPVTank,
    position=COPV_Z - NOZZLE_EXIT_Z,
)







# ---- Rocket Definition ---- #

Elytra = Rocket(
    radius=VEHICLE_RADIUS,
    mass=VEHICLE_DRY_MASS,
    inertia=(
        I_XX_DRY,
        I_YY_DRY,
        I_ZZ_DRY,
        I_XY_DRY,
        I_XZ_DRY,
        I_YZ_DRY,
    ),
    power_off_drag=CD_POWER_OFF,
    power_on_drag=CD_POWER_ON,
    center_of_mass_without_motor=CG_Z_DRY,
    coordinate_system_orientation="tail_to_nose",
)

Elytra.add_motor(
    motor=ElytraEngine,
    position=NOZZLE_EXIT_Z,
)

Elytra.add_nose(
    length=NOSECONE_LENGTH,
    kind=NOSECONE_TYPE,
    position=NOSECONE_Z,
)


Elytra.add_trapezoidal_fins(
    n=FIN_COUNT,
    root_chord=FIN_ROOT_CHORD,
    tip_chord=FIN_TIP_CHORD,
    span=FIN_SPAN,
    position=FIN_Z,
    sweep_length=FIN_SWEEP_LENGTH,
    cant_angle=FIN_CANT_ANGLE,
)







# ---- Flight Simulation ---- #
if RUN_FLIGHT:
    FlightModel = CantedThrustFlight if THRUST_MISALIGNMENT_ACTIVE else Flight

    flight_kwargs = dict(
        rocket=Elytra,
        environment=LaunchConditions,
        rail_length=RAIL_LENGTH,
        inclination=RAIL_INCLINATION,
        heading=RAIL_HEADING,
        max_time_step=MAX_TIMESTEP,
        max_time=MAX_TIME,
        terminate_on_apogee=TERMINATE_ON_APOGEE,
        verbose=VERBOSE
    )

    if THRUST_MISALIGNMENT_ACTIVE:
        ElytraFlight = FlightModel(
            **flight_kwargs,
            thrust_misalignment_angle=THRUST_MISALIGNMENT_ANGLE,
        )
    else:
        ElytraFlight = FlightModel(**flight_kwargs)

    ElytraFlight.plots.trajectory_3d()
    ElytraFlight.prints.all()

    if EXPORT_RESULTS:
        ExportFlightResults(
            flight=ElytraFlight,
            filename=RESULTS_FILENAME,
        )
