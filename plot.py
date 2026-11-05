from pathlib import Path
from scipy.constants import *
import fullplot as fplt

flight_filename = str(Path(__file__).resolve().parent / "Elytra_6DOF_Results.h5")
flight_group    = "flight"
prop_filename   = str(Path(__file__).resolve().parent / "Elytra_Prop_System.h5")
prop_group      = "Elytra_Prop_System/transient/runs/base/tracks"




flight_results = fplt.open(flight_filename).at(flight_group)
prop_results = fplt.open(prop_filename).at(prop_group)

flight_time = flight_results.time("time")
prop_time = prop_results.time("time")


apogee = flight_results.trace(
    y="altitude",
    x=flight_time,
    name="Altitude") / kilo

stability_margin = flight_results.trace(
    y="stability_margin",
    x=flight_time,
    name="Stability Margin"
)

mach = flight_results.trace(
    y="mach_number",
    x=flight_time,
    name="Mach"
)

fplt.plot(
    apogee,
    xlabel="Time [s]",
    ylabel="Altitude [km]"
)

fplt.plot(
    [stability_margin],
    y2=[mach],
    xlabel="Time [s]",
    ylabel="Altitude [km]",
    y2label="Mach Number"
)

fplt.show()
