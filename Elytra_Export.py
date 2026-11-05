import re
import warnings
from pathlib import Path
import h5py
import numpy as np
from rocketpy import Function


def ExportFlightResults(flight, filename):
    time = np.asarray(flight.solution_array[:, 0], dtype=float)
    with h5py.File(filename, "w") as h5:
        group = h5.create_group("flight")
        group.attrs["description"] = "Elytra RocketPy 6-DOF Flight Results"
        group.create_dataset("time", data=time)
        skipped = []
        for name in dir(flight):
            if name.startswith("_") or name in ("solution", "solution_array", "time"):
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    value = getattr(flight, name)
                    if caught:
                        skipped.append(name)
                        continue
                if isinstance(value, Function):
                    inputs = value.get_inputs()
                    if len(inputs) != 1 or "time" not in str(inputs[0]).lower():
                        continue
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        values = np.asarray([value(t) for t in time], dtype=float)
                        if caught:
                            skipped.append(name)
                            continue
                    dataset = group.create_dataset(name, data=values)
                    outputs = value.get_outputs()
                    if outputs:
                        dataset.attrs["description"] = str(outputs[0])
                elif isinstance(value, (bool, int, float, np.integer, np.floating)):
                    group.create_dataset(name, data=value)
            except Exception:
                skipped.append(name)
        solution = group.create_dataset("solution", data=np.asarray(flight.solution_array, float))
        solution.attrs["columns"] = "time,x,y,z,vx,vy,vz,e0,e1,e2,e3,w1,w2,w3"
        if skipped:
            group.attrs["skipped_attributes"] = ", ".join(sorted(set(skipped)))
    print(f"Flight results exported to '{filename}'.")


def _get_track(tracks, name):
    key = re.sub(r"\s+", "_", re.sub(r"[\\/\x00]", "_", name.strip()))
    if key not in tracks:
        raise KeyError(f"FullFlow track '{name}' missing; expected '{key}'.")
    return np.asarray(tracks[key], dtype=float)


def _cutoff_time(time, volume, threshold):
    indices = np.where(volume <= threshold)[0]
    if not len(indices):
        return time[-1]
    i = indices[0]
    if i == 0:
        return time[0]
    t0, t1 = time[i - 1:i + 1]
    v0, v1 = volume[i - 1:i + 1]
    return t1 if v1 == v0 else t0 + (threshold - v0) * (t1 - t0) / (v1 - v0)


def ExportRocketPyCurves(filename, output_directory="Elytra_Prop_Curves", tank_empty_volume=0.0):
    with h5py.File(filename, "r") as h5:
        names = [k for k, v in h5.items() if isinstance(v, h5py.Group) and "transient" in v]
        if not names:
            raise ValueError("No FullFlow transient network found")
        run = h5[f"{names[0]}/transient/runs/base"]
        time = np.asarray(run["time"], dtype=float)
        tracks = run["tracks"]
        defs = {
            "thrust": "Thrust [N]",
            "lox_liquid_mass": "LOX Liquid Mass [kg]",
            "fuel_liquid_mass": "Fuel Liquid Mass [kg]",
            "copv_gas_mass": "COPV Gas Mass [kg]",
            "lox_ullage_gas_mass": "LOX Ullage Gas Mass [kg]",
            "fuel_ullage_gas_mass": "Fuel Ullage Gas Mass [kg]",
            "copv_pressure": "COPV Pressure [Pa]",
            "copv_temperature": "COPV Temperature [K]",
            "lox_tank_pressure": "LOX Tank Pressure [Pa]",
            "lox_ullage_temperature": "LOX Ullage Temperature [K]",
            "fuel_tank_pressure": "Fuel Tank Pressure [Pa]",
            "fuel_ullage_temperature": "Fuel Ullage Temperature [K]",
            "lox_liquid_volume": "LOX Liquid Volume [m3]",
            "fuel_liquid_volume": "Fuel Liquid Volume [m3]",
        }
        raw = {name: _get_track(tracks, track) for name, track in defs.items()}
    cutoff = min(_cutoff_time(time, raw["lox_liquid_volume"], tank_empty_volume),
                 _cutoff_time(time, raw["fuel_liquid_volume"], tank_empty_volume))
    export_time = time[time < cutoff]
    if not np.isclose(export_time, cutoff).any():
        export_time = np.append(export_time, cutoff)
    if len(export_time) < 2:
        raise ValueError("Propulsion curve is too short to export")
    for name, value in raw.items():
        raw[name] = np.interp(export_time, time, value)
    headers = {
        "thrust":"Thrust (N)","lox_liquid_mass":"LOX Liquid Mass (kg)",
        "fuel_liquid_mass":"Fuel Liquid Mass (kg)","copv_gas_mass":"COPV Gas Mass (kg)",
        "lox_ullage_gas_mass":"LOX Ullage Gas Mass (kg)",
        "fuel_ullage_gas_mass":"Fuel Ullage Gas Mass (kg)",
        "copv_pressure":"COPV Pressure (Pa)","copv_temperature":"COPV Temperature (K)",
        "lox_tank_pressure":"LOX Tank Pressure (Pa)",
        "lox_ullage_temperature":"LOX Ullage Temperature (K)",
        "fuel_tank_pressure":"Fuel Tank Pressure (Pa)",
        "fuel_ullage_temperature":"Fuel Ullage Temperature (K)",
    }
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for stem, label in headers.items():
        data = np.column_stack((export_time, raw[stem]))
        output_name = "_".join(word.capitalize() for word in stem.split("_"))
        np.savetxt(output_directory / f"{output_name}.csv", data, delimiter=",",
                   header=f"Time (s),{label}", comments="", fmt="%.10g")
    print(f"RocketPy propulsion curves exported through {cutoff:.4f} s")
    return cutoff
