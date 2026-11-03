"""RocketPy Flight extension for a fixed, physically canted thrust vector.

This module is intentionally small: it leaves RocketPy's standard 6-DOF,
variable-mass equations untouched and adds the correction associated with
replacing RocketPy's default body-axis thrust vector [0, 0, T] by a fixed
canted vector [T sin(delta), 0, T cos(delta)].

Sign convention used here:
    positive delta -> thrust is tilted toward +body X
                    -> for an aft nozzle this produces a -body-Y pitch moment

That matches the direction of the previous Elytra implementation that used a
positive x thrust eccentricity.

The implementation was checked against RocketPy v1.13.0. It uses one private
post-processing buffer only to keep RocketPy's exported accelerations/moments
consistent with the corrected equations.
"""

from importlib.metadata import PackageNotFoundError, version
import warnings

import numpy as np
from rocketpy import Flight
from rocketpy.mathutils.vector_matrix import Matrix, Vector


SUPPORTED_ROCKETPY_VERSION = "1.13.0"


class CantedThrustFlight(Flight):
    """RocketPy Flight with a fixed thrust-vector misalignment in the X-Z plane.

    Parameters
    ----------
    thrust_misalignment_angle : float
        Fixed thrust cant angle in degrees. Positive angles tilt thrust toward
        +body X. Zero exactly recovers RocketPy's normal thrust direction.
    *args, **kwargs
        Passed directly to :class:`rocketpy.Flight`.
    """

    def __init__(self, *args, thrust_misalignment_angle=0.0, **kwargs):
        self.thrust_misalignment_angle = float(thrust_misalignment_angle)
        self._thrust_misalignment_rad = np.deg2rad(self.thrust_misalignment_angle)

        if abs(self.thrust_misalignment_angle) >= 90:
            raise ValueError("thrust_misalignment_angle must satisfy |angle| < 90 deg")

        try:
            installed_version = version("rocketpy")
        except PackageNotFoundError:  # pragma: no cover
            installed_version = None

        if installed_version not in (None, SUPPORTED_ROCKETPY_VERSION):
            warnings.warn(
                "CantedThrustFlight was validated against RocketPy "
                f"{SUPPORTED_ROCKETPY_VERSION}, but {installed_version} is installed. "
                "The equations should still be checked after upgrading RocketPy.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Flight.__init__ immediately starts the simulation, so the cant angle
        # must be stored before calling super().__init__().
        super().__init__(*args, **kwargs)

    def _net_thrust(self, t, z):
        """Return RocketPy's pressure-corrected scalar net thrust."""
        if self.rocket.motor.burn_start_time < t < self.rocket.motor.burn_out_time:
            pressure = self.env.pressure.get_value_opt(z)
            return max(
                self.rocket.motor.thrust.get_value_opt(t)
                + self.rocket.motor.pressure_thrust(pressure),
                0,
            )
        return 0.0

    def _thrust_vectors_body(self, t, z):
        """Return scalar thrust plus aligned and canted body-frame vectors."""
        net_thrust = self._net_thrust(t, z)
        sin_delta = np.sin(self._thrust_misalignment_rad)
        cos_delta = np.cos(self._thrust_misalignment_rad)

        aligned = Vector([0.0, 0.0, net_thrust])
        canted = Vector(
            [
                net_thrust * sin_delta,
                0.0,
                net_thrust * cos_delta,
            ]
        )
        return net_thrust, aligned, canted

    def u_dot_generalized(self, t, u, post_processing=False):
        """Apply a true fixed thrust cant to RocketPy's standard 6-DOF EOM.

        RocketPy v1.13.0 already evaluates the complete variable-mass 6-DOF
        equations assuming thrust is [0, 0, T] in the body frame. Because the
        external-force contribution is linear, the exact correction can be
        added without duplicating RocketPy's full equations:

            delta_F = F_canted - F_aligned
            tau_cant = r_(COM->nozzle) x F_canted

        The torque uses the instantaneous loaded COM, so the lever arm changes
        automatically as propellant drains.
        """
        # First evaluate RocketPy's normal equations exactly as implemented.
        base_u_dot = super().u_dot_generalized(t, u, post_processing=post_processing)

        if self._thrust_misalignment_rad == 0.0:
            return base_u_dot

        _, _, z, _, _, _, e0, e1, e2, e3, _, _, _ = u
        net_thrust, aligned_thrust, canted_thrust = self._thrust_vectors_body(t, z)

        if net_thrust <= 0.0:
            return base_u_dot

        total_mass = self.rocket.total_mass.get_value_opt(t)

        # RocketPy's state translational coordinates track the center of dry
        # mass (CDM). Reuse RocketPy's own COM-to-CDM vector when mapping the
        # corrected angular acceleration back to CDM acceleration.
        r_cm_t = self.rocket.com_to_cdm_function.get_value_opt(t)
        r_cm = Vector([0.0, 0.0, r_cm_t])

        # Instantaneous inertia tensor about the loaded center of mass.
        inertia_tensor_cdm = self.rocket.get_inertia_tensor_at_time(t)
        parallel_axis_term = (
            r_cm.cross_matrix @ -r_cm.cross_matrix
        ) * total_mass
        inertia_tensor_cm = inertia_tensor_cdm - parallel_axis_term

        # Physical vector from the instantaneous loaded COM to the nozzle exit,
        # expressed in RocketPy's body coordinates. For the Elytra tail-to-nose
        # convention this z component is negative, as expected for an aft nozzle.
        center_of_mass_position = self.rocket.center_of_mass.get_value_opt(t)
        nozzle_from_com_z = (
            self.rocket.nozzle_position - center_of_mass_position
        ) * self.rocket._csys
        r_com_to_nozzle = Vector([0.0, 0.0, nozzle_from_com_z])

        # Replace [0, 0, T] with the canted force and apply that force at the
        # actual nozzle exit. The aligned thrust has zero torque because both
        # its line of action and COM-to-nozzle vector are axial.
        delta_force_body = canted_thrust - aligned_thrust
        cant_moment_cm = r_com_to_nozzle ^ canted_thrust

        delta_w_dot = inertia_tensor_cm.inverse @ cant_moment_cm

        # RocketPy returns inertial-frame translational acceleration of the CDM.
        # Use the same CDM/COM kinematic relationship as RocketPy's standard EOM.
        K = Matrix.transformation([e0, e1, e2, e3])
        delta_v_dot = K @ (
            delta_force_body / total_mass - (r_cm ^ delta_w_dot)
        )

        corrected = list(base_u_dot)
        corrected[3] += delta_v_dot.x
        corrected[4] += delta_v_dot.y
        corrected[5] += delta_v_dot.z
        corrected[10] += delta_w_dot.x
        corrected[11] += delta_w_dot.y
        corrected[12] += delta_w_dot.z

        if post_processing:
            # RocketPy stores [t, ax, ay, az, alpha1, alpha2, alpha3,
            # R1, R2, R3, M1, M2, M3, net_thrust]. Keep these outputs
            # synchronized with the corrected dynamics. R1/R2/R3 remain the
            # aerodynamic force components; the cant moment is added to M.
            row = self._Flight__post_processed_variables[-1]
            row[1] += delta_v_dot.x
            row[2] += delta_v_dot.y
            row[3] += delta_v_dot.z
            row[4] += delta_w_dot.x
            row[5] += delta_w_dot.y
            row[6] += delta_w_dot.z
            row[10] += cant_moment_cm.x
            row[11] += cant_moment_cm.y
            row[12] += cant_moment_cm.z

        return corrected

    def udot_rail1(self, t, u, post_processing=False):
        """1-DOF rail dynamics with the correct axial canted-thrust component.

        While the vehicle is constrained by the rail, the lateral component and
        thrust-induced pitch moment are reacted by the rail. Therefore only the
        body-axis component T*cos(delta) contributes to motion along the rail.
        """
        _, _, z, vx, vy, vz, e0, e1, e2, e3, _, _, _ = u

        total_mass_at_t = self.rocket.total_mass.get_value_opt(t)

        free_stream_velocity = Vector(
            [
                self.env.wind_velocity_x.get_value_opt(z) - vx,
                self.env.wind_velocity_y.get_value_opt(z) - vy,
                -vz,
            ]
        )
        free_stream_speed = abs(free_stream_velocity)
        free_stream_mach = free_stream_speed / self.env.speed_of_sound.get_value_opt(z)
        rho = self.env.density.get_value_opt(z)
        stream_velocity_body = (
            Matrix.transformation([e0, e1, e2, e3]).transpose @ free_stream_velocity
        )
        dynamic_viscosity = self.env.dynamic_viscosity.get_value_opt(z)
        alpha, beta, mach, reynolds = self._Flight__compute_drag_7d_inputs(
            stream_velocity_body,
            free_stream_speed,
            free_stream_mach,
            rho,
            dynamic_viscosity,
        )
        drag_coeff = self.rocket.power_on_drag_7d(
            alpha, beta, mach, reynolds, 0, 0, 0
        )

        net_thrust, _, canted_thrust = self._thrust_vectors_body(t, z)
        axial_thrust = canted_thrust.z

        R3 = -0.5 * rho * (free_stream_speed**2) * self.rocket.area * drag_coeff

        a3 = (R3 + axial_thrust) / total_mass_at_t - (
            e0**2 - e1**2 - e2**2 + e3**2
        ) * self.env.gravity.get_value_opt(z)

        if a3 > 0:
            ax = 2 * (e1 * e3 + e0 * e2) * a3
            ay = 2 * (e2 * e3 - e0 * e1) * a3
            az = (1 - 2 * (e1**2 + e2**2)) * a3
        else:
            ax, ay, az = 0.0, 0.0, 0.0

        if post_processing:
            # The free-flight calculation records the true external cant moment
            # and thrust. Then overwrite accelerations with the rail-constrained
            # values exactly as RocketPy does for its standard rail phase.
            self.u_dot_generalized(t, u, post_processing=True)
            self._Flight__post_processed_variables[-1][1:7] = [
                ax,
                ay,
                az,
                0.0,
                0.0,
                0.0,
            ]
            # Preserve RocketPy's scalar net-thrust output explicitly.
            self._Flight__post_processed_variables[-1][13] = net_thrust

        return [vx, vy, vz, ax, ay, az, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
