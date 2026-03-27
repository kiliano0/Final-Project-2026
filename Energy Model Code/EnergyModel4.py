"""SCARA robot energy model

This module provides a lightweight numerical energy model for a SCARA-style
manipulator (2R + optional prismatic/wrist). It computes joint torques from
the standard rigid-body dynamics (M, Coriolis, gravity), instantaneous
mechanical power and a simple electrical power approximation using motor
efficiencies and regeneration efficiency.

Usage:
 - Create SCARAEnergyModel with link/mass/inertia parameters.
 - Call `torques(q, qd, qdd)` to get joint torques and power at an instant.
 - Call `energy_over_trajectory(t, q, qd=None, qdd=None)` to get cumulative
   mechanical and electrical energy over the given trajectory samples.

Notes / assumptions:
 - The dynamics implementation is for the 2-link planar arm (theta1, theta2).
   Additional joints (prismatic z, wrist) are treated as simple torque = I*acc
   or force*acc where needed and included in power sums if provided.
 - This is intended as an engineering tool you can adapt. See the docstring
   and inline comments for formulas.
"""

from typing import Optional, Sequence, Tuple, List
import csv
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# SIMULATION CONSTANTS (single place to tune physical behavior)
# =============================================================================
# Geometry (meters)
SIM_L1 = 0.200
SIM_L2 = 0.250

# Link masses (kg)
SIM_M1 = 5
SIM_M2 = 7

# Link inertias about COM (kg*m^2)
# Set to None to auto-compute as (1/12)*m*L^2 from SIM_Mx/SIM_Lx.
SIM_I1 = None
SIM_I2 = None

# Base electrical draw (W)
SIM_CONSTANT_POWER = 96.2

# Friction/loss parameters
SIM_VISCOUS = [2, 0.3, 1, 0.08]       # [J1, J2, J3, J4]
SIM_SPEED_SQ = [9, 4.0, 1, 0.0]     # [J1, J2, J3, J4]
SIM_ACCEL_SCALE = [1.0, 1.0, 1.0, 1]  # acceleration scaling [J1, J2, J3, J4]
SIM_SPEED_EXP = 2                # velocity-loss exponent n in P_loss = k * |qd|^n

# Joint-4 (wrist / quill rotation) physical parameters
# tau4 = J4*alpha4 + b4*omega4
SIM_J4_INERTIA = 0.002          # kg*m^2 (equivalent inertia at motor/output axis)

# J1 speed-shaping (optional): boost low-speed power and reduce high-speed power
# gain(|w|) = 1 + low_boost*(1-r) - high_reduction*r, where r = x/(1+x), x=(|w|/ref)^shape_exp
SIM_J1_LOW_SPEED_BOOST = 2.4
SIM_J1_HIGH_SPEED_REDUCTION = 0.42
SIM_J1_SHAPE_REF_SPEED = 1.2
SIM_J1_SHAPE_EXP = 1.8

SIM_J2_LOW_SPEED_BOOST = 2.0 #0.9
SIM_J2_HIGH_SPEED_REDUCTION = 0.4 #0.6
SIM_J2_SHAPE_REF_SPEED = 1.5 #2.5
SIM_J2_SHAPE_EXP = 2.5 #2.5

SIM_J3_LOW_SPEED_BOOST = 4.5
SIM_J3_HIGH_SPEED_BOOST = 15
SIM_J3_SHAPE_REF_SPEED = 1
SIM_J3_SHAPE_EXP = 4

SIM_J4_LOW_SPEED_BOOST = 3
SIM_J4_HIGH_SPEED_BOOST = -0.5
SIM_J4_SHAPE_REF_SPEED = 7
SIM_J4_SHAPE_EXP = 2

# Plot control
SIM_PLOT_SPEED_SHAPING_GRAPHS = True

# Runtime override flags (edit here instead of typing long CLI commands).
# Keys use argparse destination names (use '_' not '-').
# Examples:
#   'csv': 'WaitTimesIn.csv',
#   'calibration': 'CalibrationJ1Power.CSV',
#   'j1_low_speed_boost': 0.30,
#   'j1_high_speed_reduction': 0.20,
#   'j1_shape_ref_speed': 1.0,
#   'j1_shape_exp': 2.0,
#   'j2_low_speed_boost': 0.30,
#   'j2_high_speed_reduction': 0.20,
#   'j2_shape_ref_speed': 1.0,
#   'j2_shape_exp': 2.0,
#   'j3_low_speed_boost': 0.0,
#   'j3_high_speed_boost': 0.0,
#   'j3_shape_ref_speed': 1.0,
#   'j3_shape_exp': 2.0,
#   'j4_low_speed_boost': 0.0,
#   'j4_high_speed_boost': 0.0,
#   'j4_shape_ref_speed': 1.0,
#   'j4_shape_exp': 2.0,
SIM_CLI_OVERRIDE_FLAGS = {
	# 'csv': 'WaitTimesIn.csv',
	# 'calibration': 'CalibrationJ1Power.CSV',
	#'j1_low_speed_boost': 3,
	#'j1_high_speed_reduction': 0.2,
	#'j1_shape_ref_speed': 1.0,
	#'j1_shape_exp': 2,
	#'j2_low_speed_boost': 0,
	#'j2_high_speed_reduction': 0,
	#'j2_shape_ref_speed': 1.0,
	#'j2_shape_exp': 2,
}

# Efficiencies and braking model
SIM_MOTOR_EFF = [0.9, 0.9, 0.9, 0.9]
SIM_REGEN_EFF = [0.6, 0.6, 1, 0.6]
SIM_BRAKE_EFFICIENCY = 0.9 
SIM_USE_POWER_TO_BRAKE = False  # if True, negative mechanical power consumes electrical power instead of regenerating it

# Gravity / SCARA assumptions
SIM_SCARA_HORIZONTAL = True
SIM_GRAVITY = 9.81
SIM_QUILL_MASS = 2  # mass on the vertical/prismatic axis (joint index 2) if scara_horizontal is True; set to None to use older approximation of (m1+m2)
SIM_QUILL_GRAVITY_MOVING_ONLY = True
SIM_QUILL_DEADBAND = 1e-4

# Quick tuning presets (edit constants above to apply):
# 1) Higher accel-driven peaks:
#    - Increase SIM_M1/SIM_M2 and/or set SIM_I1/SIM_I2 to larger values.
#    - Optional: increase SIM_MOTOR_EFF slightly if you want lower electrical inflation from efficiency losses.
#
# 2) Faster power drop when motion slows:
#    - Decrease SIM_VISCOUS (linear with speed).
#    - Decrease SIM_SPEED_SQ (quadratic with speed; strongest at high speed).
#
# 3) Keep peaks, reduce baseline only:
#    - Decrease SIM_CONSTANT_POWER.
#
# 4) Braking behavior shape:
#    - If SIM_USE_POWER_TO_BRAKE=True, braking consumes power (servo-style).
#    - If False, negative mechanical power partially regenerates via SIM_REGEN_EFF.


class SCARAEnergyModel:
	"""Energy model for a SCARA robot (2R core dynamics).

	Parameters (typical):
	- l1, l2: link lengths (m)
	- m1, m2: link masses (kg)
	- lc1, lc2: distance from joint to link COM for link1/2 (m)
	- I1, I2: link inertias about COM (kg*m^2)
	- g: gravity (default 9.81)
	- motor_eff: list/array of motor efficiencies per joint (0-1) for motoring
	- regen_eff: list/array of regen efficiencies per joint (0-1) for braking
	- viscous_friction: per-joint viscous friction coefficient (Nm/(rad/s))
	- speed_squared_loss: per-joint velocity-squared power loss coefficient (W/(rad/s)^2)
	- constant_power: constant electrical draw (W) added to electrical power

	The model expects joint ordering [theta1, theta2, z, theta4] optionally,
	but dynamics M/C/G are computed only for theta1/theta2 (planar arm).
	Additional joints are handled as simple inertial/friction loads for power
	accounting.
	"""

	def __init__(
		self,
		l1: float,
		l2: float,
		m1: float,
		m2: float,
		lc1: Optional[float] = None,
		lc2: Optional[float] = None,
		I1: Optional[float] = None,
		I2: Optional[float] = None,
		g: float = 9.81,
		motor_eff: Optional[Sequence[float]] = None,
		regen_eff: Optional[Sequence[float]] = None,
		viscous_friction: Optional[Sequence[float]] = None,
		speed_squared_loss: Optional[Sequence[float]] = None,
		accel_scale: Optional[Sequence[float]] = None,
		speed_loss_exponent: float = 2.0,
		constant_power: float = 0.0,
		scara_horizontal: bool = True,
		use_power_to_brake: bool = True,
		brake_efficiency: float = 0.9,
		quill_mass: Optional[float] = 3.0,
		quill_gravity_moving_only: bool = True,
		quill_deadband: float = 1e-4,
		j1_low_speed_boost: float = 0.0,
		j1_high_speed_reduction: float = 0.0,
		j1_shape_ref_speed: float = 1.0,
		j1_shape_exp: float = 2.0,
		j2_low_speed_boost: float = 0.0,
		j2_high_speed_reduction: float = 0.0,
		j2_shape_ref_speed: float = 1.0,
		j2_shape_exp: float = 2.0,
		j3_low_speed_boost: float = 0.0,
		j3_high_speed_boost: float = 0.0,
		j3_shape_ref_speed: float = 1.0,
		j3_shape_exp: float = 2.0,
		j4_low_speed_boost: float = 0.0,
		j4_high_speed_boost: float = 0.0,
		j4_shape_ref_speed: float = 1.0,
		j4_shape_exp: float = 2.0,
		j4_inertia: float = 0.0,
	) -> None:
		self.l1 = l1
		self.l2 = l2
		self.m1 = m1
		self.m2 = m2
		self.lc1 = lc1 if lc1 is not None else l1 / 2.0
		self.lc2 = lc2 if lc2 is not None else l2 / 2.0
		# If inertias not provided, approximate each link as a slender rod about COM: I = (1/12)*m*L^2
		self.I1 = float(I1) if I1 is not None else (1.0 / 12.0) * self.m1 * (self.l1 ** 2)
		self.I2 = float(I2) if I2 is not None else (1.0 / 12.0) * self.m2 * (self.l2 ** 2)
		self.g = g

		# default to three/four joints efficiencies if not provided
		self.default_joints = 4
		if motor_eff is None:
			self.motor_eff = np.array(SIM_MOTOR_EFF)
		else:
			self.motor_eff = np.array(motor_eff)
		# Regeneration efficiency (used only when not in use_power_to_brake mode)
		if regen_eff is None:
			self.regen_eff = np.array(SIM_REGEN_EFF)
		else:
			self.regen_eff = np.array(regen_eff)
		if viscous_friction is None:
			# Default viscous friction coefficients (Nm/(rad/s)) for J1, J2, J3, J4
			# Keep moderate to preserve acceleration modeling, rely more on speed-squared losses
			self.viscous = np.array(SIM_VISCOUS)  # J1, J2 (rotary), J3 (prismatic N/m/s), J4 (wrist)
		else:
			self.viscous = np.array(viscous_friction)
		
		if speed_squared_loss is None:
			# Default velocity-squared loss coefficients (W/(rad/s)^2) for bearing friction, windage
			# These create power losses proportional to velocity squared: P_loss = k * qd^2
			# Increased significantly to match constant-speed power without affecting acceleration torques
			self.speed_sq_loss = np.array(SIM_SPEED_SQ)  # J1, J2, J3, J4
		else:
			self.speed_sq_loss = np.array(speed_squared_loss)

		if accel_scale is None:
			self.accel_scale = np.array(SIM_ACCEL_SCALE)
		else:
			self.accel_scale = np.array(accel_scale)
		self.speed_loss_exponent = float(speed_loss_exponent)

		self.constant_power = float(constant_power)
		# If True, treat the 2R planar arm as a horizontal SCARA: gravity does not
		# produce torques on the planar joints (theta1/theta2). Instead gravity is
		# applied to the vertical/prismatic axis (joint index 2) if present.
		self.scara_horizontal = bool(scara_horizontal)
		# Servo brake behavior: when True, negative mechanical power consumes
		# electrical power instead of regenerating it. brake_efficiency models
		# how much electrical power is required relative to |mechanical power|.
		self.use_power_to_brake = bool(use_power_to_brake)
		self.brake_efficiency = float(brake_efficiency)
		# Mass on the vertical/prismatic axis ('quill'). If provided and
		# scara_horizontal is True, gravity force on joint index 2 is
		# -(quill_mass * g) instead of the older approximation -(m1+m2)*g.
		self.quill_mass = float(quill_mass) if quill_mass is not None else None
		# Apply gravity on quill only when it's moving (simulate mechanical hold brake
		# that carries weight at rest with no electrical power). Deadband prevents
		# chattering around zero velocity.
		self.quill_gravity_moving_only = bool(quill_gravity_moving_only)
		self.quill_deadband = float(quill_deadband)
		self.j1_low_speed_boost = float(max(j1_low_speed_boost, 0.0))
		self.j1_high_speed_reduction = float(max(j1_high_speed_reduction, 0.0))
		self.j1_shape_ref_speed = float(max(j1_shape_ref_speed, 1e-9))
		self.j1_shape_exp = float(max(j1_shape_exp, 0.5))
		self.j2_low_speed_boost = float(max(j2_low_speed_boost, 0.0))
		self.j2_high_speed_reduction = float(max(j2_high_speed_reduction, 0.0))
		self.j2_shape_ref_speed = float(max(j2_shape_ref_speed, 1e-9))
		self.j2_shape_exp = float(max(j2_shape_exp, 0.5))
		self.j3_low_speed_boost = float(max(j3_low_speed_boost, -0.95))
		self.j3_high_speed_boost = float(max(j3_high_speed_boost, 0.0))
		self.j3_shape_ref_speed = float(max(j3_shape_ref_speed, 1e-9))
		self.j3_shape_exp = float(max(j3_shape_exp, 0.5))
		self.j4_low_speed_boost = float(max(j4_low_speed_boost, -0.95))
		self.j4_high_speed_boost = float(max(j4_high_speed_boost, -0.95))
		self.j4_shape_ref_speed = float(max(j4_shape_ref_speed, 1e-9))
		self.j4_shape_exp = float(max(j4_shape_exp, 0.5))
		# Non-core joint rotational physics (currently used for J4)
		self.j4_inertia = float(max(j4_inertia, 0.0))

	# ---- dynamics (2R planar manipulator core) ----
	def M(self, q: np.ndarray) -> np.ndarray:
		"""Mass matrix M(q) for the 2-link planar arm.

		q: array-like with at least two entries [theta1, theta2]
		returns 2x2 numpy array
		"""
		th1, th2 = float(q[0]), float(q[1])
		m1, m2 = self.m1, self.m2
		l1, lc1, lc2 = self.l1, self.lc1, self.lc2
		I1, I2 = self.I1, self.I2

		# common shorthand
		c2 = np.cos(th2)

		M11 = I1 + I2 + m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2)
		M12 = I2 + m2 * (lc2 ** 2 + l1 * lc2 * c2)
		M22 = I2 + m2 * lc2 ** 2

		M = np.array([[M11, M12], [M12, M22]], dtype=float)
		return M

	def coriolis_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
		"""Compute Coriolis/centrifugal torque vector for 2R arm.

		Using standard (approximate) closed-form expression for 2-link arm:
		h = -m2*l1*lc2*sin(th2)
		cori = [h*(2*qd1*qd2 + qd2**2), -h*(qd1**2)] with sign consistent so that
		tau = M qdd + cori + G
		"""
		th1, th2 = float(q[0]), float(q[1])
		qd1, qd2 = float(qd[0]), float(qd[1])
		h = -self.m2 * self.l1 * self.lc2 * np.sin(th2)
		# coriolis torque vector (2x)
		tau1 = h * (2 * qd1 * qd2 + qd2 ** 2)
		tau2 = -h * (qd1 ** 2)
		return np.array([tau1, tau2], dtype=float)

	def gravity_torque(self, q: np.ndarray) -> np.ndarray:
		"""Gravity torque vector for 2R arm (planar, gravity acts in -y, use cos).

		tau_g1 = (m1*lc1 + m2*l1)*g*cos(th1) + m2*lc2*g*cos(th1+th2)
		tau_g2 = m2*lc2*g*cos(th1+th2)
		"""
		# For SCARA (horizontal) robots gravity typically does not produce torques
		# about the base/shoulder joints because the arm rotates in a horizontal
		# plane. The vertical motion is usually handled by a prismatic 'quill'. If
		# self.scara_horizontal is True we return zero gravity torques for the
		# planar joints; otherwise return the standard planar gravity terms.
		if self.scara_horizontal:
			return np.array([0.0, 0.0], dtype=float)

		th1, th2 = float(q[0]), float(q[1])
		g = self.g
		tau_g1 = (self.m1 * self.lc1 + self.m2 * self.l1) * g * np.cos(th1) + self.m2 * self.lc2 * g * np.cos(th1 + th2)
		tau_g2 = self.m2 * self.lc2 * g * np.cos(th1 + th2)
		return np.array([tau_g1, tau_g2], dtype=float)

	def speed_gain_profile(
		self,
		speed_abs: float,
		low_speed_boost: float,
		high_speed_reduction: float,
		shape_ref_speed: float,
		shape_exp: float,
	) -> float:
		"""Generic multiplier profile vs absolute speed.

		gain(speed) = high_gain + (low_gain - high_gain) / (1 + (|speed|/ref)^exp)
		where:
		- low_gain  = 1 + low_speed_boost
		- high_gain = 1 - high_speed_reduction (clamped to >= 0.05)
		"""
		w = abs(float(speed_abs))
		ref = float(max(shape_ref_speed, 1e-9))
		exp = float(max(shape_exp, 0.5))
		x = (w / ref) ** exp
		low_gain = 1.0 + float(max(low_speed_boost, 0.0))
		high_gain = max(0.05, 1.0 - float(max(high_speed_reduction, 0.0)))
		gain = high_gain + (low_gain - high_gain) / (1.0 + x)
		return max(gain, 0.05)

	def j1_speed_gain(self, speed_abs: float) -> float:
		"""Multiplicative J1 gain as a smooth function of absolute speed.

		This is intentionally multiplier-only (no additive power).
		- Near zero speed: gain -> (1 + j1_low_speed_boost)
		- At very high speed: gain -> (1 - j1_high_speed_reduction)
		Transition is controlled by j1_shape_ref_speed and j1_shape_exp.
		"""
		return self.speed_gain_profile(
			speed_abs=speed_abs,
			low_speed_boost=self.j1_low_speed_boost,
			high_speed_reduction=self.j1_high_speed_reduction,
			shape_ref_speed=self.j1_shape_ref_speed,
			shape_exp=self.j1_shape_exp,
		)

	def j2_speed_gain(self, speed_abs: float) -> float:
		"""Multiplicative J2 gain as a smooth function of absolute speed."""
		return self.speed_gain_profile(
			speed_abs=speed_abs,
			low_speed_boost=self.j2_low_speed_boost,
			high_speed_reduction=self.j2_high_speed_reduction,
			shape_ref_speed=self.j2_shape_ref_speed,
			shape_exp=self.j2_shape_exp,
		)

	def j3_speed_gain(self, speed_abs: float) -> float:
		"""Multiplicative J3 gain as a smooth function of absolute speed.

		For J3, both low-speed and high-speed terms are boosts:
		- Near zero speed: gain -> (1 + j3_low_speed_boost)
		- At high speed:   gain -> (1 + j3_high_speed_boost)
		"""
		w = abs(float(speed_abs))
		ref = float(max(self.j3_shape_ref_speed, 1e-9))
		exp = float(max(self.j3_shape_exp, 0.5))
		x = (w / ref) ** exp
		r = x / (1.0 + x)
		low_gain = 1.0 + float(max(self.j3_low_speed_boost, 0.0))
		high_gain = 1.0 + float(max(self.j3_high_speed_boost, 0.0))
		gain = low_gain + (high_gain - low_gain) * r
		return max(gain, 0.05)

	def j4_speed_gain(self, speed_abs: float) -> float:
		"""Multiplicative J4 gain as a smooth function of absolute speed.

		For J4, low-speed term is a boost/reduction and high-speed term is a boost/reduction:
		- Near zero speed: gain -> (1 + j4_low_speed_boost)
		- At high speed:   gain -> (1 + j4_high_speed_boost)
		So a negative j4_high_speed_boost reduces power at high speed.
		"""
		w = abs(float(speed_abs))
		ref = float(max(self.j4_shape_ref_speed, 1e-9))
		exp = float(max(self.j4_shape_exp, 0.5))
		x = (w / ref) ** exp
		r = x / (1.0 + x)
		low_gain = 1.0 + float(max(self.j4_low_speed_boost, -0.95))
		high_gain = 1.0 + float(max(self.j4_high_speed_boost, -0.95))
		gain = low_gain + (high_gain - low_gain) * r
		return max(gain, 0.05)

	def torques(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
		"""Compute joint torques and return (tau, mech_power_j, elec_power_total, elec_power_j).

		Returns:
		- tau: (n,) joint torque/force array
		- mech_power_j: (n,) mechanical power per joint (tau_i * qd_i)
		- elec_power_total: scalar total electrical power including constant_power
		- elec_power_j: (n,) electrical power contribution per joint (excludes constant_power)

		q, qd, qdd: arrays with length >= 2 (theta1, theta2, ...). Additional
		joints are included in power/tau arrays but not in full M/C/G dynamics.
		"""
		q = np.asarray(q, dtype=float)
		qd = np.asarray(qd, dtype=float)
		qdd = np.asarray(qdd, dtype=float)

		# ensure at least 2 joints
		if q.shape[0] < 2:
			raise ValueError("q must have at least 2 elements (theta1, theta2)")

		# 2R dynamics
		M2 = self.M(q)
		cori = self.coriolis_torque(q, qd)
		grav = self.gravity_torque(q)

		qdd2 = np.array([float(qdd[0]), float(qdd[1])], dtype=float)
		acc_scale_2r = np.ones(2)
		acc_scale_2r[0] = self.accel_scale[0] if len(self.accel_scale) > 0 else 1.0
		acc_scale_2r[1] = self.accel_scale[1] if len(self.accel_scale) > 1 else 1.0
		tau2 = M2.dot(qdd2 * acc_scale_2r) + cori + grav

		# friction/viscous
		visc = np.zeros(max(self.default_joints, len(q)))
		visc[: len(self.viscous)] = self.viscous
		# build full tau vector: include additional joints as simple inertial torque (I*qdd)
		n = max(len(q), self.default_joints)
		tau = np.zeros(n)
		tau[0:2] = tau2 + visc[:2] * qd[0:2]

		# for joints beyond 2, use simple physically-motivated per-axis models.
		for i in range(2, n):
			iqdd = float(qdd[i]) if i < len(qdd) else 0.0
			# For non-core joints approximate torque/force: viscous term plus
			# gravitational force for a vertical prismatic 'quill' (joint index 2)
			qd_i = float(qd[i]) if i < len(qd) else 0.0
			grav_term = 0.0
			inertia_term = 0.0
			if i == 2 and self.scara_horizontal:
				# vertical/prismatic axis gravity load; prefer explicit quill_mass if provided.
				# Sign convention: +J3 is upward. Actuator force needed to counter gravity is +m*g.
				# Apply when moving if quill_gravity_moving_only is True; otherwise apply always.
				if self.quill_mass is not None:
					if self.quill_gravity_moving_only:
						if abs(qd_i) > self.quill_deadband:
							grav_term = self.quill_mass * self.g
						else:
							grav_term = 0.0
					else:
						grav_term = self.quill_mass * self.g
				else:
					grav_term = 0.0
			if i == 3:
				acc_scale_i = self.accel_scale[i] if i < len(self.accel_scale) else 1.0
				inertia_term = self.j4_inertia * acc_scale_i * iqdd

			tau[i] = inertia_term + visc[i] * qd_i + grav_term

		# instantaneous mechanical power per joint (tau_i * qd_i)
		qd_full = np.zeros(n)
		qd_full[: len(qd)] = qd
		mech_power_j = tau * qd_full
		mech_power = mech_power_j.sum()

		# Add velocity-shaped power losses (bearing friction, windage, etc.)
		# These are always positive (consume power) and scale with |speed|^n
		speed_sq_coeff = np.zeros(n)
		speed_sq_coeff[: len(self.speed_sq_loss)] = self.speed_sq_loss
		speed_sq_power_j = speed_sq_coeff * (np.abs(qd_full) ** self.speed_loss_exponent)
		
		# electrical power approximation using efficiencies
		elec_power = self.constant_power
		# ensure eff arrays length
		motor_eff = np.ones(n)
		regen_eff = np.ones(n)
		motor_eff[: len(self.motor_eff)] = self.motor_eff
		regen_eff[: len(self.regen_eff)] = self.regen_eff

		elec_power_j = np.zeros(n)
		for i in range(n):
			p = mech_power_j[i]
			# Add velocity-squared losses to electrical power (always positive)
			speed_sq_loss_i = speed_sq_power_j[i]
			
			if p >= 0:
				# motoring: divide by motor efficiency and add speed-squared losses
				eff = motor_eff[i] if motor_eff[i] > 0 else 1.0
				joint_elec = p / eff + speed_sq_loss_i
			else:
				# braking behavior
				# For Joint 3 (index 2), always consume power when braking instead of regenerating.
				# This matches the measured behavior where J3 draws power in both directions.
				if self.use_power_to_brake or i == 2:
					# Servo-style braking: consume electrical power to absorb mechanical power
					# Add positive electrical power equal to |p| divided by brake efficiency
					effb = self.brake_efficiency if self.brake_efficiency > 0 else 1.0
					joint_elec = (-p) / effb + speed_sq_loss_i
				else:
					# regeneration: fraction returned to bus (reduces net electrical power)
					effr = regen_eff[i] if regen_eff[i] > 0 else 0.0
					joint_elec = p * effr + speed_sq_loss_i
			# Optional J1-only speed shaping using a multiplier based on speed.
			if i == 0 and (self.j1_low_speed_boost > 0.0 or self.j1_high_speed_reduction > 0.0):
				w = abs(float(qd_full[0]))
				joint_elec *= self.j1_speed_gain(w)
			# Optional J2-only speed shaping using a multiplier based on speed.
			if i == 1 and (self.j2_low_speed_boost > 0.0 or self.j2_high_speed_reduction > 0.0):
				w = abs(float(qd_full[1]))
				joint_elec *= self.j2_speed_gain(w)
			# Optional J3-only speed shaping using a multiplier based on speed.
			if i == 2 and (self.j3_low_speed_boost > 0.0 or self.j3_high_speed_boost > 0.0):
				w = abs(float(qd_full[2]))
				joint_elec *= self.j3_speed_gain(w)
			# Optional J4-only speed shaping using a multiplier based on speed.
			if i == 3 and (self.j4_low_speed_boost != 0.0 or self.j4_high_speed_boost != 0.0):
				w = abs(float(qd_full[3]))
				joint_elec *= self.j4_speed_gain(w)

			elec_power_j[i] = joint_elec
			elec_power += joint_elec

		return tau, mech_power_j, elec_power, elec_power_j

	# ---- trajectory helpers ----
	def energy_over_trajectory(
		self,
		t: np.ndarray,
		q: np.ndarray,
		qd: Optional[np.ndarray] = None,
		qdd: Optional[np.ndarray] = None,
	) -> Tuple[float, float]:
		"""Compute total mechanical and electrical energy over sampled trajectory.

		t: 1D time array (seconds)
		q: (N, n_joints) array of joint positions (radians/meters)
		qd, qdd: optional precomputed derivatives; if not given they'll be
				 approximated with finite differences.

		Returns (E_mech, E_elec) in Joules (electrical includes constant_power).
		"""
		t = np.asarray(t, dtype=float)
		q = np.asarray(q, dtype=float)
		if q.ndim == 1:
			q = q.reshape((q.shape[0], 1))

		N = t.size
		n = q.shape[1]

		# compute qd, qdd if needed (central differences)
		if qd is None:
			qd = np.zeros_like(q)
			qd[1:-1] = (q[2:] - q[:-2]) / (t[2:, None] - t[:-2, None])
			qd[0] = (q[1] - q[0]) / (t[1] - t[0])
			qd[-1] = (q[-1] - q[-2]) / (t[-1] - t[-2])
		if qdd is None:
			qdd = np.zeros_like(q)
			qdd[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / ((t[2:, None] - t[1:-1, None]) * (t[1:-1, None] - t[:-2, None]))
			qdd[0] = (qdd[1])
			qdd[-1] = (qdd[-2])

		mech_power_samples = np.zeros(N)
		elec_power_samples = np.zeros(N)

		for k in range(N):
			qk = q[k]
			qdk = qd[k]
			qddk = qdd[k]
			_, mech_j, elec_total, _ = self.torques(qk, qdk, qddk)
			mech_power_samples[k] = mech_j.sum()
			elec_power_samples[k] = elec_total

		# integrate in time with trapezoidal rule
		# use np.trapezoid (replacement for deprecated np.trapz)
		E_mech = np.trapezoid(mech_power_samples, t)
		E_elec = np.trapezoid(elec_power_samples, t)
		return float(E_mech), float(E_elec)


if __name__ == "__main__":
	def merge_cli_overrides(argv: List[str], overrides: dict) -> List[str]:
		merged = list(argv)
		for key, value in overrides.items():
			if value is None:
				continue
			flag = f"--{str(key).replace('_', '-')}"
			if isinstance(value, bool):
				if value:
					merged.append(flag)
			else:
				merged.extend([flag, str(value)])
		return merged

	def load_trajectory_with_derivatives_from_csv(path: str, time_col: str = 'Time_Total', 
												   pos_cols: Optional[List[str]] = None,
												   vel_cols: Optional[List[str]] = None,
												   acc_cols: Optional[List[str]] = None):
		"""Load trajectory CSV with positions, velocities, and accelerations.

		- path: csv file path
		- time_col: column name for time
		- pos_cols: list of position column names (e.g., ['J1_Position', 'J2_Position', ...])
		- vel_cols: list of velocity column names (e.g., ['J1_Velocity', 'J2_Velocity', ...])
		- acc_cols: list of acceleration column names (e.g., ['J1_Acceleration', 'J2_Acceleration', ...])
		Input is assumed to be in degrees for joint angle, velocity (deg/s) and acceleration (deg/s^2).
		Values are always converted to radians internally.

		Returns (t, q, qd, qdd) where each is a numpy array.
		"""
		with open(path, newline='') as fh:
			reader = csv.DictReader(fh)
			fields = reader.fieldnames
			
			# Auto-detect position, velocity, acceleration columns if not provided
			if pos_cols is None:
				pos_cols = [f for f in fields if '_Position' in f and f.startswith('J')]
			if vel_cols is None:
				vel_cols = [f for f in fields if '_Velocity' in f and f.startswith('J')]
			if acc_cols is None:
				acc_cols = [f for f in fields if '_Acceleration' in f and f.startswith('J')]
			
			t_list = []
			q_lists = [[] for _ in pos_cols]
			qd_lists = [[] for _ in vel_cols]
			qdd_lists = [[] for _ in acc_cols]
			
			for row in reader:
				t_list.append(float(row[time_col]))
				for i, col in enumerate(pos_cols):
					val = row[col].strip() if row[col] else '0'
					q_lists[i].append(float(val))
				for i, col in enumerate(vel_cols):
					val = row[col].strip() if row[col] else '0'
					qd_lists[i].append(float(val))
				for i, col in enumerate(acc_cols):
					val = row[col].strip() if row[col] else '0'
					qdd_lists[i].append(float(val))
		
		t = np.array(t_list, dtype=float)
		q = np.vstack(q_lists).T
		qd = np.vstack(qd_lists).T
		qdd = np.vstack(qdd_lists).T

		# Determine joint types from pos column names (default assumption: J1/J2/J4 are revolute, J3 is prismatic)
		# pos_cols example: ['J1_Position','J2_Position','J3_Position','J4_Position']
		joint_types: List[str] = []  # 'R' for revolute, 'P' for prismatic
		for col in pos_cols:
			base = col.split('_')[0]
			jtype = 'R'
			if base.startswith('J'):
				try:
					idx = int(base[1:])
					if idx == 3:
						jtype = 'P'
				except Exception:
					pass
			# treat explicit 'Z' or 'Prismatic' tokens as prismatic
			if base.upper().startswith('Z') or 'PRISM' in base.upper():
				jtype = 'P'
			joint_types.append(jtype)

		print(f"Interpreting joints (pos cols): {pos_cols} -> types {joint_types}")

		# Convert revolute joint columns from degrees->radians only; leave prismatic joint columns (meters) unchanged
		for j, jt in enumerate(joint_types):
			if jt == 'R':
				q[:, j] = q[:, j] * np.pi / 180.0
				qd[:, j] = qd[:, j] * np.pi / 180.0
				qdd[:, j] = qdd[:, j] * np.pi / 180.0
			else:
				# Prismatic joint: convert from millimeters to meters (positions, velocities, accelerations)
				q[:, j] = q[:, j] * 1e-3
				qd[:, j] = qd[:, j] * 1e-3
				qdd[:, j] = qdd[:, j] * 1e-3
				print(f"Converted prismatic joint column {pos_cols[j]} from mm->m (unconditional)")
		
		# Filter out rows where time differences are zero or negative
		if len(t) > 1:
			dt = np.diff(t)
			valid_mask = np.concatenate([[True], dt > 1e-9])
			t = t[valid_mask]
			q = q[valid_mask]
			qd = qd[valid_mask]
			qdd = qdd[valid_mask]
			if np.sum(~valid_mask) > 0:
				print(f"Warning: Removed {np.sum(~valid_mask)} rows with zero or negative time intervals")
		
		return t, q, qd, qdd

	def load_trajectory_from_csv(path: str, time_col: Optional[str] = None, joint_cols: Optional[List[str]] = None, time_is_interval: bool = False):
		"""Load trajectory CSV and return (t, q).

		- path: csv file path
		- time_col: column name for time; if None, first column is used
		- joint_cols: list of column names to use as joint values. If None,
		  all columns except time are used in left-to-right order.
		- time_is_interval: if True, treat time column as intervals/deltas and compute cumulative time

		The CSV may have a header row. Returns t (N,) and q (N, n_joints).
		"""
		with open(path, newline='') as fh:
			reader = csv.DictReader(fh)
			if reader.fieldnames is None:
				# fallback to numeric load without header
				arr = np.loadtxt(path, delimiter=',')
				t = arr[:, 0]
				q = arr[:, 1:]
				if time_is_interval:
					t = np.cumsum(t)
				return t, q

			fields = reader.fieldnames
			# determine time column
			if time_col is None:
				time_col = fields[0]

			# determine joint columns
			if joint_cols is None:
				joint_cols = [f for f in fields if f != time_col]

			t_list = []
			q_lists = [[] for _ in joint_cols]
			for row in reader:
				t_list.append(float(row[time_col]))
				for i, jc in enumerate(joint_cols):
					q_lists[i].append(float(row[jc]))

		t = np.array(t_list, dtype=float)
		# if time column contains intervals, compute cumulative sum
		if time_is_interval:
			t = np.cumsum(t)
		q = np.vstack(q_lists).T
		
		# Filter out rows where time differences are zero or negative (causes divide by zero)
		if len(t) > 1:
			dt = np.diff(t)
			valid_mask = np.concatenate([[True], dt > 1e-9])  # keep first row and rows with positive dt
			t = t[valid_mask]
			q = q[valid_mask]
			if np.sum(~valid_mask) > 0:
				print(f"Warning: Removed {np.sum(~valid_mask)} rows with zero or negative time intervals")
		
		return t, q

	def load_calibration_power_csv(path: str):
		"""Load calibration CSV (e.g., from power meter) and return (t, power).
		
		Expected columns: 'Time' (or similar) and 'P[W]' power column.
		Returns t (N,) and power (N,) arrays.
		"""
		import pandas as pd
		
		# Read the entire file and find the header line
		with open(path, 'r') as f:
			lines = f.readlines()
		
		# Find the header line (contains URMS, IRMS, P[W], Time)
		header_idx = None
		for i, line in enumerate(lines):
			if 'P[W]' in line and 'Time' in line and not line.startswith('#'):
				header_idx = i
				break
		
		if header_idx is None:
			raise ValueError(f"Could not find header row in {path}")
		
		# Use pandas to read from that header line, skip all lines before it
		df = pd.read_csv(path, skiprows=header_idx)
		
		# Find time and power columns
		time_col = None
		power_col = None
		
		for col in df.columns:
			if col.strip() == 'Time':
				time_col = col
			if col.strip() == 'P[W]':
				power_col = col
		
		if time_col is None or power_col is None:
			raise ValueError(f"Could not find time ({time_col}) and power ({power_col}) columns in {path}. Found columns: {df.columns.tolist()}")
		
		# Extract arrays and convert properly
		t = pd.to_numeric(df[time_col], errors='coerce').values
		# Handle scientific notation in power column
		power_str = df[power_col].astype(str).str.replace('E', 'e')
		power = pd.to_numeric(power_str, errors='coerce').values
		
		# Remove NaN values
		valid_mask = ~(np.isnan(t) | np.isnan(power))
		t = t[valid_mask]
		power = power[valid_mask]
		
		return t, power

	def auto_fit_j1_params(
		t: np.ndarray,
		q: np.ndarray,
		qd: np.ndarray,
		qdd: np.ndarray,
		t_cal: np.ndarray,
		power_cal: np.ndarray,
		base_viscous: List[float],
		base_speed_sq: List[float],
		base_accel_scale: List[float],
		base_speed_exp: float,
		sample_stride: int = 2,
	):
		"""Grid-search J1 parameters to fit simulated total power to measured power."""

		def eval_rmse(j1_visc: float, j1_speed_sq: float, j1_accel_scale: float, speed_exp: float, stride: int) -> float:
			visc_fit = list(base_viscous)
			speed_fit = list(base_speed_sq)
			accel_fit = list(base_accel_scale)
			visc_fit[0] = float(max(j1_visc, 1e-9))
			speed_fit[0] = float(max(j1_speed_sq, 1e-9))
			accel_fit[0] = float(max(j1_accel_scale, 1e-9))

			model_fit = SCARAEnergyModel(
				l1=HARD_L1,
				l2=HARD_L2,
				m1=args.m1,
				m2=args.m2,
				I1=args.I1,
				I2=args.I2,
				g=SIM_GRAVITY,
				motor_eff=SIM_MOTOR_EFF,
				regen_eff=SIM_REGEN_EFF,
				constant_power=args.constant_power,
				viscous_friction=visc_fit,
				speed_squared_loss=speed_fit,
				accel_scale=accel_fit,
				speed_loss_exponent=float(speed_exp),
				scara_horizontal=SIM_SCARA_HORIZONTAL,
				use_power_to_brake=SIM_USE_POWER_TO_BRAKE,
				brake_efficiency=SIM_BRAKE_EFFICIENCY,
				quill_mass=SIM_QUILL_MASS,
				quill_gravity_moving_only=SIM_QUILL_GRAVITY_MOVING_ONLY,
				quill_deadband=SIM_QUILL_DEADBAND,
				j3_low_speed_boost=0.0,
				j3_high_speed_boost=0.0,
				j3_shape_ref_speed=1.0,
				j3_shape_exp=2.0,
			)

			idx = np.arange(0, len(t), max(int(stride), 1))
			t_fit = t[idx]
			sim_power = np.zeros_like(t_fit)
			for ii, k in enumerate(idx):
				sim_power[ii] = model_fit.torques(q[k], qd[k], qdd[k])[2]

			cal_interp = np.interp(t_fit, t_cal, power_cal)
			valid = (t_fit >= t_cal[0]) & (t_fit <= t_cal[-1])
			if np.sum(valid) < 10:
				return float('inf')
			err = sim_power[valid] - cal_interp[valid]
			return float(np.sqrt(np.mean(err ** 2)))

		base_v = max(float(base_viscous[0]), 1e-6)
		base_k = max(float(base_speed_sq[0]), 1e-6)
		base_a = max(float(base_accel_scale[0]), 1e-6)
		base_n = float(base_speed_exp)

		visc_candidates = base_v * np.array([0.4, 0.7, 1.0, 1.4, 2.0])
		speed_candidates = base_k * np.array([0.4, 0.7, 1.0, 1.4, 2.0])
		accel_candidates = base_a * np.array([0.25, 0.5, 0.75, 1.0, 1.25])
		exp_candidates = np.array([max(1.2, base_n - 0.8), max(1.2, base_n - 0.4), base_n, base_n + 0.4, base_n + 0.8])

		rmse_base = eval_rmse(base_v, base_k, base_a, base_n, stride=1)
		best = (base_v, base_k, base_a, base_n)
		best_rmse = float('inf')

		for v in visc_candidates:
			for k in speed_candidates:
				for a in accel_candidates:
					for n in exp_candidates:
						rmse = eval_rmse(v, k, a, float(n), stride=sample_stride)
						if rmse < best_rmse:
							best_rmse = rmse
							best = (float(v), float(k), float(a), float(n))

		# local refinement around best candidate
		v0, k0, a0, n0 = best
		visc_ref = np.array([max(1e-6, v0 * 0.85), v0, v0 * 1.15])
		speed_ref = np.array([max(1e-6, k0 * 0.85), k0, k0 * 1.15])
		accel_ref = np.array([max(1e-6, a0 * 0.85), a0, a0 * 1.15])
		exp_ref = np.array([max(1.2, n0 - 0.2), n0, n0 + 0.2])

		for v in visc_ref:
			for k in speed_ref:
				for a in accel_ref:
					for n in exp_ref:
						rmse = eval_rmse(float(v), float(k), float(a), float(n), stride=1)
						if rmse < best_rmse:
							best_rmse = rmse
							best = (float(v), float(k), float(a), float(n))

		return {
			'base_rmse': rmse_base,
			'best_rmse': best_rmse,
			'j1_viscous': best[0],
			'j1_speed_sq': best[1],
			'j1_accel_scale': best[2],
			'speed_exp': best[3],
		}

	parser = argparse.ArgumentParser(description="SCARA energy model utility")
	parser.add_argument('--csv', type=str, help='CSV file with time and joint columns')
	parser.add_argument('--calibration', type=str, help='Calibration CSV file with actual power measurements (e.g., CalibrationJ1Power)')
	parser.add_argument('--j3-debug-csv', type=str, default=None, help='Optional output CSV path for J3 diagnostics (time, velocity, force, mechanical/electrical power)')
	parser.add_argument('--time-col', type=str, default=None, help='Time column name (default: first column)')
	parser.add_argument('--joints', type=str, default=None, help='Comma-separated joint column names (default: all except time)')
	parser.add_argument('--time-is-interval', action='store_true', help='Treat time column as intervals/deltas (compute cumulative time)')
	parser.add_argument('--l1', type=float, default=SIM_L1, help='Link 1 length in meters')
	parser.add_argument('--l2', type=float, default=SIM_L2, help='Link 2 length in meters')
	# Updated defaults per user request: joint/link masses 15 kg (link1) and 10 kg (link2)
	parser.add_argument('--m1', type=float, default=SIM_M1, help='Link 1 mass in kg')
	parser.add_argument('--m2', type=float, default=SIM_M2, help='Link 2 mass in kg')
	# Inertias: if not provided, they will be auto-computed as slender rods about COM: I = (1/12)*m*L^2
	parser.add_argument('--I1', type=float, default=SIM_I1, help='Link 1 inertia about COM (kg*m^2), None uses slender-rod estimate')
	parser.add_argument('--I2', type=float, default=SIM_I2, help='Link 2 inertia about COM (kg*m^2), None uses slender-rod estimate')
	parser.add_argument('--constant-power', type=float, default=SIM_CONSTANT_POWER, help='Constant controller power draw in watts')
	# Friction and loss parameters
	parser.add_argument('--viscous-j1', type=float, default=SIM_VISCOUS[0], help='J1 viscous friction coefficient (Nm/(rad/s))')
	parser.add_argument('--viscous-j2', type=float, default=SIM_VISCOUS[1], help='J2 viscous friction coefficient (Nm/(rad/s))')
	parser.add_argument('--viscous-j3', type=float, default=SIM_VISCOUS[2], help='J3 viscous friction coefficient (N/(m/s))')
	parser.add_argument('--viscous-j4', type=float, default=SIM_VISCOUS[3], help='J4 viscous friction coefficient (Nm/(rad/s))')
	parser.add_argument('--speed-sq-j1', type=float, default=SIM_SPEED_SQ[0], help='J1 velocity-squared loss coefficient (W/(rad/s)^2)')
	parser.add_argument('--speed-sq-j2', type=float, default=SIM_SPEED_SQ[1], help='J2 velocity-squared loss coefficient (W/(rad/s)^2)')
	parser.add_argument('--speed-sq-j3', type=float, default=SIM_SPEED_SQ[2], help='J3 velocity-squared loss coefficient (W/(m/s)^2)')
	parser.add_argument('--speed-sq-j4', type=float, default=SIM_SPEED_SQ[3], help='J4 velocity-squared loss coefficient (W/(rad/s)^2)')
	parser.add_argument('--accel-scale-j1', type=float, default=SIM_ACCEL_SCALE[0], help='J1 acceleration influence scale (multiplies qdd torque contribution)')
	parser.add_argument('--accel-scale-j2', type=float, default=SIM_ACCEL_SCALE[1], help='J2 acceleration influence scale (multiplies qdd torque contribution)')
	parser.add_argument('--accel-scale-j3', type=float, default=SIM_ACCEL_SCALE[2], help='J3 acceleration influence scale (placeholder for non-core inertia model)')
	parser.add_argument('--accel-scale-j4', type=float, default=SIM_ACCEL_SCALE[3], help='J4 acceleration influence scale (placeholder for non-core inertia model)')
	parser.add_argument('--speed-loss-exp', type=float, default=SIM_SPEED_EXP, help='Velocity-loss exponent n in P_loss = k*|qd|^n')
	parser.add_argument('--auto-fit-j1', action='store_true', help='Auto-fit J1 velocity/acceleration decay parameters to calibration data')
	parser.add_argument('--fit-sample-stride', type=int, default=2, help='Sample stride for coarse auto-fit sweep (lower is slower, more accurate)')
	parser.add_argument('--j1-low-speed-boost', type=float, default=SIM_J1_LOW_SPEED_BOOST, help='J1 low-speed electrical power boost fraction (e.g., 0.25 = +25%% near zero speed)')
	parser.add_argument('--j1-high-speed-reduction', type=float, default=SIM_J1_HIGH_SPEED_REDUCTION, help='J1 high-speed electrical power reduction fraction (e.g., 0.20 = -20%% at high speed)')
	parser.add_argument('--j1-shape-ref-speed', type=float, default=SIM_J1_SHAPE_REF_SPEED, help='J1 speed (rad/s) where low/high speed shaping transitions')
	parser.add_argument('--j1-shape-exp', type=float, default=SIM_J1_SHAPE_EXP, help='J1 shaping sharpness exponent (>0)')
	parser.add_argument('--j2-low-speed-boost', type=float, default=SIM_J2_LOW_SPEED_BOOST, help='J2 low-speed electrical power boost fraction (e.g., 0.25 = +25%% near zero speed)')
	parser.add_argument('--j2-high-speed-reduction', type=float, default=SIM_J2_HIGH_SPEED_REDUCTION, help='J2 high-speed electrical power reduction fraction (e.g., 0.20 = -20%% at high speed)')
	parser.add_argument('--j2-shape-ref-speed', type=float, default=SIM_J2_SHAPE_REF_SPEED, help='J2 speed (rad/s) where low/high speed shaping transitions')
	parser.add_argument('--j2-shape-exp', type=float, default=SIM_J2_SHAPE_EXP, help='J2 shaping sharpness exponent (>0)')
	parser.add_argument('--j3-low-speed-boost', type=float, default=SIM_J3_LOW_SPEED_BOOST, help='J3 low-speed electrical power boost fraction (e.g., 0.25 = +25%% near zero speed)')
	parser.add_argument('--j3-high-speed-boost', type=float, default=SIM_J3_HIGH_SPEED_BOOST, help='J3 high-speed electrical power boost fraction (e.g., 0.20 = +20%% at high speed)')
	parser.add_argument('--j3-high-speed-reduction', dest='j3_high_speed_boost', type=float, help=argparse.SUPPRESS)
	parser.add_argument('--j3-shape-ref-speed', type=float, default=SIM_J3_SHAPE_REF_SPEED, help='J3 speed (m/s) where low/high speed shaping transitions')
	parser.add_argument('--j3-shape-exp', type=float, default=SIM_J3_SHAPE_EXP, help='J3 shaping sharpness exponent (>0)')
	parser.add_argument('--j4-low-speed-boost', type=float, default=SIM_J4_LOW_SPEED_BOOST, help='J4 low-speed electrical power boost fraction (use negative for reduction near zero speed)')
	parser.add_argument('--j4-high-speed-boost', type=float, default=SIM_J4_HIGH_SPEED_BOOST, help='J4 high-speed electrical power adjustment fraction (e.g., 0.20 = +20%%, -0.20 = -20%% at high speed)')
	parser.add_argument('--j4-shape-ref-speed', type=float, default=SIM_J4_SHAPE_REF_SPEED, help='J4 speed (rad/s) where low/high speed shaping transitions')
	parser.add_argument('--j4-shape-exp', type=float, default=SIM_J4_SHAPE_EXP, help='J4 shaping sharpness exponent (>0)')
	active_overrides = {k: v for k, v in SIM_CLI_OVERRIDE_FLAGS.items() if v is not None}
	if active_overrides:
		print(f"Applying in-code CLI overrides: {active_overrides}")
	args = parser.parse_args(merge_cli_overrides(sys.argv[1:], active_overrides))

	# Link lengths used by this model run
	HARD_L1 = SIM_L1
	HARD_L2 = SIM_L2

	# Inertias now auto-computed in SCARAEnergyModel if None; pass through args.I1/args.I2 directly.

	if args.csv:
		# run with hardcoded link lengths (user-provided); masses/inertias can still be set via CLI
		viscous_friction = [args.viscous_j1, args.viscous_j2, args.viscous_j3, args.viscous_j4]
		speed_squared_loss = [args.speed_sq_j1, args.speed_sq_j2, args.speed_sq_j3, args.speed_sq_j4]
		accel_scale = [args.accel_scale_j1, args.accel_scale_j2, args.accel_scale_j3, args.accel_scale_j4]
		speed_loss_exp_run = args.speed_loss_exp
		
		model = SCARAEnergyModel(
			l1=HARD_L1,
			l2=HARD_L2,
			m1=args.m1,
			m2=args.m2,
			I1=args.I1,
			I2=args.I2,
			g=SIM_GRAVITY,
			motor_eff=SIM_MOTOR_EFF,
			regen_eff=SIM_REGEN_EFF,
			constant_power=args.constant_power,
			viscous_friction=viscous_friction,
			speed_squared_loss=speed_squared_loss,
			accel_scale=accel_scale,
			speed_loss_exponent=speed_loss_exp_run,
			scara_horizontal=SIM_SCARA_HORIZONTAL,
			use_power_to_brake=SIM_USE_POWER_TO_BRAKE,
			brake_efficiency=SIM_BRAKE_EFFICIENCY,
			quill_mass=SIM_QUILL_MASS,
			quill_gravity_moving_only=SIM_QUILL_GRAVITY_MOVING_ONLY,
			quill_deadband=SIM_QUILL_DEADBAND,
			j1_low_speed_boost=args.j1_low_speed_boost,
			j1_high_speed_reduction=args.j1_high_speed_reduction,
			j1_shape_ref_speed=args.j1_shape_ref_speed,
			j1_shape_exp=args.j1_shape_exp,
			j2_low_speed_boost=args.j2_low_speed_boost,
			j2_high_speed_reduction=args.j2_high_speed_reduction,
			j2_shape_ref_speed=args.j2_shape_ref_speed,
			j2_shape_exp=args.j2_shape_exp,
			j3_low_speed_boost=args.j3_low_speed_boost,
			j3_high_speed_boost=args.j3_high_speed_boost,
			j3_shape_ref_speed=args.j3_shape_ref_speed,
			j3_shape_exp=args.j3_shape_exp,
			j4_low_speed_boost=args.j4_low_speed_boost,
			j4_high_speed_boost=args.j4_high_speed_boost,
			j4_shape_ref_speed=args.j4_shape_ref_speed,
			j4_shape_exp=args.j4_shape_exp,
			# defaults: servo braking, brake_eff=0.9, quill_mass=3.0, gravity moving-only
		)
		print(f"Effective inertias: I1={model.I1:.6f} kg*m^2, I2={model.I2:.6f} kg*m^2")
		print(f"Viscous friction: J1={viscous_friction[0]:.2f}, J2={viscous_friction[1]:.2f}, J3={viscous_friction[2]:.2f}, J4={viscous_friction[3]:.2f}")
		print(f"Speed-squared loss: J1={speed_squared_loss[0]:.2f}, J2={speed_squared_loss[1]:.2f}, J3={speed_squared_loss[2]:.2f}, J4={speed_squared_loss[3]:.2f}")
		print(f"Acceleration scale: J1={accel_scale[0]:.2f}, J2={accel_scale[1]:.2f}, J3={accel_scale[2]:.2f}, J4={accel_scale[3]:.2f}")
		print(f"Velocity-loss exponent: n={speed_loss_exp_run:.2f}")
		print(
			f"J1 speed shaping: low_boost={args.j1_low_speed_boost:.3f}, "
			f"high_reduction={args.j1_high_speed_reduction:.3f}, "
			f"ref_speed={args.j1_shape_ref_speed:.3f} rad/s, exp={args.j1_shape_exp:.2f}"
		)
		print(
			f"J1 gain checkpoints: g(0)={model.j1_speed_gain(0.0):.3f}, "
			f"g(ref)={model.j1_speed_gain(args.j1_shape_ref_speed):.3f}, "
			f"g(2*ref)={model.j1_speed_gain(2.0 * args.j1_shape_ref_speed):.3f}"
		)
		print(
			f"J2 speed shaping: low_boost={args.j2_low_speed_boost:.3f}, "
			f"high_reduction={args.j2_high_speed_reduction:.3f}, "
			f"ref_speed={args.j2_shape_ref_speed:.3f} rad/s, exp={args.j2_shape_exp:.2f}"
		)
		print(
			f"J2 gain checkpoints: g(0)={model.j2_speed_gain(0.0):.3f}, "
			f"g(ref)={model.j2_speed_gain(args.j2_shape_ref_speed):.3f}, "
			f"g(2*ref)={model.j2_speed_gain(2.0 * args.j2_shape_ref_speed):.3f}"
		)
		print(
			f"J3 speed shaping: low_boost={args.j3_low_speed_boost:.3f}, "
			f"high_boost={args.j3_high_speed_boost:.3f}, "
			f"ref_speed={args.j3_shape_ref_speed:.3f} m/s, exp={args.j3_shape_exp:.2f}"
		)
		print(
			f"J3 gain checkpoints: g(0)={model.j3_speed_gain(0.0):.3f}, "
			f"g(ref)={model.j3_speed_gain(args.j3_shape_ref_speed):.3f}, "
			f"g(2*ref)={model.j3_speed_gain(2.0 * args.j3_shape_ref_speed):.3f}"
		)
		print(
			f"J4 speed shaping: low_boost={args.j4_low_speed_boost:.3f}, "
			f"high_boost={args.j4_high_speed_boost:.3f}, "
			f"ref_speed={args.j4_shape_ref_speed:.3f} rad/s, exp={args.j4_shape_exp:.2f}"
		)
		print(
			f"J4 gain checkpoints: g(0)={model.j4_speed_gain(0.0):.3f}, "
			f"g(ref)={model.j4_speed_gain(args.j4_shape_ref_speed):.3f}, "
			f"g(2*ref)={model.j4_speed_gain(2.0 * args.j4_shape_ref_speed):.3f}"
		)
		if SIM_PLOT_SPEED_SHAPING_GRAPHS:
			# Visualize speed-shaping multipliers for J1, J2, J3, and J4 in separate figures
			speed_max_j1 = max(3.0 * float(args.j1_shape_ref_speed), 3.0)
			speed_grid_j1 = np.linspace(0.0, speed_max_j1, 400)
			gain_grid_j1 = np.array([model.j1_speed_gain(v) for v in speed_grid_j1])
			fig_gain_j1, ax_gain_j1 = plt.subplots(1, 1, figsize=(8, 4.5))
			ax_gain_j1.plot(speed_grid_j1, gain_grid_j1, color='tab:blue', linewidth=2.2, label='J1 gain(speed)')
			ax_gain_j1.axvline(float(args.j1_shape_ref_speed), color='gray', linestyle='--', linewidth=1.3, label='ref speed')
			ax_gain_j1.axhline(1.0, color='black', linestyle=':', linewidth=1.0, label='no scaling (gain=1)')
			ax_gain_j1.set_title('J1 Speed-Shaping Multiplier', fontsize=12, fontweight='bold')
			ax_gain_j1.set_xlabel('Absolute J1 Speed (rad/s)', fontsize=11)
			ax_gain_j1.set_ylabel('Multiplier Gain', fontsize=11)
			ax_gain_j1.grid(True, alpha=0.3)
			ax_gain_j1.legend(fontsize=9)
			plt.tight_layout()

			speed_max_j2 = max(3.0 * float(args.j2_shape_ref_speed), 3.0)
			speed_grid_j2 = np.linspace(0.0, speed_max_j2, 400)
			gain_grid_j2 = np.array([model.j2_speed_gain(v) for v in speed_grid_j2])
			fig_gain_j2, ax_gain_j2 = plt.subplots(1, 1, figsize=(8, 4.5))
			ax_gain_j2.plot(speed_grid_j2, gain_grid_j2, color='tab:green', linewidth=2.2, label='J2 gain(speed)')
			ax_gain_j2.axvline(float(args.j2_shape_ref_speed), color='gray', linestyle='--', linewidth=1.3, label='ref speed')
			ax_gain_j2.axhline(1.0, color='black', linestyle=':', linewidth=1.0, label='no scaling (gain=1)')
			ax_gain_j2.set_title('J2 Speed-Shaping Multiplier', fontsize=12, fontweight='bold')
			ax_gain_j2.set_xlabel('Absolute J2 Speed (rad/s)', fontsize=11)
			ax_gain_j2.set_ylabel('Multiplier Gain', fontsize=11)
			ax_gain_j2.grid(True, alpha=0.3)
			ax_gain_j2.legend(fontsize=9)
			plt.tight_layout()

			speed_max_j3 = max(3.0 * float(args.j3_shape_ref_speed), 3.0)
			speed_grid_j3 = np.linspace(0.0, speed_max_j3, 400)
			gain_grid_j3 = np.array([model.j3_speed_gain(v) for v in speed_grid_j3])
			fig_gain_j3, ax_gain_j3 = plt.subplots(1, 1, figsize=(8, 4.5))
			ax_gain_j3.plot(speed_grid_j3, gain_grid_j3, color='purple', linewidth=2.2, label='J3 gain(speed)')
			ax_gain_j3.axvline(float(args.j3_shape_ref_speed), color='gray', linestyle='--', linewidth=1.3, label='ref speed')
			ax_gain_j3.axhline(1.0, color='black', linestyle=':', linewidth=1.0, label='no scaling (gain=1)')
			ax_gain_j3.set_title('J3 Speed-Shaping Multiplier', fontsize=12, fontweight='bold')
			ax_gain_j3.set_xlabel('Absolute J3 Speed (m/s)', fontsize=11)
			ax_gain_j3.set_ylabel('Multiplier Gain', fontsize=11)
			ax_gain_j3.grid(True, alpha=0.3)
			ax_gain_j3.legend(fontsize=9)
			plt.tight_layout()

			speed_max_j4 = max(3.0 * float(args.j4_shape_ref_speed), 3.0)
			speed_grid_j4 = np.linspace(0.0, speed_max_j4, 400)
			gain_grid_j4 = np.array([model.j4_speed_gain(v) for v in speed_grid_j4])
			fig_gain_j4, ax_gain_j4 = plt.subplots(1, 1, figsize=(8, 4.5))
			ax_gain_j4.plot(speed_grid_j4, gain_grid_j4, color='tab:red', linewidth=2.2, label='J4 gain(speed)')
			ax_gain_j4.axvline(float(args.j4_shape_ref_speed), color='gray', linestyle='--', linewidth=1.3, label='ref speed')
			ax_gain_j4.axhline(1.0, color='black', linestyle=':', linewidth=1.0, label='no scaling (gain=1)')
			ax_gain_j4.set_title('J4 Speed-Shaping Multiplier', fontsize=12, fontweight='bold')
			ax_gain_j4.set_xlabel('Absolute J4 Speed (rad/s)', fontsize=11)
			ax_gain_j4.set_ylabel('Multiplier Gain', fontsize=11)
			ax_gain_j4.grid(True, alpha=0.3)
			ax_gain_j4.legend(fontsize=9)
			plt.tight_layout()
		
		# Load position, velocity, and acceleration data from CSV
		# Auto-detect columns and use Time_Total as time column by default
		time_col = args.time_col if args.time_col else 'Time_Total'
		t, q, qd, qdd = load_trajectory_with_derivatives_from_csv(
			args.csv, 
			time_col=time_col
		)
		
		print(f"Loaded {len(t)} time steps from {args.csv}")
		print(f"Time range: {t[0]:.3f} to {t[-1]:.3f} seconds")
		print(f"Number of joints: {q.shape[1]}")
		if qd.shape[1] > 0 and (args.j1_low_speed_boost > 0.0 or args.j1_high_speed_reduction > 0.0):
			w_j1 = np.abs(qd[:, 0])
			x = (w_j1 / max(args.j1_shape_ref_speed, 1e-9)) ** max(args.j1_shape_exp, 0.5)
			r = x / (1.0 + x)
			low_active = float(np.mean(1.0 - r))
			high_active = float(np.mean(r))
			print(f"J1 shaping activity: low-region={low_active:.3f}, high-region={high_active:.3f} (higher low-region means boost has stronger effect)")
		if qd.shape[1] > 1 and (args.j2_low_speed_boost > 0.0 or args.j2_high_speed_reduction > 0.0):
			w_j2 = np.abs(qd[:, 1])
			x2 = (w_j2 / max(args.j2_shape_ref_speed, 1e-9)) ** max(args.j2_shape_exp, 0.5)
			r2 = x2 / (1.0 + x2)
			low_active2 = float(np.mean(1.0 - r2))
			high_active2 = float(np.mean(r2))
			print(f"J2 shaping activity: low-region={low_active2:.3f}, high-region={high_active2:.3f} (higher low-region means boost has stronger effect)")
		if qd.shape[1] > 2 and (args.j3_low_speed_boost > 0.0 or args.j3_high_speed_boost > 0.0):
			w_j3 = np.abs(qd[:, 2])
			x3 = (w_j3 / max(args.j3_shape_ref_speed, 1e-9)) ** max(args.j3_shape_exp, 0.5)
			r3 = x3 / (1.0 + x3)
			low_active3 = float(np.mean(1.0 - r3))
			high_active3 = float(np.mean(r3))
			print(f"J3 shaping activity: low-region={low_active3:.3f}, high-region={high_active3:.3f} (higher low-region means boost has stronger effect)")
		if qd.shape[1] > 3 and (args.j4_low_speed_boost != 0.0 or args.j4_high_speed_boost != 0.0):
			w_j4 = np.abs(qd[:, 3])
			x4 = (w_j4 / max(args.j4_shape_ref_speed, 1e-9)) ** max(args.j4_shape_exp, 0.5)
			r4 = x4 / (1.0 + x4)
			low_active4 = float(np.mean(1.0 - r4))
			high_active4 = float(np.mean(r4))
			print(f"J4 shaping activity: low-region={low_active4:.3f}, high-region={high_active4:.3f} (higher low-region means boost has stronger effect)")

		t_cal = None
		power_cal = None
		if args.calibration:
			try:
				t_cal, power_cal = load_calibration_power_csv(args.calibration)
				print(f"Loaded calibration data from {args.calibration}")
				print(f"Calibration time range: {t_cal[0]:.3f} to {t_cal[-1]:.3f} seconds")
				print(f"Calibration power range: {power_cal.min():.2f} to {power_cal.max():.2f} W")
			except Exception as e:
				print(f"Warning: Could not load calibration data: {e}")

		if args.auto_fit_j1:
			if t_cal is None or power_cal is None:
				print("Warning: --auto-fit-j1 requested but calibration data is unavailable. Skipping auto-fit.")
			else:
				print("Running J1 auto-fit (viscous, speed-loss gain, accel scale, exponent)...")
				fit = auto_fit_j1_params(
					t=t,
					q=q,
					qd=qd,
					qdd=qdd,
					t_cal=t_cal,
					power_cal=power_cal,
					base_viscous=viscous_friction,
					base_speed_sq=speed_squared_loss,
					base_accel_scale=accel_scale,
					base_speed_exp=speed_loss_exp_run,
					sample_stride=max(1, args.fit_sample_stride),
				)
				print(f"Auto-fit RMSE: baseline={fit['base_rmse']:.2f} W, best={fit['best_rmse']:.2f} W")
				viscous_friction[0] = fit['j1_viscous']
				speed_squared_loss[0] = fit['j1_speed_sq']
				accel_scale[0] = fit['j1_accel_scale']
				speed_loss_exp_run = fit['speed_exp']
				print(
					f"Fitted J1 params: viscous={viscous_friction[0]:.4f}, "
					f"speed_sq={speed_squared_loss[0]:.4f}, "
					f"accel_scale={accel_scale[0]:.4f}, n={speed_loss_exp_run:.4f}"
				)

				model = SCARAEnergyModel(
					l1=HARD_L1,
					l2=HARD_L2,
					m1=args.m1,
					m2=args.m2,
					I1=args.I1,
					I2=args.I2,
					g=SIM_GRAVITY,
					motor_eff=SIM_MOTOR_EFF,
					regen_eff=SIM_REGEN_EFF,
					constant_power=args.constant_power,
					viscous_friction=viscous_friction,
					speed_squared_loss=speed_squared_loss,
					accel_scale=accel_scale,
					speed_loss_exponent=speed_loss_exp_run,
					scara_horizontal=SIM_SCARA_HORIZONTAL,
					use_power_to_brake=SIM_USE_POWER_TO_BRAKE,
					brake_efficiency=SIM_BRAKE_EFFICIENCY,
					quill_mass=SIM_QUILL_MASS,
					quill_gravity_moving_only=SIM_QUILL_GRAVITY_MOVING_ONLY,
					quill_deadband=SIM_QUILL_DEADBAND,
					j1_low_speed_boost=args.j1_low_speed_boost,
					j1_high_speed_reduction=args.j1_high_speed_reduction,
					j1_shape_ref_speed=args.j1_shape_ref_speed,
					j1_shape_exp=args.j1_shape_exp,
					j2_low_speed_boost=args.j2_low_speed_boost,
					j2_high_speed_reduction=args.j2_high_speed_reduction,
					j2_shape_ref_speed=args.j2_shape_ref_speed,
					j2_shape_exp=args.j2_shape_exp,
					j3_low_speed_boost=args.j3_low_speed_boost,
					j3_high_speed_boost=args.j3_high_speed_boost,
					j3_shape_ref_speed=args.j3_shape_ref_speed,
					j3_shape_exp=args.j3_shape_exp,
					j4_low_speed_boost=args.j4_low_speed_boost,
					j4_high_speed_boost=args.j4_high_speed_boost,
					j4_shape_ref_speed=args.j4_shape_ref_speed,
					j4_shape_exp=args.j4_shape_exp,
				)
		
		# Calculate energy at each time step for plotting
		mech_power_samples = []
		elec_power_samples = []
		mech_power_j_samples = []  # list of per-joint mechanical power vectors
		elec_power_j_samples = []  # list of per-joint electrical power vectors
		tau_j3_samples = []
		for i in range(len(t)):
			# Updated torques() returns tau, mech_power_j, elec_total, elec_power_j
			tau, mech_power_j, elec_power, elec_power_j = model.torques(q[i], qd[i], qdd[i])
			mech_power_samples.append(np.sum(mech_power_j))
			elec_power_samples.append(elec_power)
			mech_power_j_samples.append(mech_power_j)
			elec_power_j_samples.append(elec_power_j)
			tau_j3_samples.append(float(tau[2]) if len(tau) > 2 else float('nan'))
		mech_power_samples = np.array(mech_power_samples)
		elec_power_samples = np.array(elec_power_samples)
		mech_power_j_samples = np.vstack(mech_power_j_samples)  # (N, n_joints)
		elec_power_j_samples = np.vstack(elec_power_j_samples)  # (N, n_joints)
		tau_j3_samples = np.array(tau_j3_samples, dtype=float)

		if args.j3_debug_csv:
			if qd.shape[1] <= 2:
				print("Warning: J3 debug export requested, but trajectory has no J3 column.")
			else:
				j3_vel = qd[:, 2]
				j3_mech = mech_power_j_samples[:, 2]
				j3_elec = elec_power_j_samples[:, 2]
				with open(args.j3_debug_csv, "w", newline="") as debug_fh:
					writer = csv.writer(debug_fh)
					writer.writerow([
						"Time_s",
						"J3_Velocity_mps",
						"J3_Force_N",
						"J3_MechPower_W",
						"J3_ElecPower_W",
					])
					for i in range(len(t)):
						writer.writerow([
							f"{t[i]:.9g}",
							f"{j3_vel[i]:.9g}",
							f"{tau_j3_samples[i]:.9g}",
							f"{j3_mech[i]:.9g}",
							f"{j3_elec[i]:.9g}",
						])

				up_mask = j3_vel > model.quill_deadband
				down_mask = j3_vel < -model.quill_deadband
				up_mean = float(np.mean(j3_elec[up_mask])) if np.any(up_mask) else float('nan')
				down_mean = float(np.mean(j3_elec[down_mask])) if np.any(down_mask) else float('nan')
				print(f"Wrote J3 diagnostics CSV: {args.j3_debug_csv}")
				print(
					f"J3 electrical power mean: up={up_mean:.3f} W, down={down_mean:.3f} W "
					f"(deadband={model.quill_deadband:g} m/s)"
				)

		# Total energies can be computed but are not printed per user's instruction to output just per-joint usage.
		E_mech = np.trapezoid(mech_power_samples, t)
		E_elec = np.trapezoid(elec_power_samples, t)
		# Print total cumulative energies (mechanical and electrical)
		print(f"Total cumulative mechanical energy = {E_mech:.2f} J")
		print(f"Total cumulative electrical energy = {E_elec:.2f} J")

		# Plot energy usage over time - compute cumulative total energy using trapezoidal integration
		cumulative_mech = np.zeros(len(t))
		cumulative_elec = np.zeros(len(t))
		# Per-joint cumulative energies
		cumulative_mech_j = np.zeros_like(mech_power_j_samples)
		cumulative_elec_j = np.zeros_like(elec_power_j_samples)
		for i in range(1, len(t)):
			dt = t[i] - t[i-1]
			cumulative_mech[i] = cumulative_mech[i-1] + (mech_power_samples[i-1] + mech_power_samples[i]) * dt / 2
			cumulative_elec[i] = cumulative_elec[i-1] + (elec_power_samples[i-1] + elec_power_samples[i]) * dt / 2
			cumulative_mech_j[i] = cumulative_mech_j[i-1] + (mech_power_j_samples[i-1] + mech_power_j_samples[i]) * dt / 2
			cumulative_elec_j[i] = cumulative_elec_j[i-1] + (elec_power_j_samples[i-1] + elec_power_j_samples[i]) * dt / 2

		# Console summary: per-joint mechanical & electrical energy usage
		print("Per-joint energy breakdown:")
		for j in range(mech_power_j_samples.shape[1]):
			E_mj = cumulative_mech_j[-1, j]
			E_ej = cumulative_elec_j[-1, j]
			print(f"  Joint {j+1}: mechanical energy = {E_mj:.2f} J, electrical energy = {E_ej:.2f} J")
		if model.constant_power != 0.0:
			E_const = np.trapezoid(np.ones_like(t) * model.constant_power, t)
			print(f"  Constant power draw energy = {E_const:.2f} J (not attributed to joints)")
		
		# Removed total/torque figure per user's request for just per-joint power usage.

		# Figure: Total electrical power (all joints + idle) with calibration overlay
		fig, ax = plt.subplots(1, 1, figsize=(12, 6))
		
		# Calculate total power: all joint power + constant power
		total_elec_power = elec_power_j_samples.sum(axis=1) + model.constant_power
		ax.plot(t, total_elec_power, label='Simulated Total Power (J1+J2+J3+J4+Idle)', linewidth=2.5, color='blue')
		
		# Overlay calibration data if provided
		if t_cal is not None and power_cal is not None:
			ax.plot(t_cal, power_cal, label='Calibration (Measured)', linewidth=2.5, color='red', linestyle='--', alpha=0.8)
		
		ax.set_title('Joint 4 Power Calibration', fontsize=14, fontweight='bold')
		ax.set_xlabel('Time (s)', fontsize=12)
		ax.set_ylabel('Electrical Power (W)', fontsize=12)
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=11)
		plt.tight_layout()
		plt.show()
	else:
		# quick demo: simple sinusoidal 2R motion for 5 seconds using hardcoded link lengths
		viscous_friction = [args.viscous_j1, args.viscous_j2, args.viscous_j3, args.viscous_j4]
		speed_squared_loss = [args.speed_sq_j1, args.speed_sq_j2, args.speed_sq_j3, args.speed_sq_j4]
		accel_scale = [args.accel_scale_j1, args.accel_scale_j2, args.accel_scale_j3, args.accel_scale_j4]
		
		model = SCARAEnergyModel(
			l1=HARD_L1,
			l2=HARD_L2,
			m1=args.m1,
			m2=args.m2,
			I1=args.I1,
			I2=args.I2,
			g=SIM_GRAVITY,
			motor_eff=SIM_MOTOR_EFF,
			regen_eff=SIM_REGEN_EFF,
			constant_power=args.constant_power,
			viscous_friction=viscous_friction,
			speed_squared_loss=speed_squared_loss,
			accel_scale=accel_scale,
			speed_loss_exponent=args.speed_loss_exp,
			scara_horizontal=SIM_SCARA_HORIZONTAL,
			use_power_to_brake=SIM_USE_POWER_TO_BRAKE,
			brake_efficiency=SIM_BRAKE_EFFICIENCY,
			quill_mass=SIM_QUILL_MASS,
			quill_gravity_moving_only=SIM_QUILL_GRAVITY_MOVING_ONLY,
			quill_deadband=SIM_QUILL_DEADBAND,
			j1_low_speed_boost=args.j1_low_speed_boost,
			j1_high_speed_reduction=args.j1_high_speed_reduction,
			j1_shape_ref_speed=args.j1_shape_ref_speed,
			j1_shape_exp=args.j1_shape_exp,
			j2_low_speed_boost=args.j2_low_speed_boost,
			j2_high_speed_reduction=args.j2_high_speed_reduction,
			j2_shape_ref_speed=args.j2_shape_ref_speed,
			j2_shape_exp=args.j2_shape_exp,
			j4_low_speed_boost=args.j4_low_speed_boost,
			j4_high_speed_boost=args.j4_high_speed_boost,
			j4_shape_ref_speed=args.j4_shape_ref_speed,
			j4_shape_exp=args.j4_shape_exp,
			# defaults: servo braking, brake_eff=0.9, quill_mass=3.0, gravity moving-only
		)
		print(f"Effective inertias (demo): I1={model.I1:.6f} kg*m^2, I2={model.I2:.6f} kg*m^2")
		T = 5.0
		N = 501
		t = np.linspace(0, T, N)
		th1 = 0.3 * np.sin(2 * np.pi * 0.2 * t)
		th2 = 0.2 * np.sin(2 * np.pi * 0.2 * t + 0.5)
		q = np.vstack([th1, th2]).T
		E_mech, E_elec = model.energy_over_trajectory(t, q)
		print(f"Demo: mechanical energy = {E_mech:.2f} J, electrical energy = {E_elec:.2f} J")

