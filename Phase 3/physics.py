"""
First-principles physics engine.
Vectorized over Monte-Carlo samples for <10 ms per timestep.
"""

import numpy as np
from scipy.spatial.transform import Rotation as SciRotation
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json

# ── Physical Constants ──
SIGMA_SB = 5.670374419e-8   # Stefan-Boltzmann [W/m^2/K^4]
T_SPACE = 3.0               # Background temp [K]


@dataclass
class ThrusterConfig:
    position: np.ndarray      # (3,) body frame [m]
    direction: np.ndarray     # (3,) unit vector body frame
    max_thrust: float         # [N]
    max_mass_flow: float      # [kg/s]


@dataclass
class ThermalNodeConfig:
    capacitance: float        # [J/K]
    emissivity: float
    area: float               # [m^2]
    absorptivity: float
    q_gen_base: float         # [W]
    solar_exposure_area: float = 0.0
    initial_temperature: float = 293.15  # [K]


@dataclass
class PowerConfig:
    battery_capacity_wh: float
    initial_soc: float        # 0–1
    solar_panel_area: float   # [m^2]
    solar_efficiency: float
    solar_constant: float = 1361.0  # [W/m^2]
    base_load_w: float        # [W]
    panel_normal_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


@dataclass
class LifeSupportConfig:
    o2_gen_rate: float        # [kg/s]
    o2_consumption_rate: float# per crew [kg/s]
    co2_scrubber_rate: float  # [kg/s]
    h2o_recovery_rate: float  # [kg/s]
    h2o_perspiration_rate: float # per crew [kg/s]
    initial_o2: float         # [kg]
    initial_co2: float        # [kg]
    initial_h2o: float        # [kg]
    n_crew: int = 3
    leak_rate: float = 0.0    # [kg/s]


@dataclass
class SpacecraftConfig:
    mass: float               # [kg]
    inertia: np.ndarray       # (3,3) [kg·m^2]
    thrusters: List[ThrusterConfig]
    thermal_nodes: List[ThermalNodeConfig]
    power: PowerConfig
    life_support: LifeSupportConfig
    mean_motion: float        # [rad/s]
    keepout_radius: float = 10.0
    max_allowable_accel: float = 5.0  # [m/s^2]
    docking_axis_body: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    retro_axis_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    initial_propellant: float = 500.0  # [kg]

    def __post_init__(self):
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.n_thrusters = len(self.thrusters)
        self.n_thermal = len(self.thermal_nodes)
        self._build_tam()
        self._build_thermal_g()

    def _build_tam(self):
        """Thruster Allocation Matrix: [F; τ] = TAM · u  (6 × n_thrusters)"""
        self.tam = np.zeros((6, self.n_thrusters))
        for i, th in enumerate(self.thrusters):
            self.tam[0:3, i] = th.direction * th.max_thrust
            self.tam[3:6, i] = np.cross(th.position, th.direction) * th.max_thrust
        self.tam_pinv = np.linalg.pinv(self.tam)

    def _build_thermal_g(self):
        """Conductance matrix G (symmetric, rows sum to zero on diagonal)."""
        n = self.n_thermal
        self.thermal_g = np.zeros((n, n))
        for i in range(n - 1):
            g = 0.1  # default conductance [W/K]; override per node if needed
            self.thermal_g[i, i + 1] = g
            self.thermal_g[i + 1, i] = g
        for i in range(n):
            self.thermal_g[i, i] = -np.sum(self.thermal_g[i, :])


# ── State Layout ──
class StateLayout:
    """Hard-coded indices prevent off-by-one errors across the entire module."""
    def __init__(self, n_thermal: int):
        self.n_thermal = n_thermal
        self.idx_x, self.idx_y, self.idx_z = 0, 1, 2
        self.idx_vx, self.idx_vy, self.idx_vz = 3, 4, 5
        self.idx_qw, self.idx_qx, self.idx_qy, self.idx_qz = 6, 7, 8, 9
        self.idx_wx, self.idx_wy, self.idx_wz = 10, 11, 12
        self.idx_thermal = 13
        self.idx_soc = 13 + n_thermal
        self.idx_m_prop = 14 + n_thermal
        self.idx_m_o2 = 15 + n_thermal
        self.idx_m_co2 = 16 + n_thermal
        self.idx_m_h2o = 17 + n_thermal
        self.dim = 18 + n_thermal


# ── SO(3) Utilities ──
def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """v: (n,3) → (n,3,3)"""
    n = v.shape[0]
    K = np.zeros((n, 3, 3))
    K[:, 0, 1] = -v[:, 2]
    K[:, 0, 2] = v[:, 1]
    K[:, 1, 0] = v[:, 2]
    K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]
    K[:, 2, 1] = v[:, 0]
    return K


def so3_exp(omega: np.ndarray) -> np.ndarray:
    """Vectorized matrix exponential on so(3).  omega: (n,3) → (n,3,3)"""
    n = omega.shape[0]
    theta = np.linalg.norm(omega, axis=1, keepdims=True)  # (n,1)
    small = theta[:, 0] < 1e-8
    R_out = np.zeros((n, 3, 3))
    if np.any(small):
        R_out[small] = np.eye(3) + skew_symmetric(omega[small])
    large = ~small
    if np.any(large):
        k = omega[large] / theta[large]
        K = skew_symmetric(k)
        sin_t = np.sin(theta[large])[:, None, None]
        one_minus_cos = (1.0 - np.cos(theta[large]))[:, None, None]
        I = np.eye(3)[None, ...]
        R_out[large] = I + sin_t * K + one_minus_cos * np.einsum('mij,mjk->mik', K, K)
    return R_out


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternions q.  Both (n,3/4). q is [w,x,y,z]."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    v_rot = np.empty_like(v)
    v_rot[:, 0] = (1 - 2 * (y ** 2 + z ** 2)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
    v_rot[:, 1] = 2 * (x * y + w * z) * vx + (1 - 2 * (x ** 2 + z ** 2)) * vy + 2 * (y * z - w * x) * vz
    v_rot[:, 2] = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x ** 2 + y ** 2)) * vz
    return v_rot


# ── Physics Simulator ──
class PhysicsSimulator:
    def __init__(self, config: SpacecraftConfig, n_mc: int = 1):
        self.cfg = config
        self.n_mc = n_mc
        self.layout = StateLayout(config.n_thermal)
        self._last_state = None

    def initialize_state(self, pose: Dict, habitat: Optional[Dict] = None) -> np.ndarray:
        """Build (n_mc, state_dim) initial ensemble from perception pose + telemetry."""
        state = np.zeros((self.n_mc, self.layout.dim))

        # Position / velocity
        t_nom = np.asarray(pose['translation'], dtype=float)
        if self.n_mc > 1 and 'sigma_t' in pose:
            t_nom = t_nom + np.random.normal(0.0, pose['sigma_t'], (self.n_mc, 3))
        state[:, self.layout.idx_x:self.layout.idx_x + 3] = t_nom
        # Velocity defaults to zero unless provided
        if 'velocity' in pose:
            state[:, self.layout.idx_vx:self.layout.idx_vx + 3] = np.asarray(pose['velocity'])

        # Attitude
        if 'hopf_grid' in pose and self.n_mc > 1:
            q = self._sample_hopf(pose['hopf_grid'])
        else:
            q = np.asarray(pose['quaternion'])  # (4,) scalar-first [w,x,y,z]
            if q.ndim == 1:
                q = np.tile(q, (self.n_mc, 1))
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        state[:, self.layout.idx_qw:self.layout.idx_qw + 4] = q

        # Habitat defaults
        for i, node in enumerate(self.cfg.thermal_nodes):
            state[:, self.layout.idx_thermal + i] = node.initial_temperature
        state[:, self.layout.idx_soc] = self.cfg.power.initial_soc
        state[:, self.layout.idx_m_prop] = self.cfg.initial_propellant
        state[:, self.layout.idx_m_o2] = self.cfg.life_support.initial_o2
        state[:, self.layout.idx_m_co2] = self.cfg.life_support.initial_co2
        state[:, self.layout.idx_m_h2o] = self.cfg.life_support.initial_h2o

        # Override from live telemetry if provided
        if habitat:
            if 'thermal' in habitat:
                for i, temp in enumerate(habitat['thermal']):
                    if i < self.cfg.n_thermal:
                        state[:, self.layout.idx_thermal + i] = temp
            if 'soc' in habitat:
                state[:, self.layout.idx_soc] = habitat['soc']
            if 'propellant' in habitat:
                state[:, self.layout.idx_m_prop] = habitat['propellant']
            if 'o2' in habitat:
                state[:, self.layout.idx_m_o2] = habitat['o2']

        self._last_state = state.copy()
        return state

    def _sample_hopf(self, hopf_grid: Dict) -> np.ndarray:
        """Sample N quaternions from Hopf grid weighted by classifier probs."""
        anchors = np.asarray(hopf_grid['anchors'], dtype=float)      # (K,4) [w,x,y,z]
        probs = np.asarray(hopf_grid['probabilities'], dtype=float)
        offsets = np.asarray(hopf_grid.get('offsets', np.zeros((len(anchors), 3))), dtype=float)

        probs = probs / np.sum(probs)
        indices = np.random.choice(len(probs), size=self.n_mc, p=probs)

        q_sel = anchors[indices]                 # (n_mc,4)
        omega_sel = offsets[indices]             # (n_mc,3)

        # Convert to rotation matrices via scipy (scalar-last convention)
        q_scipy = np.roll(q_sel, -1, axis=1)     # (n_mc,4) [x,y,z,w]
        R_anchors = SciRotation.from_quat(q_scipy).as_matrix()
        R_deltas = so3_exp(omega_sel)
        R_samples = np.einsum('nij,njk->nik', R_anchors, R_deltas)

        q_out_scipy = SciRotation.from_matrix(R_samples).as_quat()  # [x,y,z,w]
        q_out = np.roll(q_out_scipy, 1, axis=1)  # [w,x,y,z]
        return q_out

    # ── Core Derivatives ──
    def derivatives(self, state: np.ndarray, t: float, action_flags: Dict) -> np.ndarray:
        n = self.n_mc
        dst = np.zeros_like(state)
        cfg = self.cfg

        # Extract
        r = state[:, self.layout.idx_x:self.layout.idx_x + 3]
        v = state[:, self.layout.idx_vx:self.layout.idx_vx + 3]
        q = state[:, self.layout.idx_qw:self.layout.idx_qw + 4]
        w = state[:, self.layout.idx_wx:self.layout.idx_wx + 3]
        T = state[:, self.layout.idx_thermal:self.layout.idx_thermal + cfg.n_thermal]

        # Normalize quaternions
        q = q / np.linalg.norm(q, axis=1, keepdims=True)

        # ── Attitude kinematics ──
        # dq/dt = 0.5 * Omega(w) @ q
        Omega = np.zeros((n, 4, 4))
        Omega[:, 0, 1:4] = -w
        Omega[:, 1:4, 0] = w
        Omega[:, 1, 2] = w[:, 2]
        Omega[:, 1, 3] = -w[:, 1]
        Omega[:, 2, 1] = -w[:, 2]
        Omega[:, 2, 3] = w[:, 0]
        Omega[:, 3, 1] = w[:, 1]
        Omega[:, 3, 2] = -w[:, 0]
        dq = 0.5 * np.einsum('nij,nj->ni', Omega, q)
        dst[:, self.layout.idx_qw:self.layout.idx_qw + 4] = dq

        # ── Attitude dynamics (Euler) ──
        Jw = np.einsum('ij,nj->ni', cfg.inertia, w)
        wxJw = np.cross(w, Jw)
        F_body, M_body, u_thrusters = self._compute_body_wrench(
            q, w, action_flags, t, state)
        dw = np.einsum('ij,nj->ni', cfg.inertia_inv, M_body - wxJw)
        dst[:, self.layout.idx_wx:self.layout.idx_wx + 3] = dw

        # ── Translational dynamics (CWH with forcing) ──
        # F_body is body-frame; rotate to LVLH via R(q)^T for CWH equations.
        q_inv = q.copy()
        q_inv[:, 1:4] *= -1
        F_lvlh = quat_rotate_vector(q_inv, F_body)

        n_orb = cfg.mean_motion
        m = cfg.mass
        dst[:, self.layout.idx_x] = v[:, 0]
        dst[:, self.layout.idx_y] = v[:, 1]
        dst[:, self.layout.idx_z] = v[:, 2]
        dst[:, self.layout.idx_vx] = 2 * n_orb * v[:, 1] + 3 * n_orb ** 2 * r[:, 0] + F_lvlh[:, 0] / m
        dst[:, self.layout.idx_vy] = -2 * n_orb * v[:, 0] + F_lvlh[:, 1] / m
        dst[:, self.layout.idx_vz] = -n_orb ** 2 * r[:, 2] + F_lvlh[:, 2] / m

        # ── Habitat subsystems ──
        dst[:, self.layout.idx_thermal:self.layout.idx_thermal + cfg.n_thermal] = \
            self._thermal_derivatives(T, action_flags, q)
        dst[:, self.layout.idx_soc] = self._power_derivatives(state[:, self.layout.idx_soc], T, action_flags, t, q)
        dm_prop, dm_o2, dm_co2, dm_h2o = self._life_support_derivatives(
            state[:, self.layout.idx_m_prop], state[:, self.layout.idx_m_o2],
            state[:, self.layout.idx_m_co2], state[:, self.layout.idx_m_h2o],
            action_flags, u_thrusters)
        dst[:, self.layout.idx_m_prop] = dm_prop
        dst[:, self.layout.idx_m_o2] = dm_o2
        dst[:, self.layout.idx_m_co2] = dm_co2
        dst[:, self.layout.idx_m_h2o] = dm_h2o

        self._last_state = state.copy()
        return dst

    def _position_velocity(self, state: Optional[np.ndarray]):
        """Current r, v for guidance — prefer live RK4 state over cached copy."""
        n = self.n_mc
        if state is not None:
            r = state[:, self.layout.idx_x:self.layout.idx_x + 3]
            v = state[:, self.layout.idx_vx:self.layout.idx_vx + 3]
        elif self._last_state is not None:
            r = self._last_state[:, self.layout.idx_x:self.layout.idx_x + 3]
            v = self._last_state[:, self.layout.idx_vx:self.layout.idx_vx + 3]
        else:
            r = np.zeros((n, 3))
            v = np.zeros((n, 3))
        return r, v

    def _compute_body_wrench(self, q, w, action_flags, t,
                             state: Optional[np.ndarray] = None):
        """Guidance + TAM allocation → body-frame force, torque, thruster commands."""
        n = self.n_mc
        cfg = self.cfg
        F_des_lvlh = np.zeros((n, 3))
        r, v = self._position_velocity(state)

        act = action_flags.get('action', 'HOLD')

        if act == 'ABORT':
            # Retro-thrust along -V-bar in LVLH; rotated to body below.
            F_des_lvlh[:, 1] = -sum(th.max_thrust for th in cfg.thrusters)

        elif act == 'HOLD':
            # CWH natural acceleration (Hill's equations):
            #   ẍ = 3n²x + 2nẏ,  ÿ = -2nẋ,  z̈ = -n²z
            n_orb = cfg.mean_motion
            a_cwh = np.zeros((n, 3))
            a_cwh[:, 0] = 3 * n_orb ** 2 * r[:, 0] + 2 * n_orb * v[:, 1]
            a_cwh[:, 1] = -2 * n_orb * v[:, 0]
            a_cwh[:, 2] = -n_orb ** 2 * r[:, 2]
            F_des_lvlh = -cfg.mass * a_cwh - 0.5 * cfg.mass * v

        elif act in ('PROCEED_SLOW', 'PROCEED_NORMAL'):
            v_dock = 0.05 if act == 'PROCEED_SLOW' else 0.20
            n_orb = cfg.mean_motion
            a_cwh = np.zeros((n, 3))
            a_cwh[:, 0] = 3 * n_orb ** 2 * r[:, 0] + 2 * n_orb * v[:, 1]
            a_cwh[:, 1] = -2 * n_orb * v[:, 0]
            a_cwh[:, 2] = -n_orb ** 2 * r[:, 2]
            v_des = np.zeros((n, 3))
            v_des[:, 0] = -v_dock
            F_des_lvlh = cfg.mass * a_cwh - 0.5 * cfg.mass * (v - v_des)

        elif act == 'EMERGENCY_VENT' and t < 1.0:
            F_des_lvlh = np.tile(np.array([0.0, -50.0, 0.0]), (n, 1))

        M_des = self._attitude_controller(q, w, act)
        F_des_body = quat_rotate_vector(q, F_des_lvlh)

        cmd = np.concatenate([F_des_body, M_des], axis=1)
        u = np.einsum('ij,nj->ni', cfg.tam_pinv, cmd)
        u = np.clip(u, 0.0, 1.0)

        FM = np.einsum('ij,nj->ni', cfg.tam, u)
        return FM[:, 0:3], FM[:, 3:6], u

    def _attitude_controller(self, q, w, act):
        """PD attitude controller returning desired body-frame torque."""
        n = self.n_mc
        # Default desired attitude: body axes aligned with LVLH (identity quaternion)
        q_des = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

        if act == 'ABORT':
            # Align retro-axis with -y LVLH. If retro_axis_body = +y_body,
            # we want +y_body to align with -y_lvlh → 180° rotation about z.
            # Simplified: just use identity and let guidance handle direction via force.
            pass

        # Quaternion error: q_err = q_des^{-1} ⊗ q
        q_des_inv = q_des.copy()
        q_des_inv[:, 1:4] *= -1
        # Hamilton product q_des_inv ⊗ q
        w1, x1, y1, z1 = q_des_inv[:, 0], q_des_inv[:, 1], q_des_inv[:, 2], q_des_inv[:, 3]
        w2, x2, y2, z2 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        q_err = np.empty_like(q)
        q_err[:, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        q_err[:, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        q_err[:, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        q_err[:, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        # Use vector part as error signal (proportional to rotation axis)
        Kp, Kd = 50.0, 10.0
        M = -Kp * q_err[:, 1:4] - Kd * w
        return M

    def _thermal_derivatives(self, T, action_flags, q):
        cfg = self.cfg
        n = self.n_mc
        Q_gen = np.zeros((n, cfg.n_thermal))
        for i, node in enumerate(cfg.thermal_nodes):
            q_base = node.q_gen_base * np.ones(n)
            # Solar absorption depends on attitude (simplified constant flux)
            q_solar = node.absorptivity * node.solar_exposure_area * cfg.power.solar_constant * 0.5
            Q_gen[:, i] = q_base + q_solar

        if action_flags.get('action') == 'RECONFIGURE_POWER':
            Q_gen *= 0.65  # 35% load shed

        # Conduction: T @ G^T = T @ G (since G symmetric)
        Q_cond = np.einsum('ij,jk->ik', T, cfg.thermal_g)

        Q_rad = np.zeros((n, cfg.n_thermal))
        for i, node in enumerate(cfg.thermal_nodes):
            Q_rad[:, i] = node.emissivity * SIGMA_SB * node.area * (T_SPACE ** 4 - T[:, i] ** 4)

        C = np.array([node.capacitance for node in cfg.thermal_nodes])
        dT = (Q_gen + Q_cond + Q_rad) / C[None, :]
        return dT

    def _power_derivatives(self, soc, T, action_flags, t, q):
        cfg = self.cfg
        n = self.n_mc

        # Solar power (panel normal in body, sun direction in LVLH)
        s_sun_lvlh = np.tile(np.array([1.0, 0.0, 0.0]), (n, 1))  # simplified sun dir
        s_sun_body = quat_rotate_vector(q, s_sun_lvlh)
        n_panel = cfg.power.panel_normal_body
        cos_theta = np.sum(s_sun_body * n_panel[None, :], axis=1)
        cos_theta = np.maximum(cos_theta, 0.0)
        P_solar = cfg.power.solar_efficiency * cfg.power.solar_panel_area * cfg.power.solar_constant * cos_theta

        P_load = cfg.power.base_load_w * np.ones(n)
        if action_flags.get('action') == 'RECONFIGURE_POWER':
            P_load *= 0.65
        if action_flags.get('action') in ('ABORT', 'PROCEED_SLOW', 'PROCEED_NORMAL', 'EMERGENCY_VENT'):
            P_load += 50.0  # valve/heater power

        E_max = cfg.power.battery_capacity_wh * 3600.0  # J
        dE = P_solar - P_load
        dsoc = dE / E_max

        # Soft-limit at bounds
        soc_clip = np.clip(soc, 0.0, 1.0)
        dsoc = np.where((soc_clip >= 0.999) & (dsoc > 0), 0.0, dsoc)
        dsoc = np.where((soc_clip <= 0.001) & (dsoc < 0), 0.0, dsoc)
        return dsoc

    def _life_support_derivatives(self, m_prop, m_o2, m_co2, m_h2o, action_flags, u_thrusters):
        cfg = self.cfg
        n = self.n_mc

        ls = cfg.life_support
        dm_o2 = (ls.o2_gen_rate - ls.n_crew * ls.o2_consumption_rate) * np.ones(n)
        dm_co2 = (ls.n_crew * ls.o2_consumption_rate - ls.co2_scrubber_rate) * np.ones(n)
        dm_h2o = (ls.h2o_recovery_rate - ls.n_crew * ls.h2o_perspiration_rate) * np.ones(n)

        if not action_flags.get('isolate_module', False):
            dm_o2 -= ls.leak_rate
            dm_h2o -= ls.leak_rate * 0.5

        if action_flags.get('action') == 'EMERGENCY_VENT' and action_flags.get('vent_time', 0.0) < 1.0:
            dm_o2 -= 0.1
            dm_h2o -= 0.05

        # Propellant consumption from thrusters
        mass_flows = np.array([th.max_mass_flow for th in cfg.thrusters])
        dm_prop = -np.sum(u_thrusters * mass_flows[None, :], axis=1)

        # Clamp to prevent negative masses
        dm_o2 = np.where(m_o2 <= 0, np.maximum(dm_o2, 0.0), dm_o2)
        dm_co2 = np.where(m_co2 <= 0, np.maximum(dm_co2, 0.0), dm_co2)
        dm_h2o = np.where(m_h2o <= 0, np.maximum(dm_h2o, 0.0), dm_h2o)
        dm_prop = np.where(m_prop <= 0, np.maximum(dm_prop, 0.0), dm_prop)

        return dm_prop, dm_o2, dm_co2, dm_h2o


def default_spacecraft_config() -> SpacecraftConfig:
    """Example configuration for a crewed proximity-ops vehicle."""
    thrusters = [
        ThrusterConfig(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]), 400.0, 0.18),
        ThrusterConfig(np.array([1.0, -1.0, 0.0]), np.array([0.0, 1.0, 0.0]), 400.0, 0.18),
        ThrusterConfig(np.array([-1.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), 400.0, 0.18),
        ThrusterConfig(np.array([-1.0, -1.0, 0.0]), np.array([0.0, -1.0, 0.0]), 400.0, 0.18),
        ThrusterConfig(np.array([1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 200.0, 0.09),
        ThrusterConfig(np.array([1.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]), 200.0, 0.09),
        ThrusterConfig(np.array([-1.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 200.0, 0.09),
        ThrusterConfig(np.array([-1.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0]), 200.0, 0.09),
    ]
    thermal_nodes = [
        ThermalNodeConfig(5000.0, 0.85, 5.0, 0.3, 200.0, 2.0, 293.15),
        ThermalNodeConfig(20000.0, 0.9, 15.0, 0.2, 50.0, 5.0, 293.15),
        ThermalNodeConfig(80000.0, 0.95, 40.0, 0.15, 0.0, 10.0, 273.15),
    ]
    power = PowerConfig(1000.0, 0.85, 10.0, 0.28, base_load_w=300.0)
    life_support = LifeSupportConfig(
        o2_gen_rate=0.005, o2_consumption_rate=0.001,
        co2_scrubber_rate=0.004, h2o_recovery_rate=0.003,
        h2o_perspiration_rate=0.001, initial_o2=100.0,
        initial_co2=0.0, initial_h2o=50.0, n_crew=3, leak_rate=0.0001
    )
    return SpacecraftConfig(
        mass=10000.0,
        inertia=np.diag([5000.0, 6000.0, 4000.0]),
        thrusters=thrusters,
        thermal_nodes=thermal_nodes,
        power=power,
        life_support=life_support,
        mean_motion=0.001107,  # ~400 km LEO
        keepout_radius=10.0,
        initial_propellant=500.0
    )
