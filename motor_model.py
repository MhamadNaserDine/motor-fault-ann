"""
motor_model.py
==============
Dynamic model of a 3-phase squirrel-cage induction motor, written in the
stationary (alpha-beta, "dq stationary") reference frame and integrated with
scipy's ODE solver (solve_ivp).

State vector  x = [psi_s_alpha, psi_s_beta, psi_r_alpha, psi_r_beta, omega_m, theta_r]
    psi_s : stator flux linkage (Wb)       psi_r : rotor flux linkage (Wb, referred to stator)
    omega_m : mechanical speed (rad/s)     theta_r : electrical rotor angle (rad)

Equations (amplitude-invariant Clarke transform, isolated star point):
    d psi_s / dt = v_s - R_s . i_s
    d psi_r / dt = -R_r(theta_r) . i_r + omega_r . J . psi_r        J = [[0,-1],[1,0]]
    [psi_s; psi_r] = L . [i_s; i_r]                                  (4x4 inductance matrix)
    T_e = 3/2 . p . (psi_r_beta . i_r_alpha - psi_r_alpha . i_r_beta)
    J_m d omega_m / dt = T_e - T_L(t) - B . omega_m ,   d theta_r/dt = p . omega_m

The model is written so that faults enter *physically*:
  * broken rotor bars   -> rotor resistance asymmetry R_r(theta_r)  (rotor d/q axes differ)
  * inter-turn short    -> phase-a stator winding loses a fraction mu of its turns
                           (reduced-turns model: R, leakage and magnetising coupling of phase a change)
  * bearing fault       -> load-torque oscillation at the bearing characteristic frequency
  * supply unbalance    -> unequal phase-voltage amplitudes (phase loss = amplitude 0)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

SQ3 = np.sqrt(3.0)
# Clarke "C" matrix: i_abc = C @ i_ab (no zero sequence because the star point is isolated)
C = np.array([[1.0, 0.0], [-0.5, SQ3 / 2], [-0.5, -SQ3 / 2]])
CT23 = (2.0 / 3.0) * C.T  # abc -> alpha-beta (amplitude invariant)


@dataclass
class MotorParams:
    """2.2 kW, 400 V (Y), 50 Hz, 4-pole squirrel-cage motor (typical catalogue values)."""
    P_rated: float = 2.2e3      # W
    V_ll: float = 400.0         # V rms line-to-line
    f_s: float = 50.0           # Hz supply frequency
    p: int = 2                  # pole PAIRS (4-pole machine)
    Rs: float = 2.9             # ohm stator resistance
    Rr: float = 2.2             # ohm rotor resistance (referred to stator)
    Lls: float = 0.0125         # H stator leakage inductance
    Llr: float = 0.0125         # H rotor leakage inductance
    Lm: float = 0.36            # H magnetising inductance
    J: float = 0.012            # kg m^2 rotor + load inertia
    B: float = 0.001            # N m s/rad viscous friction
    n_bars: int = 28            # number of rotor bars
    # bearing 6205-like geometry (used for the outer-race characteristic frequency)
    n_balls: int = 9
    ball_d: float = 7.94e-3     # m
    pitch_D: float = 39.04e-3   # m

    @property
    def V_ph(self) -> float:
        return self.V_ll / SQ3

    @property
    def omega_s(self) -> float:
        return 2 * np.pi * self.f_s

    @property
    def n_sync_rpm(self) -> float:
        return 60 * self.f_s / self.p

    @property
    def T_rated(self) -> float:
        # rated torque at ~ rated speed 1430 rpm
        return self.P_rated / (1430 * 2 * np.pi / 60)

    @property
    def bpfo_ratio(self) -> float:
        """Outer-race defect frequency / shaft frequency: (Nb/2)(1 - d/D cos(beta)), beta = 0."""
        return self.n_balls / 2 * (1 - self.ball_d / self.pitch_D)


@dataclass
class FaultState:
    """Physical fault parameters applied to the model (all 0 = perfectly healthy machine)."""
    rotor_asym: float = 0.0     # k: R_rq = R_r (1 + k)  (broken bars)
    turns_short: float = 0.0    # mu: fraction of phase-a turns short-circuited
    bearing_amp: float = 0.0    # torque oscillation amplitude (fraction of rated torque)
    v_unbalance: tuple = (1.0, 1.0, 1.0)  # per-phase voltage amplitude factors
    stator_asym: float = 0.0    # small natural asymmetry of phase-a resistance (healthy spread)


# --------------------------------------------------------------------------- matrices
def _stator_matrices(mp: MotorParams, fs: FaultState):
    """Return (Rs_ab, Lss_ab, N) 2x2 matrices for the possibly-faulty stator.

    Reduced-turns model: phase a keeps (1-mu) of its turns.
       self/mutual magnetising terms scale with n_i n_j, leakage with n_i^2,
       resistance with n_i (shorter conductor).
    """
    n = np.array([1.0 - fs.turns_short, 1.0, 1.0])
    rfac = np.array([(1.0 - fs.turns_short) * (1.0 + fs.stator_asym), 1.0, 1.0])
    Lms = mp.Lm / 1.5
    M_abc = Lms * np.array([[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]])
    Lss_abc = np.outer(n, n) * M_abc + mp.Lls * np.diag(n ** 2)
    Lss_ab = CT23 @ Lss_abc @ C
    Rs_ab = CT23 @ (mp.Rs * np.diag(rfac)) @ C
    N = CT23 @ np.diag(n) @ C          # stator-rotor coupling factor (= I when healthy)
    N = 0.5 * (N + N.T)
    return Rs_ab, Lss_ab, N


def inductance_matrix(mp: MotorParams, fs: FaultState):
    Rs_ab, Lss_ab, N = _stator_matrices(mp, fs)
    Lr = mp.Lm + mp.Llr
    L4 = np.zeros((4, 4))
    L4[:2, :2] = Lss_ab
    L4[:2, 2:] = mp.Lm * N
    L4[2:, :2] = mp.Lm * N
    L4[2:, 2:] = Lr * np.eye(2)
    return Rs_ab, L4


# --------------------------------------------------------------------------- steady state
def steady_state(mp: MotorParams, T_load: float):
    """Healthy-machine steady state from the space-vector equations (used as initial condition).
    Returns slip s and complex stator / rotor current & flux phasors."""
    w = mp.omega_s
    Vs = np.sqrt(2) * mp.V_ph
    Ls, Lr = mp.Lls + mp.Lm, mp.Llr + mp.Lm

    def solve(s):
        A = np.array([[mp.Rs + 1j * w * Ls, 1j * w * mp.Lm],
                      [1j * s * w * mp.Lm, mp.Rr + 1j * s * w * Lr]])
        Is, Ir = np.linalg.solve(A, np.array([Vs, 0.0]))
        return Is, Ir

    def torque(s):
        Is, Ir = solve(s)
        psir = mp.Lm * Is + Lr * Ir
        return 1.5 * mp.p * (psir.imag * Ir.real - psir.real * Ir.imag)

    def residual(s):
        wm = w * (1 - s) / mp.p
        return torque(s) - T_load - mp.B * wm

    s = brentq(residual, 1e-6, 0.3)
    Is, Ir = solve(s)
    return s, Is, Ir, mp.Lls * Is + mp.Lm * (Is + Ir), mp.Llr * Ir + mp.Lm * (Is + Ir)


# --------------------------------------------------------------------------- simulation
def simulate(mp: MotorParams | None = None, fs: FaultState | None = None, load: float = 1.0,
             t_end: float = 3.0, t_settle: float = 0.4, f_sample: float = 4000.0,
             start_from_standstill: bool = False, rtol: float = 1e-6, atol: float = 1e-7):
    """Simulate the motor and return a dict of sampled signals.

    load : load torque as a fraction of rated torque
    t_end : length of the recorded window (s)       t_settle : discarded transient (s)
    start_from_standstill : True -> direct-on-line start (shows start-up transient, slower)
    """
    mp = mp or MotorParams()
    fs = fs or FaultState()
    T_L0 = load * mp.T_rated
    Rs_ab, L4 = inductance_matrix(mp, fs)
    L4inv = np.linalg.inv(L4)
    w = mp.omega_s
    Vpk = np.sqrt(2) * mp.V_ph
    kv = np.asarray(fs.v_unbalance, float)
    ph = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])
    Rr0 = mp.Rr
    k = fs.rotor_asym
    Rr_avg = Rr0 * (1 + k / 2)
    Rr_dif = -Rr0 * k / 2         # (R_rd - R_rq)/2
    Kb = mp.bpfo_ratio
    Ab = fs.bearing_amp * mp.T_rated
    p = mp.p

    def v_ab(t):
        v_abc = Vpk * kv * np.cos(w * t + ph)
        return CT23 @ v_abc

    def load_torque(t, theta_r):
        # bearing defect: torque pulse every time a ball passes the defect -> cos(Kb * theta_m)
        return T_L0 + Ab * np.cos(Kb * theta_r / p)

    def rhs(t, x):
        psi = x[:4]
        wm, th = x[4], x[5]
        i = L4inv @ psi
        isab, irab = i[:2], i[2:]
        wr = p * wm
        vs = v_ab(t)
        dpsis = vs - Rs_ab @ isab
        c2, s2 = np.cos(2 * th), np.sin(2 * th)
        Rr_m = np.array([[Rr_avg + Rr_dif * c2, Rr_dif * s2],
                         [Rr_dif * s2, Rr_avg - Rr_dif * c2]])
        dpsir = -Rr_m @ irab + wr * np.array([-psi[3], psi[2]])
        Te = 1.5 * p * (psi[3] * irab[0] - psi[2] * irab[1])
        dwm = (Te - load_torque(t, th) - mp.B * wm) / mp.J
        return np.concatenate([dpsis, dpsir, [dwm, wr]])

    if start_from_standstill:
        x0 = np.zeros(6)
        t_settle = 0.0
    else:
        s, Is, Ir, psis, psir = steady_state(mp, T_L0)
        x0 = np.array([psis.real, psis.imag, psir.real, psir.imag, w * (1 - s) / p, 0.0])

    t0, t1 = 0.0, t_settle + t_end
    n = int(round(t_end * f_sample))
    t_eval = t_settle + np.arange(n) / f_sample
    sol = solve_ivp(rhs, (t0, t1), x0, method="RK45", t_eval=t_eval, rtol=rtol, atol=atol,
                    max_step=1.0 / (20 * mp.f_s))
    X = sol.y
    I = L4inv @ X[:4]
    i_ab = I[:2]
    i_abc = C @ i_ab
    v_abc = (Vpk * kv[:, None]) * np.cos(w * sol.t[None, :] + ph[:, None])
    psi_r = X[2:4]
    Te = 1.5 * p * (psi_r[1] * I[2] - psi_r[0] * I[3])
    wm = X[4]
    return {
        "t": sol.t - t_settle,
        "i_abc": i_abc,                      # 3 x n stator line currents (A)
        "v_abc": v_abc,                      # 3 x n phase voltages (V)
        "speed_rpm": wm * 60 / (2 * np.pi),
        "torque": Te,
        "load_torque": load_torque(sol.t, X[5]),
        "f_sample": f_sample,
        "f_s": mp.f_s,
        "slip": 1 - (wm.mean() * p) / w,
        "params": asdict(mp),
    }


if __name__ == "__main__":
    import time
    mp = MotorParams()
    s, Is, *_ = steady_state(mp, mp.T_rated)
    print(f"T_rated={mp.T_rated:.2f} N.m, rated slip={s:.4f}, speed={mp.n_sync_rpm*(1-s):.0f} rpm, "
          f"I_rms={abs(Is)/np.sqrt(2):.2f} A")
    t = time.time()
    out = simulate(mp, FaultState(), load=1.0)
    print(f"sim took {time.time()-t:.2f}s; mean speed {out['speed_rpm'].mean():.1f} rpm; "
          f"Irms_a {np.sqrt(np.mean(out['i_abc'][0]**2)):.2f} A; Te mean {out['torque'].mean():.2f}")
