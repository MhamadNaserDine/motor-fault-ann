"""
faults.py
=========
Fault definitions, severity levels and the mapping "fault class + severity -> physical
model parameters" (FaultState) used by motor_model.simulate().

Classes
-------
0 Healthy
1 Broken rotor bars (BRB)           -> sidebands at (1 +/- 2ks) f_s in the stator current
2 Stator inter-turn short (ITSC)    -> current unbalance, negative-sequence current
3 Bearing fault (outer race)        -> torque oscillation at f_o = BPFO, current components at |f_s +/- k f_o|
4 Supply voltage unbalance          -> unbalanced voltages *and* currents (negative-sequence voltage)
"""
from __future__ import annotations

import numpy as np
from motor_model import MotorParams, FaultState

CLASSES = ["Healthy", "Broken rotor bars", "Stator inter-turn short",
           "Bearing fault", "Supply voltage unbalance"]
SHORT = ["HLT", "BRB", "ITSC", "BRG", "VUNB"]
ICONS = ["✅", "⚠️", "⚠️", "⚠️", "⚠️"]

# severity levels (1 = incipient, 2 = medium, 3 = severe) and their physical meaning
SEVERITY_LABELS = {
    0: {0: "—"},
    1: {1: "1 broken bar", 2: "2 broken bars", 3: "3 broken bars"},
    2: {1: "1.5 % turns shorted", 2: "4 % turns shorted", 3: "8 % turns shorted"},
    3: {1: "incipient defect", 2: "medium defect", 3: "severe defect"},
    4: {1: "2 % unbalance", 2: "4 % unbalance", 3: "7 % unbalance"},
}


def broken_bar_asymmetry(n_broken: int, n_bars: int) -> float:
    """Equivalent rotor-resistance increase for n adjacent broken bars:
    dR/R = 3n / (N - 3n)   (classic approximation, e.g. Bellini et al.)."""
    return 3 * n_broken / (n_bars - 3 * n_broken)


def make_fault_state(fault: int, severity: int, mp: MotorParams | None = None,
                     rng: np.random.Generator | None = None, natural_spread: bool = True) -> FaultState:
    """Build the physical FaultState for a given class/severity.

    natural_spread=True adds the small manufacturing / supply imperfections every real
    motor has (slight rotor & stator asymmetry, 0-1 % supply unbalance), so that the
    healthy class is not a perfect mathematical machine and the problem is realistic.
    """
    mp = mp or MotorParams()
    rng = rng or np.random.default_rng()
    fs = FaultState()
    if natural_spread:
        fs.rotor_asym = abs(rng.normal(0, 0.008))
        fs.stator_asym = rng.normal(0, 0.01)
        u = rng.uniform(0, 0.01)        # real grids: 0-1 % voltage unbalance
        fs.v_unbalance = tuple(1 - u * rng.dirichlet([1, 1, 1]) * 3)

    jitter = lambda: rng.uniform(0.85, 1.15)  # severity is never exactly the nominal value
    if fault == 1:
        fs.rotor_asym += broken_bar_asymmetry(severity, mp.n_bars) * jitter()
    elif fault == 2:
        fs.turns_short = {1: 0.015, 2: 0.04, 3: 0.08}[severity] * jitter()
    elif fault == 3:
        fs.bearing_amp = {1: 0.02, 2: 0.05, 3: 0.10}[severity] * jitter()
    elif fault == 4:
        u = {1: 0.02, 2: 0.04, 3: 0.07}[severity] * jitter()
        ph = rng.integers(3)                 # the weak phase is random
        kv = np.ones(3)
        kv[ph] -= u
        fs.v_unbalance = tuple(kv)
    return fs


def fault_frequencies(fault: int, f_s: float, slip: float, mp: MotorParams | None = None, kmax: int = 2):
    """Theoretical characteristic frequencies (Hz) of each fault in the stator-current spectrum."""
    mp = mp or MotorParams()
    f_r = f_s * (1 - slip) / mp.p          # shaft rotation frequency
    if fault == 1:
        return sorted({abs((1 + 2 * k * s_) * f_s) for k in range(1, kmax + 1) for s_ in (slip, -slip)})
    if fault == 2:
        # shorted turns: unbalance (negative sequence at f_s) + slot/rotor related components f_s(1 +/- k(1-s)/p)
        return sorted({abs(f_s * (1 + k * (1 - slip) / mp.p)) for k in (-1, 1)} | {3 * f_s})
    if fault == 3:
        f_o = mp.bpfo_ratio * f_r
        return sorted({abs(f_s + m * f_o) for m in (-1, 1)})
    if fault == 4:
        return [f_s, 3 * f_s]
    return []
