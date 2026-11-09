"""
features.py
===========
Motor Current Signature Analysis (MCSA) feature extraction.

Input : one simulation record (3 stator currents, 3 phase voltages, speed) + measurement noise
Output: a fixed-length feature vector (FEATURE_NAMES)

Feature groups
--------------
time domain      : RMS, peak, crest factor, kurtosis, skewness of the phase current
symmetry         : current unbalance, negative-sequence current I2/I1, negative-sequence voltage V2/V1
Park's vector    : (iD, iQ) pattern -> eccentricity of the ellipse, ripple of the modulus |i|
spectrum (FFT)   : amplitude (dB rel. fundamental) of the broken-bar sidebands (1 +/- 2s) f_s,
                   of the bearing components |f_s +/- f_o|, 3rd harmonic, THD, low/high band energy
operating point  : slip (from the speed sensor) -- tells the network the load level
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew
from motor_model import MotorParams

SQ3 = np.sqrt(3.0)
A_OP = np.exp(2j * np.pi / 3)   # Fortescue operator a

FEATURE_NAMES = [
    "I_rms", "I_peak", "crest_factor", "kurtosis", "skewness",
    "I_unbalance_%", "I2_I1_negseq", "V2_V1_negseq",
    "park_eccentricity", "park_mod_ripple",
    "brb_lsb_dB", "brb_usb_dB", "brg_lsb_dB", "brg_usb_dB",
    "h3_dB", "THD_%", "band_low_dB", "band_high_dB", "slip",
]


def add_noise(sig: dict, noise_level: float, rng: np.random.Generator) -> dict:
    """Additive white Gaussian measurement noise, std = noise_level x RMS of each signal,
    plus 0.5 % sensor gain error per phase and +/- 2 rpm speed-sensor error."""
    out = dict(sig)
    i = sig["i_abc"]
    v = sig["v_abc"]
    irms = np.sqrt(np.mean(i ** 2, axis=1, keepdims=True))
    vrms = np.sqrt(np.mean(v ** 2, axis=1, keepdims=True))
    gain_i = 1 + rng.normal(0, 0.005, (3, 1))
    gain_v = 1 + rng.normal(0, 0.005, (3, 1))
    out["i_abc"] = gain_i * i + rng.normal(0, 1, i.shape) * noise_level * irms
    out["v_abc"] = gain_v * v + rng.normal(0, 1, v.shape) * noise_level * vrms
    out["speed_rpm"] = sig["speed_rpm"] + rng.normal(0, 2.0)
    return out


def spectrum(x: np.ndarray, f_sample: float, n_fft: int | None = None):
    """Single-sided amplitude spectrum with a Hann window (amplitude-corrected), zero-padded."""
    n = len(x)
    n_fft = n_fft or 4 * n
    w = np.hanning(n)
    X = np.fft.rfft((x - x.mean()) * w, n_fft)
    amp = 2 * np.abs(X) / w.sum()
    f = np.fft.rfftfreq(n_fft, 1 / f_sample)
    return f, amp


def _peak(f, amp, f0, half_width=0.35):
    m = (f > f0 - half_width) & (f < f0 + half_width)
    return amp[m].max() if m.any() else 1e-12


def _band(f, amp, lo, hi):
    m = (f >= lo) & (f <= hi)
    return np.sqrt(np.sum(amp[m] ** 2)) if m.any() else 1e-12


def phasors(x3: np.ndarray, f_sample: float, f0: float):
    """Complex fundamental phasors of 3 signals by single-bin DFT (window = integer number of periods)."""
    n = len(x3[0])
    per = int(round(f_sample / f0))
    n_use = (n // per) * per
    t = np.arange(n_use) / f_sample
    e = np.exp(-2j * np.pi * f0 * t)
    return 2 * (x3[:, :n_use] @ e) / n_use


def sequences(ph3):
    """Fortescue: positive and negative sequence magnitudes."""
    a, b, c = ph3
    pos = (a + A_OP * b + A_OP ** 2 * c) / 3
    neg = (a + A_OP ** 2 * b + A_OP * c) / 3
    return abs(pos), abs(neg)


def extract_features(sig: dict, mp: MotorParams | None = None) -> np.ndarray:
    mp = mp or MotorParams()
    fsamp = sig["f_sample"]
    f_s = sig["f_s"]
    i3 = sig["i_abc"]
    v3 = sig["v_abc"]
    ia = i3[0]

    # --- time domain -------------------------------------------------------
    rms3 = np.sqrt(np.mean(i3 ** 2, axis=1))
    I_rms = rms3.mean()
    I_peak = np.abs(ia).max()
    crest = I_peak / rms3[0]
    kur = kurtosis(ia, fisher=False)
    sk = skew(ia)
    unb = 100 * np.max(np.abs(rms3 - I_rms)) / I_rms           # NEMA definition

    # --- symmetrical components -------------------------------------------
    I1, I2 = sequences(phasors(i3, fsamp, f_s))
    V1, V2 = sequences(phasors(v3, fsamp, f_s))
    i_neg = I2 / I1
    v_neg = V2 / V1

    # --- Park's vector ------------------------------------------------------
    iD = np.sqrt(2 / 3) * ia - (1 / np.sqrt(6)) * (i3[1] + i3[2])
    iQ = (1 / np.sqrt(2)) * (i3[1] - i3[2])
    cov = np.cov(np.vstack([iD, iQ]))
    ev = np.sort(np.linalg.eigvalsh(cov))
    ecc = np.sqrt(1 - ev[0] / ev[1])                 # 0 = perfect circle (healthy)
    mod = np.hypot(iD, iQ)
    mod_ripple = mod.std() / mod.mean()

    # --- spectrum -----------------------------------------------------------
    slip = 1 - (sig["speed_rpm"].mean() / 60 * mp.p) / f_s
    slip = float(np.clip(slip, 1e-3, 0.2))
    f, A = spectrum(ia, fsamp)
    A1 = _peak(f, A, f_s, 0.5)
    db = lambda x: 20 * np.log10(max(x, 1e-12) / A1)
    brb_l = db(_peak(f, A, (1 - 2 * slip) * f_s))
    brb_u = db(_peak(f, A, (1 + 2 * slip) * f_s))
    f_r = f_s * (1 - slip) / mp.p
    f_o = mp.bpfo_ratio * f_r
    brg_l = db(_peak(f, A, abs(f_s - f_o), 0.6))
    brg_u = db(_peak(f, A, f_s + f_o, 0.6))
    h3 = db(_peak(f, A, 3 * f_s))
    harm = np.sqrt(sum(_peak(f, A, k * f_s) ** 2 for k in range(2, 10)))
    thd = 100 * harm / A1
    band_low = db(_band(f, A, 2, f_s - 3))                    # sub-fundamental band
    band_high = db(_band(f, A, f_s + 3, 2 * f_s + 50))        # just above fundamental

    return np.array([I_rms, I_peak, crest, kur, sk, unb, i_neg, v_neg, ecc, mod_ripple,
                     brb_l, brb_u, brg_l, brg_u, h3, thd, band_low, band_high, slip], dtype=float)


class StandardScaler:
    """z-score normalisation  x' = (x - mean) / std  (fitted on the TRAINING set only)."""

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)
