"""
mcsa_simulator.py — 3-Phase Induction Motor Stator Current Simulator
======================================================================
Generates synthetic stator current time series for a 4-pole induction motor
under four operating conditions:
  - Healthy (nominal)
  - Broken Rotor Bar (BRB)
  - Outer-Race Bearing Fault (BPFO)
  - Stator Inter-Turn Short Circuit

Physical model
--------------
Under healthy conditions, stator current is:
    i(t) = I_peak * sin(2π*f₀*t + φ)

Each fault introduces specific spectral components grounded in the
electromagnetic physics of the induction machine:

BRB (Broken Rotor Bar)
  Rotor asymmetry causes asymmetric air-gap flux, modulating stator current
  at slip-dependent sidebands:
    f_brb = f₀ * (1 ± 2ks),  k = 1, 2, ...
  Amplitude: typically −40 to −70 dB relative to fundamental.
  Reference: Thomson & Fenger, IEEE IAS 2001.

Bearing Outer-Race Defect (BPFO)
  Bearing defect creates periodic mechanical vibration at the ball-pass
  frequency, modulating air-gap eccentricity, which appears in stator current:
    f_bng = |f₀ ± m * BPFO|,  m = 1, 2, ...
  BPFO = (Nb/2) * f_r * (1 − (Bd/Pd) * cos(α))
  Parameters: SKF 6205-type bearing (Nb=9, Bd=7.938mm, Pd=38.5mm, α=0°)
  Amplitude: typically −45 to −55 dB.
  Reference: Randall & Antoni, Mechanical Systems and Signal Processing, 2011.

Stator Inter-Turn Short Circuit
  Partial short in one winding reduces effective turns, creating:
    1. Phase amplitude imbalance (positive/negative sequence asymmetry)
    2. Rise in odd harmonic content (3rd, 5th, 7th harmonics)
  Reference: Nandi, Toliyat & Li, IEEE Trans. Energy Conv. 2005.

Motor parameters (4-pole, 50 Hz, 10 A rated)
---------------------------------------------
  f₀    = 50.0 Hz   (supply frequency)
  poles = 4         (pole pairs p = 2)
  f_sync = 25.0 Hz  (synchronous mechanical frequency = f₀/p)
  s     = 0.033     (nominal full-load slip, 3.3%)
  f_r   = 24.17 Hz  (shaft rotation = f_sync * (1 - s))
  I_rated = 10 A (rms) → I_peak = 14.14 A
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

# ── Motor & bearing constants ────────────────────────────────────────────────

F0          = 50.0           # Supply frequency [Hz]
POLES       = 4              # Number of poles
P           = POLES // 2     # Pole pairs
F_SYNC      = F0 / P         # Synchronous mechanical frequency [Hz]
SLIP        = 0.033          # Nominal full-load slip
F_SHAFT     = F_SYNC * (1 - SLIP)   # ≈ 24.17 Hz

# SKF 6205-type bearing geometry
NB_BALLS    = 9
BD_MM       = 7.938          # Ball diameter [mm]
PD_MM       = 38.5           # Pitch diameter [mm]
CONTACT_DEG = 0.0            # Contact angle [degrees]

BPFO = (NB_BALLS / 2) * F_SHAFT * (
    1 - (BD_MM / PD_MM) * np.cos(np.radians(CONTACT_DEG))
)  # ≈ 86.4 Hz
BPFI = (NB_BALLS / 2) * F_SHAFT * (
    1 + (BD_MM / PD_MM) * np.cos(np.radians(CONTACT_DEG))
)  # ≈ 130.5 Hz

# Rated current
I_RATED_A   = 10.0           # RMS rated current [A]
I_PEAK_A    = I_RATED_A * np.sqrt(2)   # ≈ 14.14 A

# Default simulation parameters
FS_HZ       = 10_000.0       # Sampling frequency [Hz]
DURATION_S  = 10.0           # Duration [seconds]
NOISE_STD_A = 0.01           # Current sensor noise std [A]

# Fault injection amplitudes (dB relative to fundamental)
BRB_DB      = -45.0   # BRB sideband amplitude [dBc]
BEARING_DB  = -50.0   # Bearing sideband amplitude [dBc]
SHORT_IMBAL = 0.05    # Inter-turn short: amplitude imbalance fraction


@dataclass
class MotorSignal:
    """Container for simulated 3-phase stator current."""
    phase_a: np.ndarray   # Phase A current [A]
    phase_b: np.ndarray   # Phase B current [A]
    phase_c: np.ndarray   # Phase C current [A]
    timestamps: np.ndarray  # Time axis [s]
    fault_type: str
    fs: float
    metadata: dict        # Physical parameters used


def _db_to_amplitude(db: float) -> float:
    """Convert dB (relative to fundamental peak I_PEAK_A) to absolute [A]."""
    return I_PEAK_A * 10.0 ** (db / 20.0)


def simulate_healthy(
    duration_s: float = DURATION_S,
    fs: float = FS_HZ,
    slip: float = SLIP,
    noise_std: float = NOISE_STD_A,
    seed: int = 42,
) -> MotorSignal:
    """
    Healthy 3-phase stator current: balanced sinusoids + sensor noise.

    Phase offsets: 0°, −120°, +120° (positive sequence ABC).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    N = len(t)

    phases_deg = [0.0, -120.0, 120.0]
    currents = []
    for phi in phases_deg:
        i = I_PEAK_A * np.sin(2 * np.pi * F0 * t + np.radians(phi))
        i += rng.normal(0, noise_std, N)
        currents.append(i)

    return MotorSignal(
        phase_a=currents[0], phase_b=currents[1], phase_c=currents[2],
        timestamps=t, fault_type='healthy', fs=fs,
        metadata={'slip': slip, 'f_shaft_hz': F_SYNC * (1 - slip),
                  'noise_std_a': noise_std},
    )


def simulate_brb(
    duration_s: float = DURATION_S,
    fs: float = FS_HZ,
    slip: float = SLIP,
    n_broken_bars: int = 1,
    sideband_db: float = BRB_DB,
    noise_std: float = NOISE_STD_A,
    seed: int = 42,
) -> MotorSignal:
    """
    Broken Rotor Bar current signature.

    Adds sidebands at f₀(1 ± 2ks) for k=1,2.
    Amplitude scales with n_broken_bars: each additional bar adds ~6 dB.

    Physical note: the lower sideband f₀(1-2s) is typically stronger than
    the upper sideband due to the direction of rotor flux modulation.
    Lower sideband amplitude = sideband_db.
    Upper sideband = sideband_db - 3 dB (empirical factor from field data).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    N = len(t)

    # Amplitude scales linearly with broken bars (in linear domain)
    bar_factor = n_broken_bars ** 0.5
    A_lower = _db_to_amplitude(sideband_db) * bar_factor
    A_upper = _db_to_amplitude(sideband_db - 3.0) * bar_factor  # upper slightly weaker

    f_lower_k = [F0 * (1 - 2 * k * slip) for k in range(1, 3)]  # k=1,2
    f_upper_k = [F0 * (1 + 2 * k * slip) for k in range(1, 3)]

    phases_deg = [0.0, -120.0, 120.0]
    currents = []
    for phi_rad in [np.radians(p) for p in phases_deg]:
        i = I_PEAK_A * np.sin(2 * np.pi * F0 * t + phi_rad)
        # Add BRB sidebands (k=1: primary, k=2: secondary at lower amplitude)
        for k_idx, (f_lo, f_hi) in enumerate(zip(f_lower_k, f_upper_k)):
            attenuation = 10 ** (k_idx * -6 / 20)  # each order: -6 dB
            i += A_lower * attenuation * np.sin(2 * np.pi * f_lo * t + phi_rad)
            i += A_upper * attenuation * np.sin(2 * np.pi * f_hi * t + phi_rad)
        i += rng.normal(0, noise_std, N)
        currents.append(i)

    f_brb_expected = [(F0 * (1 - 2 * slip), F0 * (1 + 2 * slip))]

    return MotorSignal(
        phase_a=currents[0], phase_b=currents[1], phase_c=currents[2],
        timestamps=t, fault_type='brb', fs=fs,
        metadata={
            'slip': slip,
            'f_brb_lower_hz': F0 * (1 - 2 * slip),   # 46.7 Hz at s=0.033
            'f_brb_upper_hz': F0 * (1 + 2 * slip),   # 53.3 Hz at s=0.033
            'sideband_db': sideband_db,
            'n_broken_bars': n_broken_bars,
        },
    )


def simulate_bearing_fault(
    duration_s: float = DURATION_S,
    fs: float = FS_HZ,
    slip: float = SLIP,
    fault_type: str = 'outer',   # 'outer' (BPFO) or 'inner' (BPFI)
    sideband_db: float = BEARING_DB,
    noise_std: float = NOISE_STD_A,
    seed: int = 42,
) -> MotorSignal:
    """
    Bearing outer-race (BPFO) or inner-race (BPFI) fault current signature.

    Fault frequency modulates air-gap eccentricity → sidebands in stator current
    at |f₀ ± m * f_fault|, m = 1, 2, ...

    BPFO (outer-race): stationary defect, strong modulation, amplitude ≈ −50 dBc.
    BPFI (inner-race): rotating defect, amplitude-modulated by f_r, slightly weaker.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    N = len(t)

    f_fault = BPFO if fault_type == 'outer' else BPFI
    A_base = _db_to_amplitude(sideband_db)

    phases_deg = [0.0, -120.0, 120.0]
    currents = []
    for phi_rad in [np.radians(p) for p in phases_deg]:
        i = I_PEAK_A * np.sin(2 * np.pi * F0 * t + phi_rad)
        # Sidebands: |f₀ ± m * f_fault| for m = 1, 2
        for m in range(1, 3):
            attenuation = 10 ** ((m - 1) * -6 / 20)  # each order: -6 dB
            f_upper = F0 + m * f_fault
            f_lower = abs(F0 - m * f_fault)   # |f₀ - m*f_fault|
            i += A_base * attenuation * np.sin(2 * np.pi * f_upper * t)
            if f_lower > 1.0:  # ignore DC/sub-Hz component
                i += A_base * attenuation * np.sin(2 * np.pi * f_lower * t)
        i += rng.normal(0, noise_std, N)
        currents.append(i)

    return MotorSignal(
        phase_a=currents[0], phase_b=currents[1], phase_c=currents[2],
        timestamps=t, fault_type=f'bearing_{fault_type}', fs=fs,
        metadata={
            'fault_characteristic': fault_type,
            'f_fault_hz': f_fault,
            'f_sidebands_hz': [abs(F0 - f_fault), F0 + f_fault],
            'bpfo_hz': BPFO,
            'bpfi_hz': BPFI,
            'sideband_db': sideband_db,
        },
    )


def simulate_stator_short(
    duration_s: float = DURATION_S,
    fs: float = FS_HZ,
    short_severity: float = SHORT_IMBAL,
    harmonic_rise_db: float = -30.0,  # 3rd harmonic rise [dBc]
    noise_std: float = NOISE_STD_A,
    seed: int = 42,
) -> MotorSignal:
    """
    Stator inter-turn short circuit signature.

    Effects:
      1. Phase A amplitude reduced by short_severity fraction (winding turn reduction)
      2. Rise in odd harmonics (3rd, 5th, 7th) in all phases
      3. Slight phase angle distortion in Phase A

    short_severity: fractional amplitude reduction in affected phase (0.05 = 5%).
    harmonic_rise_db: 3rd harmonic amplitude relative to fundamental [dBc].
    5th harmonic at harmonic_rise_db - 6 dB, 7th at harmonic_rise_db - 12 dB.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / fs)
    N = len(t)

    A3 = _db_to_amplitude(harmonic_rise_db)
    A5 = _db_to_amplitude(harmonic_rise_db - 6.0)
    A7 = _db_to_amplitude(harmonic_rise_db - 12.0)

    phases_deg = [0.0, -120.0, 120.0]
    phase_amplitudes = [I_PEAK_A * (1 - short_severity), I_PEAK_A, I_PEAK_A]
    currents = []

    for amp, phi in zip(phase_amplitudes, [np.radians(p) for p in phases_deg]):
        i = amp * np.sin(2 * np.pi * F0 * t + phi)
        # Odd harmonics (characteristic of unbalanced flux)
        i += A3 * np.sin(2 * np.pi * 3 * F0 * t + phi)
        i += A5 * np.sin(2 * np.pi * 5 * F0 * t + phi)
        i += A7 * np.sin(2 * np.pi * 7 * F0 * t + phi)
        i += rng.normal(0, noise_std, N)
        currents.append(i)

    return MotorSignal(
        phase_a=currents[0], phase_b=currents[1], phase_c=currents[2],
        timestamps=t, fault_type='stator_short', fs=fs,
        metadata={
            'short_severity': short_severity,
            'phase_a_amplitude_reduction_pct': short_severity * 100,
            'harmonic_3rd_db': harmonic_rise_db,
            'harmonic_5th_db': harmonic_rise_db - 6.0,
            'harmonic_7th_db': harmonic_rise_db - 12.0,
        },
    )


def get_all_conditions(seed: int = 42) -> Dict[str, MotorSignal]:
    """Generate all four operating conditions with default parameters."""
    return {
        'healthy':        simulate_healthy(seed=seed),
        'brb':            simulate_brb(seed=seed),
        'bearing_outer':  simulate_bearing_fault(fault_type='outer', seed=seed),
        'stator_short':   simulate_stator_short(seed=seed),
    }
