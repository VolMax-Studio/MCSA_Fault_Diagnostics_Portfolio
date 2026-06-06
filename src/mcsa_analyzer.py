"""
mcsa_analyzer.py — MCSA Diagnostics: PSD, Sideband Detection, Unbalance
=========================================================================
Core signal processing for Motor Current Signature Analysis.

Pipeline per condition:
  Phase A current
    → IIR Notch filter (suppress 50 Hz fundamental, Q=50)
    → Welch PSD (Kaiser window, β=14, 2.5-sec segments)
    → Sideband tracker (slip-based search windows)
    → Fault indicator extraction

Three diagnostic modules:
  1. mcsa_psd()          — High-resolution Welch PSD with Kaiser window
  2. notch_filter()      — IIR notch at f₀, configurable Q
  3. find_sideband_peak()— Peak search within ±δf Hz of target frequency
  4. compute_unbalance() — Negative-sequence current ratio (Fortescue)
  5. harmonic_content()  — Individual harmonic amplitudes + THD
"""

import numpy as np
from scipy.signal import iirnotch, sosfilt, sosfiltfilt, welch, find_peaks
from scipy.signal import butter, sosfiltfilt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .mcsa_simulator import F0, FS_HZ


@dataclass
class DiagnosticResult:
    """All diagnostic indicators for one operating condition."""
    fault_type: str
    freqs_hz: np.ndarray          # PSD frequency axis [Hz]
    psd_db: np.ndarray            # PSD [dBc relative to fundamental peak]
    fundamental_db: float         # Fundamental peak power [dB ref 1 A²/Hz]
    brb_lower_db: float           # BRB lower sideband [dBc]
    brb_upper_db: float           # BRB upper sideband [dBc]
    bearing_upper_db: float       # Bearing f₀+BPFO sideband [dBc]
    bearing_lower_db: float       # Bearing |f₀-BPFO| sideband [dBc]
    harmonic_3rd_db: float        # 3rd harmonic [dBc]
    harmonic_5th_db: float        # 5th harmonic [dBc]
    negative_seq_ratio: float     # I₂/I₁ unbalance ratio (0–1)
    thd_f: float                  # THD-F (ratio, not percent)


def notch_filter(
    signal: np.ndarray,
    f_notch: float = F0,
    fs: float = FS_HZ,
    quality: float = 50.0,
) -> np.ndarray:
    """
    Apply IIR notch filter to suppress the fundamental frequency.

    Parameters
    ----------
    signal : np.ndarray
        1-D stator current time series [A].
    f_notch : float
        Notch center frequency [Hz]. Default: 50 Hz (fundamental).
    fs : float
        Sampling frequency [Hz].
    quality : float
        Quality factor Q = f_notch / bandwidth. Q=50 → 3-dB bandwidth = 1 Hz.
        Narrow bandwidth avoids attenuating BRB sidebands at f₀ ± 3.3 Hz.

    Returns
    -------
    np.ndarray
        Filtered signal [A]. Fundamental is attenuated ≥ 60 dB.

    Notes
    -----
    Uses iirnotch (SOS form) via sosfilt for numerical stability.
    sosfilt (causal, not zero-phase) is used here to avoid the phase
    distortion that filtfilt introduces near the notch — for spectral
    analysis purposes, phase response is irrelevant; what matters is
    the amplitude attenuation at f_notch.
    """
    b, a = iirnotch(f_notch, quality, fs=fs)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    # Zero-phase (sosfiltfilt) is correct for spectral analysis:
    # phase is discarded when computing PSD, and sosfiltfilt eliminates the
    # startup transient that causes ~24 dB instead of >60 dB attenuation
    # when measuring notch depth via FFT on the full time-domain signal.
    return sosfiltfilt(sos, signal)


def mcsa_psd(
    signal: np.ndarray,
    fs: float = FS_HZ,
    nperseg_s: float = 2.5,
    beta: float = 14.0,
    apply_notch: bool = True,
    f_notch: float = F0,
    notch_q: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    High-resolution one-sided PSD via Welch's method with Kaiser window.

    The Kaiser window with β=14 achieves −60 dB sidelobes — critical for
    resolving BRB sidebands that are 45–70 dB below the fundamental.
    A rectangular window (β=0) has only −13 dB sidelobes; BRB sidebands
    would be completely buried in leakage from the 50 Hz peak.

    Parameters
    ----------
    signal : np.ndarray
        1-D stator current [A].
    fs : float
        Sampling frequency [Hz].
    nperseg_s : float
        Welch segment duration [seconds]. Longer = better frequency resolution.
        2.5 s → Δf = 1/2.5 = 0.4 Hz. Resolves BRB sidebands at ±3.3 Hz from f₀.
    beta : float
        Kaiser window shape parameter. β=14 gives −60 dB sidelobes.
    apply_notch : bool
        If True, apply notch filter before PSD to suppress fundamental leakage.
    f_notch : float
        Notch frequency [Hz].
    notch_q : float
        Notch quality factor.

    Returns
    -------
    freqs : np.ndarray   Frequency axis [Hz], one-sided.
    psd   : np.ndarray   Power spectral density [A²/Hz].
    """
    if apply_notch:
        signal = notch_filter(signal, f_notch=f_notch, fs=fs, quality=notch_q)

    nperseg = int(nperseg_s * fs)
    window = np.kaiser(nperseg, beta)

    freqs, psd = welch(
        signal,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2,   # 50% overlap
        scaling='density',
    )
    return freqs, psd


def find_sideband_peak(
    freqs: np.ndarray,
    psd: np.ndarray,
    target_hz: float,
    search_window_hz: float = 2.0,
    min_prominence_db: float = 3.0,
) -> Tuple[Optional[float], float]:
    """
    Find the dominant spectral peak within a search window around target_hz.

    Parameters
    ----------
    freqs : np.ndarray   Frequency axis [Hz].
    psd : np.ndarray     Power spectral density [A²/Hz].
    target_hz : float    Expected sideband frequency [Hz].
    search_window_hz : float
        Search radius around target_hz [Hz]. Default: 2.0 Hz.
        Wider window accommodates slip estimation uncertainty.
    min_prominence_db : float
        Minimum peak prominence [dB] to qualify as a detected sideband.
        Prevents noise floor fluctuations from triggering false detections.

    Returns
    -------
    peak_freq : float or None   Detected peak frequency [Hz].
    peak_psd_db : float         Peak PSD in dB [10*log10(A²/Hz)], or
                                 noise floor estimate if no peak found.
    """
    mask = (freqs >= target_hz - search_window_hz) & \
           (freqs <= target_hz + search_window_hz)
    if not np.any(mask):
        return None, -200.0

    local_freqs = freqs[mask]
    local_psd = psd[mask]

    psd_db = 10 * np.log10(np.maximum(local_psd, 1e-20))

    # Find peaks with minimum prominence
    peaks, props = find_peaks(psd_db, prominence=min_prominence_db)

    if len(peaks) == 0:
        # No prominent peak: return noise floor estimate
        noise_floor = float(np.median(psd_db))
        return None, noise_floor

    # Select highest-amplitude peak
    best = peaks[np.argmax(psd_db[peaks])]
    return float(local_freqs[best]), float(psd_db[best])


def psd_at_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    target_hz: float,
    window_hz: float = 1.0,
) -> float:
    """Return max PSD [dB re A²/Hz] within ±window_hz of target_hz."""
    mask = (freqs >= target_hz - window_hz) & (freqs <= target_hz + window_hz)
    if not np.any(mask):
        return -200.0
    return float(10 * np.log10(np.max(psd[mask]) + 1e-20))


def compute_unbalance(signal_a: np.ndarray, signal_b: np.ndarray,
                      signal_c: np.ndarray, fs: float = FS_HZ) -> float:
    """
    Compute negative-sequence current ratio I₂/I₁ using Fortescue decomposition.

    For a balanced 3-phase system, I₂ ≈ 0. Under stator short circuit or
    phase loss, I₂/I₁ increases measurably.

    Method: extract fundamental phasor from each phase via FFT, then apply
    Fortescue transformation:
        a = exp(j*2π/3)
        I₁ = (1/3)(I_a + a*I_b + a²*I_c)   [positive sequence]
        I₂ = (1/3)(I_a + a²*I_b + a*I_c)   [negative sequence]

    Returns
    -------
    float : |I₂| / |I₁| in range [0, 1].
    Healthy motor: < 0.02 (< 2%).
    Inter-turn short (5% imbalance): typically 0.025–0.05.
    Severe fault (10%+ imbalance): > 0.05.
    """
    N = len(signal_a)

    def phasor_at_f0(signal: np.ndarray) -> complex:
        """Extract complex phasor at f₀ Hz via DFT."""
        fft = np.fft.rfft(signal) / N
        freqs = np.fft.rfftfreq(N, d=1.0 / fs)
        k = int(round(F0 * N / fs))
        k = min(k, len(fft) - 1)
        return fft[k] * 2  # one-sided → two-sided factor

    Ia = phasor_at_f0(signal_a)
    Ib = phasor_at_f0(signal_b)
    Ic = phasor_at_f0(signal_c)

    a = np.exp(1j * 2 * np.pi / 3)    # 120° rotation operator

    I1 = (Ia + a * Ib + a**2 * Ic) / 3
    I2 = (Ia + a**2 * Ib + a * Ic) / 3

    abs_I1 = abs(I1)
    if abs_I1 < 1e-9:
        return 0.0
    return float(abs(I2) / abs_I1)


def harmonic_content(
    signal: np.ndarray,
    fs: float = FS_HZ,
    harmonics: List[int] = None,
) -> Dict[str, float]:
    """
    Extract amplitudes of individual harmonics relative to fundamental.

    Parameters
    ----------
    signal : np.ndarray
    harmonics : list of harmonic orders to extract. Default: [3, 5, 7, 9].

    Returns
    -------
    dict {
        'h3_db': float,  # 3rd harmonic [dBc]
        'h5_db': float,
        'h7_db': float,
        'thd_f': float,  # THD-F (ratio, not percent)
    }
    """
    if harmonics is None:
        harmonics = [3, 5, 7, 9]

    N = len(signal)
    fft = np.fft.rfft(signal * np.hanning(N)) / (N * 0.5)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    def amplitude_at(f_target: float, window_hz: float = 0.5) -> float:
        """RMS amplitude at target frequency."""
        mask = (freqs >= f_target - window_hz) & (freqs <= f_target + window_hz)
        if not np.any(mask):
            return 1e-12
        return float(np.max(np.abs(fft[mask]))) / np.sqrt(2)

    A1 = amplitude_at(F0)
    if A1 < 1e-9:
        A1 = 1e-9

    result = {}
    harmonic_amps_sq = []
    for h in harmonics:
        Ah = amplitude_at(h * F0)
        db = 20 * np.log10(Ah / A1 + 1e-12)
        result[f'h{h}_db'] = float(db)
        harmonic_amps_sq.append(Ah ** 2)

    thd_f = np.sqrt(sum(harmonic_amps_sq)) / A1
    result['thd_f'] = float(thd_f)
    return result


def analyze(
    motor_signal,
    slip: float = 0.033,
    bpfo_hz: float = None,
) -> DiagnosticResult:
    """
    Run full diagnostic pipeline on a MotorSignal.

    Parameters
    ----------
    motor_signal : MotorSignal from mcsa_simulator.
    slip : float  Measured or estimated motor slip (default 0.033).
    bpfo_hz : float  BPFO for this bearing (default from simulator constants).

    Returns
    -------
    DiagnosticResult with all fault indicators populated.
    """
    from .mcsa_simulator import BPFO as DEFAULT_BPFO

    if bpfo_hz is None:
        bpfo_hz = DEFAULT_BPFO

    ia = motor_signal.phase_a
    ib = motor_signal.phase_b
    ic = motor_signal.phase_c
    fs = motor_signal.fs

    # High-resolution PSD with notch filter
    freqs, psd = mcsa_psd(ia, fs=fs, apply_notch=True)

    # PSD without notch (for fundamental level reference)
    freqs_raw, psd_raw = mcsa_psd(ia, fs=fs, apply_notch=False)
    fundamental_db = psd_at_frequency(freqs_raw, psd_raw, F0, window_hz=0.5)

    # BRB sidebands
    f_brb_lower = F0 * (1 - 2 * slip)
    f_brb_upper = F0 * (1 + 2 * slip)
    _, brb_lower_db = find_sideband_peak(freqs, psd, f_brb_lower)
    _, brb_upper_db = find_sideband_peak(freqs, psd, f_brb_upper)

    # Convert PSD levels to dBc (relative to fundamental power)
    # fundamental_db is absolute [dB A²/Hz]; sideband dBs are also absolute
    # dBc = sideband_abs_db - fundamental_abs_db
    brb_lower_dbc = brb_lower_db - fundamental_db
    brb_upper_dbc = brb_upper_db - fundamental_db

    # Bearing sidebands
    f_bear_upper = F0 + bpfo_hz
    f_bear_lower = abs(F0 - bpfo_hz)
    _, bear_upper_db = find_sideband_peak(freqs, psd, f_bear_upper)
    _, bear_lower_db = find_sideband_peak(freqs, psd, f_bear_lower)
    bearing_upper_dbc = bear_upper_db - fundamental_db
    bearing_lower_dbc = bear_lower_db - fundamental_db

    # Harmonics
    hc = harmonic_content(ia, fs=fs)

    # Phase unbalance
    neg_seq = compute_unbalance(ia, ib, ic, fs=fs)

    # Normalize PSD to dBc
    fund_level = np.max(psd_raw)
    psd_dbc = 10 * np.log10(np.maximum(psd, 1e-20)) - 10 * np.log10(fund_level + 1e-20)

    return DiagnosticResult(
        fault_type=motor_signal.fault_type,
        freqs_hz=freqs,
        psd_db=psd_dbc,
        fundamental_db=float(fundamental_db),
        brb_lower_db=float(brb_lower_dbc),
        brb_upper_db=float(brb_upper_dbc),
        bearing_upper_db=float(bearing_upper_dbc),
        bearing_lower_db=float(bearing_lower_dbc),
        harmonic_3rd_db=float(hc.get('h3_db', -100.0)),
        harmonic_5th_db=float(hc.get('h5_db', -100.0)),
        negative_seq_ratio=float(neg_seq),
        thd_f=float(hc.get('thd_f', 0.0)),
    )
