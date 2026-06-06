# MCSA Fault Diagnostics Portfolio

Motor Current Signature Analysis (MCSA) for non-intrusive induction motor fault detection. Identifies three fault classes from stator current alone — no vibration sensors, no shaft encoders, no thermal cameras required.

**Data source: synthetic simulation. NOT a real motor or testbed measurement.**

The simulator models the electromagnetic physics of a 4-pole, 50 Hz induction motor (SKF 6205-type bearing geometry, rated 10 A). Pipeline is designed to accept real stator current data when available.

---

## Physical basis

Under healthy conditions, stator current is dominated by the 50 Hz fundamental. Each fault class introduces specific spectral components:

| Fault | Mechanism | Spectral signature |
|-------|-----------|-------------------|
| **Broken Rotor Bar (BRB)** | Rotor asymmetry modulates air-gap flux | Sidebands at f₀(1 ± 2ks), k=1,2 |
| **Bearing outer-race defect** | Periodic vibration at BPFO modulates eccentricity | Sidebands at \|f₀ ± m·BPFO\|, m=1,2 |
| **Stator inter-turn short** | Phase amplitude imbalance + odd harmonic rise | Elevated 3rd/5th/7th harmonics + I₂/I₁ > 2% |

Motor parameters: 4-pole, f₀=50 Hz, slip s=0.033 → shaft f_r=24.17 Hz  
BRB sidebands: 50·(1−0.066)=**46.7 Hz** and 50·(1+0.066)=**53.3 Hz**  
Bearing BPFO (SKF 6205-type, 9 balls): **86.4 Hz** → sidebands at **136.4 Hz** and **36.4 Hz**

---

## Pipeline

1. **Fundamental suppression** — IIR notch filter at 50 Hz (Q=50, BW=1 Hz). Zero-phase (`sosfiltfilt`) eliminates startup transients. Passband gain at ±3.3 Hz (BRB sidebands): < 0.1 dB loss.

2. **High-resolution PSD** — Welch's method, Kaiser window (β=14, −60 dB sidelobes), 2.5-second segments, 50% overlap. Resolution: 0.4 Hz/bin. Required to resolve BRB sidebands 3.3 Hz from the fundamental.

3. **Sideband tracker** — Peak search in ±2 Hz windows around slip-predicted frequencies. Prominence threshold: 3 dB above local noise floor.

4. **Phase unbalance (Fortescue)** — Negative-sequence ratio I₂/I₁. Healthy: < 0.02. Inter-turn short (5% imbalance): > 0.015.

---

## Results (synthetic simulation)

| Indicator | Healthy | BRB | Bearing | Stator Short |
|-----------|---------|-----|---------|-------------|
| BRB lower sideband [dBc] | −97 | **−45** | −97 | −97 |
| BRB upper sideband [dBc] | −97 | **−48** | −97 | −97 |
| Bearing f₀+BPFO [dBc] | −97 | −97 | **−50** | −97 |
| 3rd harmonic [dBc] | −100 | −100 | −100 | **−30** |
| 5th harmonic [dBc] | −103 | −103 | −103 | **−36** |
| Neg. seq. ratio I₂/I₁ | 0.000 | 0.000 | 0.000 | **0.017** |

Each fault type has a unique fingerprint. Healthy noise floor ≈ −97 dBc.

---

## Quick start

```bash
pip install -r requirements.txt
python3 run_pipeline.py          # generates results/mcsa_spectra.png
pytest tests/ -v                 # 12 tests, all pass
```

---

## Library usage

```python
from src.mcsa_simulator import simulate_brb, simulate_healthy
from src.mcsa_analyzer import mcsa_psd, find_sideband_peak, compute_unbalance

# Simulate or load real stator current
signal = simulate_brb(slip=0.033, sideband_db=-45.0)

# High-resolution PSD with fundamental suppression
freqs, psd = mcsa_psd(signal.phase_a, fs=signal.fs, apply_notch=True)

# Check for BRB lower sideband
f_target = 50.0 * (1 - 2*0.033)   # 46.7 Hz
peak_f, peak_db = find_sideband_peak(freqs, psd, f_target)

# Phase unbalance (stator short indicator)
ratio = compute_unbalance(signal.phase_a, signal.phase_b, signal.phase_c)
```

---

## Connection to power_signal_tools

This repo is a domain-specific application of the spectral and signal processing primitives in [power_signal_tools](https://github.com/VolMax-Studio/power_signal_tools). The Kaiser window PSD and IIR filter design are directly portable. MCSA adds the fault-physics layer: slip-based sideband prediction and Fortescue sequence analysis.

## Domain background

The same non-intrusive principle as NILM (load disaggregation from aggregate current) applied to rotating machines: extract machine state from a single current sensor. Ivan's 20+ years of industrial electrical experience with motors and drives informs the fault severity levels and the physical parameters used in the simulator.

## License

MIT
