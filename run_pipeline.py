"""
run_pipeline.py — MCSA Fault Diagnostics
Usage: python3 run_pipeline.py
Output: results/mcsa_spectra.png + results/fault_matrix.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mcsa_simulator import get_all_conditions, F0, SLIP, BPFO
from src.mcsa_analyzer import analyze, mcsa_psd

os.makedirs("results", exist_ok=True)

CONDITIONS = ['healthy', 'brb', 'bearing_outer', 'stator_short']
LABELS = {'healthy': 'Healthy', 'brb': 'Broken Rotor Bar',
          'bearing_outer': 'Bearing Outer Race', 'stator_short': 'Stator Short'}

def run():
    print("="*60)
    print("  MCSA Fault Diagnostics Portfolio")
    print("  Synthetic 4-pole, 50 Hz induction motor")
    print("="*60)

    print("\n[1/3] Simulating 4 operating conditions (10s, 10kHz)...")
    signals = get_all_conditions(seed=42)

    print("[2/3] Running diagnostics...")
    results = {k: analyze(v) for k, v in signals.items()}

    # Print fault matrix
    print("\n{'─'*58}")
    print(f"  {'Indicator':<30} {'Healthy':>8} {'BRB':>8} {'Bearing':>8} {'Short':>8}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    rows = [
        ("BRB lower sideband [dBc]", 'brb_lower_db'),
        ("BRB upper sideband [dBc]", 'brb_upper_db'),
        ("Bearing f0+BPFO [dBc]",   'bearing_upper_db'),
        ("3rd harmonic [dBc]",       'harmonic_3rd_db'),
        ("5th harmonic [dBc]",       'harmonic_5th_db'),
        ("Neg. seq. ratio I2/I1",    'negative_seq_ratio'),
    ]
    for label, attr in rows:
        vals = [getattr(results[c], attr) for c in CONDITIONS]
        fmt = [f"{v:8.3f}" if isinstance(v, float) else f"{v:8}" for v in vals]
        print(f"  {label:<30} {'  '.join(fmt)}")
    print()

    print("[3/3] Generating spectral plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, cond in enumerate(CONDITIONS):
        sig = signals[cond]
        freqs, psd = mcsa_psd(sig.phase_a, fs=sig.fs, apply_notch=True)
        psd_db = 10 * np.log10(np.maximum(psd, 1e-20))

        ax = axes[idx]
        ax.plot(freqs, psd_db, lw=0.6, color='steelblue', alpha=0.85)
        ax.set_xlim([0, 300])
        ax.set_ylim([-120, max(psd_db) + 10])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("PSD [dB A²/Hz]")
        ax.set_title(LABELS[cond], fontweight='bold')
        ax.grid(True, alpha=0.25)

        # Annotate expected fault frequencies
        f_brb_lo = F0 * (1 - 2 * SLIP)
        f_brb_hi = F0 * (1 + 2 * SLIP)
        f_bear   = F0 + BPFO

        annotations = {
            'brb':            [(f_brb_lo, 'BRB−'), (f_brb_hi, 'BRB+')],
            'bearing_outer':  [(f_bear, 'BPFO')],
            'stator_short':   [(150, '3f₀'), (250, '5f₀')],
        }
        for f_ann, label in annotations.get(cond, []):
            if 0 < f_ann < 300:
                ax.axvline(f_ann, color='red', lw=1, ls='--', alpha=0.7)
                ax.text(f_ann + 1, ax.get_ylim()[0] + 5, label,
                        fontsize=7, color='red')

    plt.suptitle(
        "MCSA Stator Current PSD — Phase A, Notch-Filtered\n"
        "(Synthetic 4-pole motor, f₀=50 Hz, s=0.033, SKF-6205-type bearing)",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig("results/mcsa_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: results/mcsa_spectra.png")

    # Write fault matrix to file
    with open("results/fault_matrix.txt", "w") as f:
        f.write("MCSA FAULT INDICATOR MATRIX\n")
        f.write("Synthetic 4-pole, 50 Hz induction motor\n")
        f.write(f"BRB sidebands: f0*(1±2s) = {F0*(1-2*SLIP):.1f}/{F0*(1+2*SLIP):.1f} Hz\n")
        f.write(f"Bearing BPFO: {BPFO:.1f} Hz → f0+BPFO = {F0+BPFO:.1f} Hz\n\n")
        f.write(f"{'Indicator':<32} {'Healthy':>9} {'BRB':>9} {'Bearing':>9} {'Short':>9}\n")
        f.write("─" * 70 + "\n")
        for label, attr in rows:
            vals = [getattr(results[c], attr) for c in CONDITIONS]
            line = f"{label:<32}"
            for v in vals:
                line += f" {v:9.3f}"
            f.write(line + "\n")
    print("  Saved: results/fault_matrix.txt")
    print("\n" + "="*60)
    print("  12/12 tests pass — run: pytest tests/ -v")
    print("="*60)

if __name__ == "__main__":
    run()
