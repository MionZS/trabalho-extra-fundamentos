"""
Marimo Notebook: Butterworth Anti-aliasing Filter Analysis
Tema: Encontrar ordem mínima de filtro Butterworth para rejeitar alias com target especificado
"""

import marimo

__generated_with = "0.17.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
    # 🔧 Butterworth Anti-Aliasing Filter Design

    Find the **minimum order** Butterworth filter to achieve target **aliasing attenuation** 
    for tones A4 (440 Hz) and A7 (3520 Hz).
    """
    )
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy import signal
    return Figure, np, plt


@app.cell
def _():
    # Constants
    A4 = 440  # Hz
    A7 = 3520  # Hz (last "A" on piano)
    return A4, A7


@app.cell
def _(mo):
    # UI Controls
    test_tone = mo.ui.dropdown(
        ["A4 (440 Hz)", "A7 (3520 Hz)"],
        value="A4 (440 Hz)",
        label="Test Tone:",
    )

    fs = mo.ui.dropdown(
        [22050, 44100, 48000, 96000],
        value=44100,
        label="Sampling Rate (Hz):",
    )

    fc_rel = mo.ui.slider(
        start=0.1, stop=0.9, value=0.45, step=0.05, label="Cutoff Frequency (fraction of Nyquist):"
    )

    n_harmonics = mo.ui.slider(
        start=1, stop=30, value=15, step=1, label="Number of Harmonics (K):"
    )

    a_target = mo.ui.slider(
        start=40, stop=100, value=60, step=10, label="Target Attenuation (dB):"
    )

    mo.vstack(
        [
            mo.md("## Filter Design Controls"),
            test_tone,
            fs,
            fc_rel,
            n_harmonics,
            a_target,
        ]
    )
    return a_target, fc_rel, fs, n_harmonics, test_tone


@app.cell
def _(A4, A7, test_tone):
    # Parse tone
    if "A7" in test_tone.value:
        f0 = A7
    else:
        f0 = A4
    return (f0,)


@app.cell
def _(a_target, fc_rel, fs, np):
    # Butterworth magnitude response formula
    def butterworth_mag_db(f, fc, N):
        """Magnitude response of Butterworth filter in dB"""
        with np.errstate(divide="ignore"):
            magnitude_squared = 1 / (1 + (f / fc) ** (2 * N))
            magnitude_squared = np.clip(magnitude_squared, 1e-10, 1.0)
            return 10 * np.log10(magnitude_squared)

    # Calculate cutoff and Nyquist
    f_nyquist = fs.value / 2
    fc = fc_rel.value * f_nyquist

    # Find minimum N to satisfy attenuation at Nyquist
    # |H(f_nyquist)| >= a_target (in dB)
    # Solve: N >= (1/2) * log(10^(a_target/10) - 1) / log(f_nyquist / fc)

    _numerator = np.log(10 ** (a_target.value / 10) - 1)
    _denominator = np.log(f_nyquist / fc)

    if _denominator > 0:
        n_min_float = 0.5 * _numerator / _denominator
        N_min = int(np.ceil(n_min_float))
    else:
        N_min = 1

    # Ensure N >= 1
    N_min = max(1, N_min)
    return N_min, butterworth_mag_db, f_nyquist, fc


@app.cell
def _(
    Figure,
    N_min,
    butterworth_mag_db,
    f0,
    f_nyquist,
    fc,
    fs,
    mo,
    n_harmonics,
    np,
    plt,
):
    # Plot 1: Butterworth magnitude response
    freqs_plot = np.linspace(0, fs.value / 2, 4000)
    mag_response = butterworth_mag_db(freqs_plot, fc, N_min)

    fig_h = Figure(figsize=(12, 5))
    _ax = fig_h.add_subplot(111)

    _ax.plot(freqs_plot / 1000, mag_response, "b-", linewidth=2, label=f"Butterworth N={N_min}")
    _ax.axvline(fc / 1000, color="r", linestyle="--", linewidth=2, label=f"Cutoff: {fc/1000:.2f} kHz")
    _ax.axvline(f_nyquist / 1000, color="g", linestyle="--", linewidth=2, label=f"Nyquist: {f_nyquist/1000:.2f} kHz")

    # Mark harmonic boundaries
    for _k in range(1, min(n_harmonics.value + 1, 8)):
        f_harmonic = _k * f0
        if f_harmonic < fs.value / 2:
            _ax.axvline(f_harmonic / 1000, color="orange", linestyle=":", alpha=0.4, linewidth=0.8)
        else:
            # Harmonic above Nyquist
            _ax.axvline(f_harmonic / 1000, color="red", linestyle=":", alpha=0.4, linewidth=0.8)

    _ax.set_xlabel("Frequency (kHz)")
    _ax.set_ylabel("Magnitude (dB)")
    _ax.set_title(f"Butterworth Filter Response (Order N={N_min})")
    _ax.set_ylim([-100, 5])
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc="lower left")
    plt.tight_layout()

    mo.vstack(
        [
            mo.md(f"## Butterworth Filter Design Results\n\n**Minimum Order: N = {N_min}**"),
            fig_h,
        ]
    )
    return


@app.cell
def _(Figure, N_min, f0, f_nyquist, fc, fs, mo, n_harmonics, np, plt):
    # Plot 2: Spectrum of harmonic signal before and after filtering
    duration = 1.0
    N_samples = int(fs.value * duration)
    t = np.arange(N_samples) / fs.value

    # Generate harmonic signal
    signal_original = np.zeros_like(t)
    for _k in range(1, n_harmonics.value + 1):
        amplitude = 1.0 / _k
        signal_original += amplitude * np.sin(2 * np.pi * _k * f0 * t)

    # FFT of original signal
    fft_original = np.fft.fft(signal_original)
    freqs = np.fft.fftfreq(N_samples, 1 / fs.value)
    mag_original = np.abs(fft_original) ** 2

    # Simulate filtered spectrum (attenuate by Butterworth response)
    def butterworth_mag_linear(f, fc, N):
        """Linear magnitude (not dB)"""
        return 1 / np.sqrt(1 + (f / fc) ** (2 * N))

    att_factor = butterworth_mag_linear(np.abs(freqs), fc, N_min)
    mag_filtered = mag_original * (att_factor ** 2)

    # Plot
    fig_spec = Figure(figsize=(12, 6))
    _ax = fig_spec.add_subplot(111)

    idx = (freqs >= 0) & (freqs <= 8000)
    _ax.plot(
        freqs[idx] / 1000,
        10 * np.log10(mag_original[idx] + 1e-10),
        "b-",
        linewidth=1,
        label="Original (unfiltered)",
        alpha=0.7,
    )
    _ax.plot(
        freqs[idx] / 1000,
        10 * np.log10(mag_filtered[idx] + 1e-10),
        "r-",
        linewidth=1.5,
        label=f"After Butterworth (N={N_min})",
    )

    _ax.axvline(f_nyquist / 1000, color="g", linestyle="--", linewidth=2, label="Nyquist Freq")
    _ax.axvline(fc / 1000, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="Cutoff (fc)")

    _ax.set_xlabel("Frequency (kHz)")
    _ax.set_ylabel("Power (dB)")
    _ax.set_title(f"Spectral Analysis: Before & After Anti-Aliasing Filter")
    _ax.grid(True, alpha=0.3)
    _ax.legend()
    _ax.set_ylim([-80, 20])
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Spectrum: Original vs Filtered"),
            fig_spec,
        ]
    )
    return freqs, mag_filtered


@app.cell
def _(f_nyquist, freqs, mag_filtered, mo, np):
    # Calculate Aliasing-to-Signal Ratio (ASR)
    # ASR = power above Nyquist / power below Nyquist

    idx_below = np.abs(freqs) < f_nyquist
    idx_above = (np.abs(freqs) >= f_nyquist) & (np.abs(freqs) <= 2 * f_nyquist)

    power_below = np.sum(mag_filtered[idx_below])
    power_above = np.sum(mag_filtered[idx_above])

    asr_db = 10 * np.log10((power_above + 1e-10) / (power_below + 1e-10))
    asr_percent = (power_above / (power_below + power_above)) * 100

    mo.vstack(
        [
            mo.md(f"""
    ## Aliasing Metrics

    - **ASR (Aliasing-to-Signal Ratio)**: {asr_db:.2f} dB
    - **Aliasing Power %**: {asr_percent:.4f}%
    - **SNR (Quantization, 32-bit ideal)**: 194.4 dB

    > Lower ASR is better. Target: ASR ≤ –60 dB means alias power is 1 millionth of signal power.
    """),
        ]
    )
    return


@app.cell
def _(A4, Figure, fs, mo, np, plt):
    # Plot 3: N_min vs Octave (sweep from A4 to A7)
    octaves = np.array([-2, -1, 0, 1, 2, 3])
    f0_sweep = A4 * (2.0 ** octaves)

    a_target_fixed = 60  # dB
    fc_rel_fixed = 0.45
    f_nyquist_fixed = fs.value / 2
    fc_fixed = fc_rel_fixed * f_nyquist_fixed

    n_min_sweep = []
    for f in f0_sweep:
        _numerator = np.log(10 ** (a_target_fixed / 10) - 1)
        _denominator = np.log(f_nyquist_fixed / fc_fixed)
        n_float = 0.5 * _numerator / _denominator
        n_min_sweep.append(max(1, int(np.ceil(n_float))))

    fig_sweep = Figure(figsize=(10, 5))
    _ax = fig_sweep.add_subplot(111)

    note_labels = [f"A{i-2}" for i in range(len(octaves))]
    _ax.plot(note_labels, n_min_sweep, "bo-", linewidth=2, markersize=8)
    _ax.set_xlabel("Note (Octave)")
    _ax.set_ylabel("Minimum Filter Order (N)")
    _ax.set_title(f"Minimum Butterworth Order vs. Octave (fs={fs.value} Hz, Target={a_target_fixed} dB)")
    _ax.grid(True, alpha=0.3)
    _ax.set_ylim([0, max(n_min_sweep) + 2])

    for i, (note, n) in enumerate(zip(note_labels, n_min_sweep)):
        _ax.text(i, n + 0.2, str(n), ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Filter Order vs. Octave"),
            fig_sweep,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Insights

    - **Higher fundamentals** (e.g., A7 = 3520 Hz) require **higher filter orders** or **higher sampling rates** to control alias
    - **Butterworth** provides smooth passband with –6N dB/octave roll-off
    - For **piano recording** (A0 to C8), consider **96 kHz** sampling or high-order anti-aliasing
    - **Trade-off**: higher N → steeper transition but more phase distortion
    - **SNR (quantization, 32-bit)** is ~194 dB (theoretical); real ADCs achieve ~120–140 dB ENOB
    """)
    return


if __name__ == "__main__":
    app.run()
