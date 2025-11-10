"""
Marimo Notebook: Piano Synthesis and Frequency Analysis
Tema: Síntese de notas de piano com harmônicos, fading in/out, e análise de frequências
"""

import marimo

__generated_with = "0.17.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md("""
    # 🎹 Piano Synthesis & Frequency Analysis

    Explore synthesized piano notes with harmonics, fading envelopes, and frequency visualization.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    return Figure, np, plt


@app.cell
def _():
    # Piano frequency range
    A0 = 27.5  # Hz
    C8 = 4186  # Hz
    A4 = 440  # Hz (ISO 16 standard)
    A7 = 3520  # Hz

    # Piano: 88 keys spanning A0 to C8
    piano_keys = {
        "A0": A0,
        "A1": A0 * 2,
        "A2": A0 * 4,
        "A3": A0 * 8,
        "A4": A4,
        "A5": A4 * 2,
        "A6": A4 * 4,
        "A7": A7,
        "C1": 32.70,
        "C4": 261.63,
        "C8": C8,
    }
    return A4, piano_keys


@app.cell
def _(mo):
    # UI Controls
    key_selector = mo.ui.dropdown(
        ["A4 (440 Hz)", "A7 (3520 Hz)", "C4 (261.63 Hz)", "C1 (32.70 Hz)"],
        value="A4 (440 Hz)",
        label="Piano Key:",
    )

    n_harmonics = mo.ui.slider(
        start=1, stop=25, value=10, step=1, label="Number of Harmonics (K):"
    )

    duration = mo.ui.slider(
        start=0.5, stop=5.0, value=2.0, step=0.5, label="Duration (s):"
    )

    fade_time = mo.ui.slider(
        start=0.1, stop=1.0, value=0.3, step=0.1, label="Fade Time (s):"
    )

    fs = mo.ui.dropdown(
        [8000, 16000, 22050, 44100, 48000, 96000],
        value=44100,
        label="Sampling Rate (Hz):",
    )

    play_button = mo.ui.button(label="▶ Play Note", on_click=lambda _: None)
    stop_button = mo.ui.button(label="⏹ Stop", on_click=lambda _: None)

    mo.vstack(
        [
            mo.md("## Controls"),
            key_selector,
            n_harmonics,
            duration,
            fade_time,
            fs,
            mo.hstack([play_button, stop_button]),
        ]
    )
    return duration, fade_time, fs, key_selector, n_harmonics


@app.cell
def _(A4, key_selector):
    # Parse selected key
    key_str = key_selector.value.split("(")[1].strip(")")
    if "Hz" in key_str:
        f0 = float(key_str.replace(" Hz", ""))
    else:
        f0 = A4  # default
    return (f0,)


@app.cell
def _(duration, f0, fade_time, fs, n_harmonics, np):
    # Generate harmonic piano note
    N = int(fs.value * duration.value)
    t = np.arange(N) / fs.value

    # Envelope: linear fade in/out
    fade_samples = min(int(fade_time.value * fs.value), N // 2)
    envelope = np.ones_like(t)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

    # Harmonic series (amplitude decays as 1/k)
    signal = np.zeros_like(t)
    for _k in range(1, n_harmonics.value + 1):
        amplitude = 1.0 / _k
        signal += amplitude * np.sin(2 * np.pi * _k * f0 * t)

    # Apply envelope
    signal = signal * envelope

    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.9
    return signal, t, envelope


@app.cell
def _(Figure, mo, plt, signal, t, envelope):
    # Time-domain plot: signal with envelope (improved for clarity)
    fig_time = Figure(figsize=(12, 5))
    _ax = fig_time.add_subplot(111)

    # Plot main waveform as a line
    _ax.plot(t, signal, color="tab:blue", linewidth=0.9, alpha=0.9, label="Synthesized Piano Note")

    # Add semi-transparent envelope (no heavy filling)
    _ax.fill_between(t, -envelope, envelope, color="gray", alpha=0.12, label="Envelope")

    # Draw sample markers sparsely to avoid overplotting
    _N = len(t)
    max_markers = 1000
    step = max(1, _N // max_markers)
    _ax.plot(t[::step], signal[::step], marker='o', linestyle='None', markersize=2, alpha=0.35, color='tab:blue', label='Sample points')

    # Add a small zoom inset to show waveform detail at the start
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axins = inset_axes(_ax, width="30%", height="30%", loc='upper right')
        zoom_samples = min(1000, _N)
        axins.plot(t[:zoom_samples], signal[:zoom_samples], color='tab:blue', linewidth=0.9)
        axins.set_title('Zoom (start)', fontsize=8)
        axins.set_xlabel('s', fontsize=8)
        axins.set_ylabel('A', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=7)
    except Exception:
        # If inset is unavailable, continue without it
        pass

    # Adjust y-limits and labels
    y_max = max(abs(signal.min()), abs(signal.max()), 1e-6)
    _ax.set_ylim(-y_max * 1.15, y_max * 1.15)
    _ax.set_xlabel("Time (s)")
    _ax.set_ylabel("Amplitude")
    _ax.set_title("Piano Note Synthesis (Time Domain)")
    _ax.grid(True, alpha=0.3)
    _ax.legend(loc='upper right')
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Waveform (Time Domain)"),
            fig_time,
        ]
    )
    return


@app.cell
def _(Figure, f0, fs, mo, n_harmonics, np, plt, signal):
    # Frequency-domain plot: FFT
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / fs.value)
    mag = np.abs(fft_signal) ** 2

    # Only plot positive frequencies up to 10 kHz
    idx = (freqs >= 0) & (freqs <= 10000)

    # Use dB scale
    mag_db = 10 * np.log10(mag + 1e-10)

    fig_freq = Figure(figsize=(12, 5))
    _ax_freq = fig_freq.add_subplot(111)

    # Plot spectrum as a clean line (no fills)
    _ax_freq.plot(freqs[idx], mag_db[idx], color='tab:red', linewidth=1.0, label='Spectrum')

    # Mark harmonics (vertical dashed lines) and label them
    ylim = _ax_freq.get_ylim()
    for _k in range(1, min(n_harmonics.value + 1, 30)):
        f_harmonic = _k * f0
        if f_harmonic < 0 or f_harmonic > 10000:
            continue
        _ax_freq.axvline(f_harmonic, color='green', linestyle='--', alpha=0.35, linewidth=0.8)
        _ax_freq.text(f_harmonic, ylim[0] + (ylim[1] - ylim[0]) * 0.05, str(_k), ha='center', va='bottom', fontsize=8, color='green')

    _ax_freq.set_xlabel("Frequency (Hz)")
    _ax_freq.set_ylabel("Power (dB)")
    _ax_freq.set_title(f"Frequency Spectrum: {f0:.1f} Hz (Fundamental) + Harmonics")
    _ax_freq.grid(True, alpha=0.25)
    _ax_freq.set_xlim(0, min(10000, fs.value / 2))
    plt.tight_layout()

    mo.vstack([
        mo.md("## Frequency Spectrum"),
        fig_freq,
    ])
    return


@app.cell
def _(Figure, mo, piano_keys, plt):
    # All piano keys frequency chart
    notes = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "C1", "C4", "C8"]
    frequencies = [piano_keys[note] for note in notes]

    fig_piano = Figure(figsize=(12, 6))
    _ax_piano = fig_piano.add_subplot(111)

    colors = ["red" if note.startswith("A") else "blue" for note in notes]
    bars = _ax_piano.bar(range(len(notes)), frequencies, color=colors, alpha=0.7)

    _ax_piano.set_xticks(range(len(notes)))
    _ax_piano.set_xticklabels(notes, rotation=45)
    _ax_piano.set_ylabel("Frequency (Hz)")
    _ax_piano.set_title("Piano Key Frequencies (Note: Red = A notes, Blue = C notes)")
    _ax_piano.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (note, freq) in enumerate(zip(notes, frequencies)):
        _ax_piano.text(i, freq + 100, f"{freq:.1f}", ha="center", fontsize=9)

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Piano Key Frequencies"),
            fig_piano,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    - **Fundamental Frequency**: Set via key selector (A4 = 440 Hz standard)
    - **Harmonics**: Added with amplitude ∝ 1/k (realistic piano timbre)
    - **Envelope**: Linear fade in/out for smooth sound
    - **Sampling Rate**: Variable; higher fs allows more harmonics without aliasing
    - **A7 (3520 Hz)** is the highest "A" on the piano — demanding for anti-aliasing filters!
    """)
    return


if __name__ == "__main__":
    app.run()
