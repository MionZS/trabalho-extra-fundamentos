"""
Marimo Notebook: Noisy Signal Filtering with Butterworth LP Filter
Tema: Adicionar ruído branco gaussiano e filtrar com Butterworth para demonstrar rejeição de ruído
"""
import marimo as mo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import signal as scipy_signal

app = mo.App()


@app.cell
def _():
    mo.md(
        """
    # 🔊 Noisy Signal Filtering & Noise Attenuation
    
    Add Gaussian white noise to a signal and demonstrate filtering with a **Butterworth low-pass filter**.
    Visualize **SNR improvement** and spectral attenuation.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy import signal as scipy_signal

    return np, plt, Figure, scipy_signal


@app.cell
def _():
    # UI Controls
    signal_type = mo.ui.dropdown(
        ["Sine (440 Hz)", "Piano A4 (440 Hz + Harmonics)", "Chirp (100-1000 Hz)"],
        value="Sine (440 Hz)",
        label="Base Signal:",
    )

    noise_snr_db = mo.ui.slider(
        start=-10, stop=40, value=10, step=5, label="Input SNR (dB):"
    )

    fs = mo.ui.dropdown(
        [8000, 16000, 22050, 44100, 48000, 96000],
        value=44100,
        label="Sampling Rate (Hz):",
    )

    filter_order = mo.ui.slider(
        start=1, stop=10, value=4, step=1, label="Filter Order (N):"
    )

    fc_rel = mo.ui.slider(
        start=0.05, stop=0.9, value=0.4, step=0.05, label="Cutoff Frequency (fraction of Nyquist):"
    )

    duration = mo.ui.slider(
        start=0.5, stop=5.0, value=2.0, step=0.5, label="Duration (s):"
    )

    mo.vstack(
        [
            mo.md("## Signal & Filter Controls"),
            signal_type,
            noise_snr_db,
            fs,
            filter_order,
            fc_rel,
            duration,
        ]
    )

    return signal_type, noise_snr_db, fs, filter_order, fc_rel, duration


@app.cell
def _(signal_type, fs, duration, np):
    # Generate base signal
    N = int(fs.value * duration.value)
    t = np.arange(N) / fs.value

    if "Sine" in signal_type.value:
        # Simple sine
        f0 = 440
        signal_clean = np.sin(2 * np.pi * f0 * t)

    elif "Piano" in signal_type.value:
        # Piano A4 with harmonics
        f0 = 440
        signal_clean = np.zeros_like(t)
        for k in range(1, 11):
            amplitude = 1.0 / k
            signal_clean += amplitude * np.sin(2 * np.pi * k * f0 * t)
        signal_clean = signal_clean / np.max(np.abs(signal_clean))

    elif "Chirp" in signal_type.value:
        # Chirp from 100 to 1000 Hz
        f_start, f_end = 100, 1000
        signal_clean = scipy_signal.chirp(t, f_start, t[-1], f_end)

    return N, t, signal_clean


@app.cell
def _(signal_clean, noise_snr_db, np):
    # Add Gaussian white noise
    # SNR_dB = 10 * log10(P_signal / P_noise)
    # => P_noise = P_signal / 10^(SNR_dB/10)

    power_signal = np.mean(signal_clean**2)
    snr_linear = 10 ** (noise_snr_db.value / 10)
    power_noise = power_signal / snr_linear

    noise = np.sqrt(power_noise) * np.random.randn(len(signal_clean))
    signal_noisy = signal_clean + noise

    return power_signal, snr_linear, power_noise, noise, signal_noisy


@app.cell
def _(signal_noisy, fs, filter_order, fc_rel, scipy_signal, np):
    # Design and apply Butterworth filter
    f_nyquist = fs.value / 2
    fc_normalized = fc_rel.value  # Already a fraction of Nyquist

    # scipy.signal.butter expects normalized frequency: fc / (fs/2)
    sos = scipy_signal.butter(
        filter_order.value, fc_normalized, btype="low", output="sos"
    )

    # Apply filter
    signal_filtered = scipy_signal.sosfilt(sos, signal_noisy)

    return sos, fc_normalized, signal_filtered


@app.cell
def _(signal_clean, signal_noisy, signal_filtered, np):
    # Calculate SNR before and after filtering
    # SNR = P_signal / P_noise

    def calculate_snr(clean, noisy):
        error = noisy - clean
        snr = np.mean(clean**2) / (np.mean(error**2) + 1e-10)
        return 10 * np.log10(snr)

    snr_input = calculate_snr(signal_clean, signal_noisy)
    snr_output = calculate_snr(signal_clean, signal_filtered)

    snr_improvement = snr_output - snr_input

    return calculate_snr, snr_input, snr_output, snr_improvement


@app.cell
def _(Figure, t, signal_clean, signal_noisy, signal_filtered, plt):
    # Plot 1: Time-domain signals
    fig_time = Figure(figsize=(14, 8))

    ax1 = fig_time.add_subplot(3, 1, 1)
    ax1.plot(t, signal_clean, "b-", linewidth=1, label="Clean Signal")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Time Domain: Clean Signal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig_time.add_subplot(3, 1, 2)
    ax2.plot(t, signal_noisy, "r-", linewidth=0.5, alpha=0.8, label="Noisy Signal")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Time Domain: Signal + Gaussian White Noise")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig_time.add_subplot(3, 1, 3)
    ax3.plot(t, signal_filtered, "g-", linewidth=0.8, label="Filtered Signal")
    ax3.plot(t, signal_clean, "b--", linewidth=0.8, alpha=0.6, label="Reference")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Amplitude")
    ax3.set_title("Time Domain: After Butterworth Filter")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Time-Domain Comparison"),
            fig_time,
        ]
    )

    return fig_time


@app.cell
def _(Figure, signal_clean, signal_noisy, signal_filtered, fs, fc_rel, plt, np):
    # Plot 2: Frequency-domain (FFT)
    fft_clean = np.fft.fft(signal_clean)
    fft_noisy = np.fft.fft(signal_noisy)
    fft_filtered = np.fft.fft(signal_filtered)

    freqs = np.fft.fftfreq(len(signal_clean), 1 / fs.value)

    # Positive frequencies only
    idx = freqs >= 0

    mag_clean = np.abs(fft_clean[idx]) ** 2
    mag_noisy = np.abs(fft_noisy[idx]) ** 2
    mag_filtered = np.abs(fft_filtered[idx]) ** 2

    fig_freq = Figure(figsize=(14, 6))
    ax = fig_freq.add_subplot(111)

    # Limit frequency display to 5 kHz for clarity
    idx_plot = freqs[idx] <= 5000

    ax.plot(
        freqs[idx][idx_plot],
        10 * np.log10(mag_clean[idx_plot] + 1e-10),
        "b-",
        linewidth=1.5,
        label="Clean Signal",
    )
    ax.plot(
        freqs[idx][idx_plot],
        10 * np.log10(mag_noisy[idx_plot] + 1e-10),
        "r-",
        linewidth=0.8,
        alpha=0.7,
        label="Noisy Signal",
    )
    ax.plot(
        freqs[idx][idx_plot],
        10 * np.log10(mag_filtered[idx_plot] + 1e-10),
        "g-",
        linewidth=1.5,
        label="Filtered Signal",
    )

    cutoff_freq_hz = fc_rel.value * (fs.value / 2)
    ax.axvline(cutoff_freq_hz / 1000, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="Cutoff freq")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Frequency Domain: Clean, Noisy, and Filtered Signals")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 5000])
    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Frequency-Domain Comparison"),
            fig_freq,
        ]
    )

    return fig_freq, fft_clean, fft_noisy, fft_filtered, freqs, mag_clean, mag_noisy, mag_filtered


@app.cell
def _(Figure, sos, fs, fc_rel, np, plt, scipy_signal):
    # Plot 3: Butterworth filter response (magnitude and phase)
    w, h = scipy_signal.sosfreqz(sos, fs=fs.value)

    fig_response = Figure(figsize=(14, 6))

    ax1 = fig_response.add_subplot(1, 2, 1)
    ax1.plot(w / 1000, 20 * np.log10(np.abs(h) + 1e-10), "b-", linewidth=2)
    cutoff_freq_hz = fc_rel.value * (fs.value / 2)
    ax1.axvline(cutoff_freq_hz / 1000, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="Cutoff")
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Butterworth Filter: Magnitude Response")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-80, 5])
    ax1.legend()

    ax2 = fig_response.add_subplot(1, 2, 2)
    ax2.plot(w / 1000, np.angle(h) * 180 / np.pi, "r-", linewidth=2)
    ax2.set_xlabel("Frequency (kHz)")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Butterworth Filter: Phase Response")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    mo.vstack(
        [
            mo.md("## Filter Frequency & Phase Response"),
            fig_response,
        ]
    )

    return fig_response, w, h


@app.cell
def _(snr_input, snr_output, snr_improvement):
    # Summary metrics
    mo.vstack(
        [
            mo.md(f"""
    ## Performance Metrics
    
    | Metric | Value |
    |--------|-------|
    | **Input SNR** | {snr_input:.2f} dB |
    | **Output SNR** | {snr_output:.2f} dB |
    | **SNR Improvement** | {snr_improvement:.2f} dB |
    
    > The filter attenuates high-frequency noise while preserving the signal.
    > Larger improvements indicate more effective noise rejection.
    """),
        ]
    )

    return


@app.cell
def _():
    mo.md(
        """
    ## Design Insights
    
    - **Butterworth** filters are popular for audio due to smooth passband (no ripple)
    - **Roll-off**: –6N dB/octave means a 4th-order filter attenuates by ~24 dB/octave
    - **Trade-off**: higher order → steeper transition but more phase distortion
    - **Real-world**: combine anti-aliasing (analog) + decimation (digital) for best results
    - **Noise**: Gaussian white noise is broadband; filtering removes high-frequency components
    - For music/piano, preserving phase is often more important than aggressive filtering
    
    ### Next Steps
    - Experiment with different filter orders and cutoff frequencies
    - Compare with **Chebyshev** filters (steeper but with passband ripple)
    - Add **equalizer** stages for frequency-selective filtering
    """
    )
    return


if __name__ == "__main__":
    app.run()
