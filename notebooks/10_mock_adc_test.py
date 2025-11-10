import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy import signal
    from scipy.signal import freqs
    return Figure, freqs, mo, np, plt, signal


@app.cell
def _(mo):
    mo.md(r"""
    # Filtro Butterworth para Tons de Piano
    """)
    return


@app.cell
def _():
    # Only A notes (Lás): A0 to A7
    piano_notes = {}
    for octave in range(0, 8):
        freq = 27.5 * (2 ** octave)
        piano_notes[f"A{octave} ({freq:.2f} Hz)"] = float(freq)
    default_note = "A4 (440.00 Hz)" if "A4 (440.00 Hz)" in piano_notes else next(iter(piano_notes))
    return default_note, piano_notes


@app.cell
def _(default_note, mo, piano_notes):
    tone = mo.ui.dropdown(options=list(piano_notes.keys()), value=default_note, label="Selecione o Tom:")
    mo.md(f"{tone}")
    return tone,


@app.cell
def _(piano_notes, tone, np):
    # Get tone properties
    f0 = piano_notes[tone.value]
    T = 1.0 / f0
    A = 1.0
    
    # Generate multiple periods with very high density for genuine simulation
    num_periods = 10  # Increase for better FFT resolution
    fs_display = 4410000  # 100x higher density (4.41 MHz) for ultra-dense continuous signal
    samples = int(fs_display * T * num_periods)
    t = np.linspace(0, T * num_periods, samples, endpoint=False)
    x = A * np.sin(2 * np.pi * f0 * t)
    
    return A, T, f0, fs_display, num_periods, samples, t, x


@app.cell
def _(mo, f0, T, A):
    mo.md(f"""
    ### Características do Tom
    
    - **Frequência (f):** {f0:.2f} Hz
    - **Período (T):** {T*1e3:.3f} ms
    - **Amplitude (A):** {A:.2f}
    """)
    return


@app.cell
def _(Figure, plt, t, x, T):
    # Plot only one period for clarity
    mask_one_period = t < T
    fig_time = Figure(figsize=(12, 4))
    ax_time = fig_time.add_subplot(1, 1, 1)
    ax_time.plot(t[mask_one_period] * 1e3, x[mask_one_period], 'b-', linewidth=2)
    ax_time.set_xlabel('Tempo (ms)', fontsize=11)
    ax_time.set_ylabel('Amplitude', fontsize=11)
    ax_time.set_title('Sinal: Um Período (Alta Densidade)', fontsize=12)
    ax_time.grid(True, alpha=0.3)
    fig_time.tight_layout()
    plt.close(fig_time)
    fig_time
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Amostragem e Quantização ADC
    """)
    return


@app.cell
def _(mo):
    bits = mo.ui.slider(4, 16, step=1, value=8, label="Bits de resolução:", show_value=True)
    sampling_freq = mo.ui.text(value="44100", label="Frequência de Amostragem (Hz):")
    
    mo.md(f"""
    {bits}
    
    {sampling_freq}
    """)
    return bits, sampling_freq


@app.cell
def _(A, bits, f0, np, sampling_freq, T):
    # Sampling and Quantization
    fs_str = sampling_freq.value.strip()
    fs = float(fs_str) if fs_str else 44100  # Default to 44100 if empty
    t_sample = np.arange(0, T, 1/fs)
    x_sample = A * np.sin(2 * np.pi * f0 * t_sample)
    
    levels = 2 ** bits.value
    delta = 2 * A / (levels - 1)  # For bipolar signal -A to A
    x_q = np.round(x_sample / delta) * delta
    noise = x_sample - x_q
    
    # Check for aliasing
    aliasing_detected = fs < 2 * f0
    
    return aliasing_detected, delta, fs, levels, noise, t_sample, x_q, x_sample


@app.cell
def _(Figure, plt, t_sample, x_q):
    fig_q = Figure(figsize=(12, 4))
    ax_q = fig_q.add_subplot(1, 1, 1)
    ax_q.plot(t_sample * 1e3, x_q, 'r.-', linewidth=1, markersize=3)
    ax_q.set_xlabel('Tempo (ms)', fontsize=11)
    ax_q.set_ylabel('Amplitude', fontsize=11)
    ax_q.set_title('Sinal Amostrado e Quantizado', fontsize=12)
    ax_q.grid(True, alpha=0.3)
    fig_q.tight_layout()
    plt.close(fig_q)
    fig_q
    return


@app.cell
def _(Figure, plt, noise, t_sample):
    fig_noise = Figure(figsize=(12, 4))
    ax_noise = fig_noise.add_subplot(1, 1, 1)
    ax_noise.plot(t_sample * 1e3, noise, 'g.-', linewidth=1, markersize=3)
    ax_noise.set_xlabel('Tempo (ms)', fontsize=11)
    ax_noise.set_ylabel('Amplitude', fontsize=11)
    ax_noise.set_title('Ruído de Quantização', fontsize=12)
    ax_noise.grid(True, alpha=0.3)
    fig_noise.tight_layout()
    plt.close(fig_noise)
    fig_noise
    return


@app.cell
def _(Figure, aliasing_detected, f0, fs, np, plt, x_q):
    fig_spectrum = Figure(figsize=(12, 4))
    ax_spectrum = fig_spectrum.add_subplot(1, 1, 1)
    # FFT with zero-padding for better resolution
    N_pad = 8192
    yf_spec = np.fft.fft(x_q, n=N_pad)
    xf_spec = np.fft.fftfreq(N_pad, 1/fs)
    mag = np.abs(yf_spec)
    # Plot positive frequencies
    mask_pos_spec = xf_spec >= 0
    ax_spectrum.plot(xf_spec[mask_pos_spec], 20 * np.log10(mag[mask_pos_spec] + 1e-12), 'b-', linewidth=2)
    
    # Mark fundamental frequency (aliased if necessary)
    aliased_f0 = f0 % fs
    if aliased_f0 > fs / 2:
        aliased_f0 = fs - aliased_f0
    ax_spectrum.axvline(aliased_f0, color='g', linestyle=':', linewidth=2, label=f'f₀ (aliased): {aliased_f0:.1f} Hz')
    
    title = 'Espectro do Sinal Amostrado e Quantizado (FFT)'
    if aliasing_detected:
        title += ' - Aliasing Detectado!'
        ax_spectrum.set_facecolor('lightcoral')
    
    ax_spectrum.set_xlabel('Frequência (Hz)', fontsize=11)
    ax_spectrum.set_ylabel('Magnitude (dB)', fontsize=11)
    ax_spectrum.set_title(title, fontsize=12)
    ax_spectrum.grid(True, alpha=0.3)
    ax_spectrum.legend(loc='upper right', fontsize=10)
    fig_spectrum.tight_layout()
    plt.close(fig_spectrum)
    fig_spectrum
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Filtro Butterworth Anti-aliasing
    """)
    return


@app.cell
def _(mo):
    filter_order = mo.ui.slider(1, 8, step=1, value=4, label="Ordem:", show_value=True)
    cutoff_freq = mo.ui.text(value="2000", label="Frequência de Corte (Hz):")
    
    mo.md(f"""
    {filter_order}
    
    {cutoff_freq}
    """)
    return cutoff_freq, filter_order


@app.cell
def _(mo, T):
    periods_to_show = mo.ui.slider(1, 10, value=2, label="Períodos para mostrar:", show_value=True)
    
    mo.md(f"""
    {periods_to_show}
    """)
    return periods_to_show,


@app.cell
def _(cutoff_freq, filter_order, np, signal):
    # Design Butterworth filter using analog design
    order = filter_order.value
    fc_str = cutoff_freq.value.strip()
    fc = float(fc_str) if fc_str else 2000
    
    # Use analog design: Wn in rad/s for analog=True
    z, p, k = signal.butter(order, 2*np.pi*fc, 'low', analog=True, output='zpk')
    
    return fc, k, order, p, z


@app.cell
def _(fc, f0, Figure, freqs, k, np, order, p, plt, z):
    # Compute frequency response for analog filter
    w = np.logspace(-1, 5, 4096)  # 0.1 to 100 kHz in rad/s
    # Construct transfer function: H(s) = wc^n / poly, where wc = 2*pi*fc
    wc = 2 * np.pi * fc
    num = np.array([wc**order])
    denom = np.poly(p)
    w_resp, h = freqs(num, denom, w)
    
    # Convert w_resp to Hz for plotting
    freq_resp = w_resp / (2 * np.pi)
    
    # Show from 0 to 4*fc (cutoff at 1/4 of the range)
    freq_max = 4 * fc
    mask = freq_resp <= freq_max
    
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    fig_filter = Figure(figsize=(12, 5))
    ax_filter = fig_filter.add_subplot(1, 1, 1)
    
    # Plot in Hz
    ax_filter.plot(freq_resp[mask], magnitude_db[mask], 'r-', linewidth=2.5, label=f'Butterworth Ordem {order}')
    
    # Mark cutoff frequency (at 1/4 = 25% of range)
    ax_filter.axvline(fc, color='b', linestyle='--', linewidth=1.5, label=f'Cutoff (fc): {fc:.0f} Hz')
    
    # Mark fundamental frequency
    if f0 <= freq_max:
        ax_filter.axvline(f0, color='g', linestyle=':', linewidth=2, label=f'f₀: {f0:.1f} Hz')
    
    # Add -3dB line
    ax_filter.axhline(-3, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='-3 dB')
    
    ax_filter.set_xlim(0, freq_max)
    ax_filter.set_ylim(-80, 5)
    ax_filter.set_xlabel('Frequência (Hz)', fontsize=11)
    ax_filter.set_ylabel('Magnitude (dB)', fontsize=11)
    ax_filter.set_title(f'Resposta do Filtro Butterworth (Ordem {order}) - fc no 1º quarto', fontsize=12)
    ax_filter.grid(True, alpha=0.3)
    ax_filter.legend(loc='upper right', fontsize=10)
    
    fig_filter.tight_layout()
    plt.close(fig_filter)
    fig_filter
    return


@app.cell
def _(Figure, N_pad, aliasing_detected, denom, f0, freqs, fs, np, num, plt, x_q):
    # Compute filtered spectrum
    yf_filt = np.fft.fft(x_q, n=N_pad)
    xf_filt = np.fft.fftfreq(N_pad, 1/fs)
    mag_orig = np.abs(yf_filt)
    mask_pos_filt = xf_filt >= 0
    
    # Filter response at FFT frequencies
    w_fft_filt = 2 * np.pi * xf_filt[mask_pos_filt]
    _, h_filtered_resp = freqs(num, denom, w_fft_filt)
    mag_filtered = mag_orig[mask_pos_filt] * np.abs(h_filtered_resp)
    
    # Find truncation point: last frequency where magnitude > -100 dB
    magnitude_db_all = 20 * np.log10(mag_filtered + 1e-12)
    valid_indices = magnitude_db_all > -100
    if np.any(valid_indices):
        freq_max_filtered = xf_filt[mask_pos_filt][valid_indices][-1]
    else:
        freq_max_filtered = xf_filt[mask_pos_filt][-1]  # fallback
    
    mask_freq = xf_filt[mask_pos_filt] <= freq_max_filtered
    
    magnitude_db_filtered = magnitude_db_all[mask_freq]
    w_resp_filtered = xf_filt[mask_pos_filt][mask_freq]
    
    fig_filtered = Figure(figsize=(12, 5))
    ax_filtered = fig_filtered.add_subplot(1, 1, 1)
    
    # Plot filtered spectrum (styled like the filter plot)
    ax_filtered.plot(w_resp_filtered, magnitude_db_filtered, 'b-', linewidth=2.5, label=f'Espectro Filtrado (Ordem {len(denom)-1})')
    
    # Mark cutoff frequency
    fc_val = (num[0] ** (1 / (len(denom)-1))) / (2 * np.pi)  # Cutoff frequency in Hz
    if fc_val <= freq_max_filtered:
        ax_filtered.axvline(fc_val, color='b', linestyle='--', linewidth=1.5, label=f'Cutoff (fc): {fc_val:.0f} Hz')
    
    # Mark fundamental frequency
    if f0 <= freq_max_filtered:
        ax_filtered.axvline(f0, color='g', linestyle=':', linewidth=2, label=f'f₀: {f0:.1f} Hz')
    
    # Add -3dB line
    ax_filtered.axhline(-3, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='-3 dB')
    
    # Add -100dB truncation line
    ax_filtered.axhline(-100, color='red', linestyle='--', linewidth=1, alpha=0.7, label='-100 dB (truncamento)')
    
    # Fit ylim: bottom -120 dB, top max value +5
    ylim_bottom = -120
    ylim_top = np.max(magnitude_db_filtered) + 5 if len(magnitude_db_filtered) > 0 else 5
    
    ax_filtered.set_xlim(0, freq_max_filtered)
    ax_filtered.set_ylim(ylim_bottom, ylim_top)
    ax_filtered.set_xlabel('Frequência (Hz)', fontsize=11)
    ax_filtered.set_ylabel('Magnitude (dB)', fontsize=11)
    ax_filtered.set_title(f'Espectro Filtrado - truncado em -100 dB', fontsize=12)
    ax_filtered.grid(True, alpha=0.3)
    ax_filtered.legend(loc='upper right', fontsize=10)
    
    fig_filtered.tight_layout()
    plt.close(fig_filtered)
    fig_filtered
    return


@app.cell
def _(Figure, N_pad, T, denom, freqs, fs, np, num, periods_to_show, plt, x_q):
    # Compute filtered time-domain signal
    yf_time = np.fft.fft(x_q, n=N_pad)
    xf_time = np.fft.fftfreq(N_pad, 1/fs)
    mask_pos_time = xf_time >= 0
    
    # Filter response
    w_fft_time = 2 * np.pi * xf_time[mask_pos_time]
    _, h_resp_time = freqs(num, denom, w_fft_time)
    
    # Apply filter in frequency domain
    yf_filtered_time = yf_time.copy()
    yf_filtered_time[mask_pos_time] *= h_resp_time
    # For negative frequencies, conjugate
    yf_filtered_time[~mask_pos_time] *= np.conj(h_resp_time[::-1])
    
    # Inverse FFT to get filtered time signal
    x_filtered = np.fft.ifft(yf_filtered_time).real  # Take real part
    
    # Plot original vs filtered in time domain
    fig_time_domain = Figure(figsize=(12, 5))
    ax_time_domain = fig_time_domain.add_subplot(1, 1, 1)
    
    # Time axis
    t_time = np.arange(len(x_q)) / fs
    
    # Show only up to selected periods
    t_max = periods_to_show.value * T
    mask_show = t_time <= t_max
    
    ax_time_domain.plot(t_time[mask_show], x_q[mask_show], 'b-', linewidth=1.5, label='Sinal Original Quantizado', alpha=0.7)
    ax_time_domain.plot(t_time[mask_show], x_filtered[mask_show], 'r-', linewidth=1.5, label='Sinal Filtrado')
    
    ax_time_domain.set_xlabel('Tempo (s)', fontsize=11)
    ax_time_domain.set_ylabel('Amplitude', fontsize=11)
    ax_time_domain.set_title(f'Sinal no Domínio do Tempo - {periods_to_show.value} Períodos', fontsize=12)
    ax_time_domain.grid(True, alpha=0.3)
    ax_time_domain.legend(loc='upper right', fontsize=10)
    
    fig_time_domain.tight_layout()
    plt.close(fig_time_domain)
    fig_time_domain
    return


if __name__ == "__main__":
    app.run()

