"""
Marimo Notebook: Pure Tone + Independent Butterworth Filter

- Pure sine tone selectable on each piano A (A0..A7)
- Time domain + FFT visualization
- Independent Butterworth low-pass filter designer (order, cutoff, sampling)
- Join cell: apply filter to signal (with optional syncing of sampling rates)
- Resulting filtered signal (time domain) and playback button
"""

import marimo

app = marimo.App()


@app.cell
def _intro(mo):
    intro_text = (
        "# 🛠️ Pure Tone + Butterworth Filter\n\n"
        "This notebook generates a pure sine tone (choose any piano A), shows its waveform and spectrum, "
        "and provides an independent Butterworth low-pass filter you can configure. Use the \"Join\" cell "
        "to apply the filter to the signal (optionally syncing sample rates)."
    )
    mo.md(intro_text)
    return


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scipy import signal

    return mo, np, plt, Figure, signal


@app.cell
def _a_notes(mo):
    # A notes (A0..A7)
    A_notes = {
        "A0 (27.5 Hz)": 27.5,
        "A1 (55.0 Hz)": 55.0,
        "A2 (110.0 Hz)": 110.0,
        "A3 (220.0 Hz)": 220.0,
        "A4 (440.0 Hz)": 440.0,
        "A5 (880.0 Hz)": 880.0,
        "A6 (1760.0 Hz)": 1760.0,
        "A7 (3520.0 Hz)": 3520.0,
    }
    return A_notes


@app.cell
def _signal_controls(mo, A_notes):
    mo.md("## Signal Controls")

    tone_selector = mo.ui.dropdown(list(A_notes.keys()), value="A4 (440.0 Hz)", label="Select A note:")
    fs_sig = mo.ui.dropdown([8000, 16000, 22050, 44100, 48000], value=44100, label="Signal sampling rate (Hz)")
    duration = mo.ui.slider(start=0.1, stop=5.0, value=1.0, step=0.1, label="Duration (s)")
    amplitude = mo.ui.slider(start=0.0, stop=1.0, value=0.9, step=0.01, label="Amplitude")
    noise_level = mo.ui.slider(start=0.0, stop=0.5, value=0.1, step=0.01, label="Noise level")

    play_sig = mo.ui.button(label="▶ Play Signal", on_click=lambda _: None)

    mo.vstack([tone_selector, fs_sig, duration, amplitude, noise_level, mo.hstack([play_sig])])
    return tone_selector, fs_sig, duration, amplitude, noise_level, play_sig


@app.cell
def _gen_signal(mo, tone_selector, fs_sig, duration, amplitude, noise_level, np):
    # Generate pure sine tone
    f0 = float(tone_selector.value.split("(")[1].strip(") Hz")) if "(" in tone_selector.value else float(tone_selector.value)
    fs_signal = int(fs_sig.value)
    _N = int(fs_signal * duration.value)
    t = np.arange(_N) / fs_signal
    
    # Pure tone
    sig_clean = amplitude.value * np.sin(2 * np.pi * f0 * t)
    
    # Add noise
    noise = noise_level.value * np.random.randn(_N)
    sig = sig_clean + noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val * float(amplitude.value)
    
    return sig, sig_clean, t, f0, fs_signal


@app.cell
def _plot_time(mo, Figure, plt, sig, sig_clean, t, f0, fs_signal, np):
    # Time domain plot
    fig_time = Figure(figsize=(10, 3))
    _ax = fig_time.add_subplot(111)
    _ax.plot(t, sig, color='tab:blue', linewidth=0.8, alpha=0.7, label='Signal + Noise')
    _ax.plot(t, sig_clean, color='tab:orange', linewidth=0.6, alpha=0.5, linestyle='--', label='Clean Signal')
    _N_local = len(t)
    step = max(1, _N_local // 400)
    _ax.plot(t[::step], sig[::step], 'o', markersize=1.5, alpha=0.3, color='tab:blue')
    _ax.set_xlabel('Time (s)')
    _ax.set_ylabel('Amplitude')
    _ax.set_title(f'Pure Tone: {f0:.1f} Hz (Time Domain)')
    _ax.grid(True, alpha=0.25)
    _ax.legend(loc='upper right', fontsize=8)
    
    # Add zoom inset showing ~20 periods
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # Calculate time span for 20 periods
        period = 1.0 / f0
        zoom_duration = min(20 * period, t[-1] * 0.3)  # At most 30% of total duration
        zoom_samples = int(zoom_duration * fs_signal)
        zoom_samples = min(zoom_samples, len(t))
        
        axins = inset_axes(_ax, width="35%", height="35%", loc='upper left', bbox_to_anchor=(0.05, 0.05, 0.9, 0.9), bbox_transform=_ax.transAxes)
        axins.plot(t[:zoom_samples], sig[:zoom_samples], color='tab:blue', linewidth=0.8, alpha=0.7)
        axins.plot(t[:zoom_samples], sig_clean[:zoom_samples], color='tab:orange', linewidth=0.6, alpha=0.5, linestyle='--')
        axins.set_title(f'Zoom (~{min(20, int(zoom_duration/period)):.0f} periods)', fontsize=9)
        axins.set_xlabel('Time (s)', fontsize=8)
        axins.set_ylabel('Amplitude', fontsize=8)
        axins.tick_params(axis='both', which='major', labelsize=7)
        axins.grid(True, alpha=0.2)
    except Exception:
        # If inset is unavailable, continue without it
        pass
    
    plt.tight_layout()

    mo.vstack([mo.md('## Signal (Time Domain)'), fig_time])
    return


@app.cell
def _plot_freq(mo, Figure, plt, sig, fs_signal, np):
    # Frequency domain plot
    fft_sig = np.fft.fft(sig)
    freqs = np.fft.fftfreq(len(sig), 1 / fs_signal)
    idx = (freqs >= 0) & (freqs <= min(20000, fs_signal/2))
    mag_db = 10 * np.log10(np.abs(fft_sig) ** 2 + 1e-12)

    fig_f = Figure(figsize=(10, 3))
    _ax = fig_f.add_subplot(111)
    _ax.plot(freqs[idx], mag_db[idx], color='tab:orange', linewidth=1.0)
    _ax.set_xlabel('Frequency (Hz)')
    _ax.set_ylabel('Power (dB)')
    _ax.set_title('Signal Spectrum')
    _ax.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.vstack([mo.md('## Signal (Frequency Domain)'), fig_f])
    return freqs, fft_sig


@app.cell
def _butter_header(mo):
    mo.md('## Butterworth Filter Controls (Independent)')
    return


@app.cell
def _butter_controls(mo, np):
    order = mo.ui.slider(start=1, stop=20, value=4, step=1, label='Filter order (N)')
    cutoff_hz = mo.ui.slider(start=100, stop=10000, value=1000, step=50, label='Cutoff frequency (Hz)')
    fs_filt = mo.ui.dropdown([8000, 16000, 22050, 44100, 48000], value=44100, label='Filter sampling rate (Hz)')
    design_btn = mo.ui.button(label='Design Filter', on_click=lambda _: None)

    mo.vstack([order, cutoff_hz, fs_filt, mo.hstack([design_btn])])
    return order, cutoff_hz, fs_filt, design_btn


@app.cell
def _butter_design(mo, order, cutoff_hz, fs_filt, Figure, plt, signal, np):
    # Design Butterworth filter (digital) using SciPy
    _fs_f = int(fs_filt.value)
    _cutoff_hz = float(cutoff_hz.value)
    _N_design = int(order.value)

    # design in SOS for numerical stability
    try:
        sos = signal.butter(_N_design, _cutoff_hz, btype='low', output='sos', fs=_fs_f)
        _w_design, _h_design = signal.sosfreqz(sos, worN=2048, fs=_fs_f)
        _mag_db_design = 20 * np.log10(np.abs(_h_design) + 1e-12)
    except Exception:
        sos = None
        _w_design = np.array([])
        _mag_db_design = np.array([])

    fig_b = Figure(figsize=(10, 3))
    _ax = fig_b.add_subplot(111)
    if _w_design.size:
        _ax.plot(_w_design, _mag_db_design, color='tab:green', linewidth=1.5)
        # Mark cutoff frequency
        _ax.axvline(_cutoff_hz, color='red', linestyle='--', alpha=0.5, label=f'Cutoff: {_cutoff_hz:.0f} Hz')
        _ax.axhline(-3, color='gray', linestyle=':', alpha=0.5, label='-3 dB')
    _ax.set_xlabel('Frequency (Hz)')
    _ax.set_ylabel('Amplitude (dB)')
    _ax.set_title(f'Butterworth (N={_N_design}) Frequency Response')
    _ax.set_xlim(0, min(10000, _fs_f / 2))
    _ax.set_ylim(-80, 5)
    _ax.legend(loc='upper right', fontsize=8)
    _ax.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.vstack([mo.md('## Butterworth Filter (Design)'), fig_b])
    return sos, _w_design, _mag_db_design


@app.cell
def _join_apply(mo, sig, t, fs_signal, sos, fs_filt, cutoff_hz):
    mo.md('## Join: Apply Filter to Signal')

    # Controls for joining
    sync_sampling = mo.ui.checkbox(value=True, label='Use signal sampling for filtering (sync)')
    apply_btn = mo.ui.button(label='Apply Filter', on_click=lambda _: None)
    sync_btn = mo.ui.button(label='Sync filter -> signal', on_click=lambda _: (_set_filter_to_signal(fs_filt, cutoff_hz, fs_signal)))

    mo.vstack([sync_sampling, mo.hstack([apply_btn, sync_btn])])
    return sync_sampling, apply_btn, sync_btn


# helper used by sync button (must be defined at module scope)
def _set_filter_to_signal(fs_widget, cutoff_widget, signal_fs):
    try:
        fs_widget.value = int(signal_fs)
        # keep same cutoff frequency
        cutoff_widget.value = float(cutoff_widget.value)
    except Exception:
        pass


@app.cell
def _apply_filter(mo, apply_btn, sync_sampling, sig, t, fs_signal, sos, fs_filt, order, cutoff_hz, np, Figure, plt, signal):
    # When Apply is clicked, compute filtered signal according to controls
    # Determine target sampling
    use_sync = bool(sync_sampling.value)
    if use_sync:
        target_fs = int(fs_signal)
    else:
        target_fs = int(fs_filt.value)

    # If filter was designed at different sampling, redesign at target_fs
    _N_target = int(order.value)
    _cutoff_hz_target = float(cutoff_hz.value)
    try:
        sos_target = signal.butter(_N_target, _cutoff_hz_target, btype='low', output='sos', fs=target_fs)
    except Exception:
        sos_target = None

    # Resample signal if needed (simple linear interpolation)
    if target_fs != int(fs_signal):
        # resample to target_fs
        _duration_target = t[-1] + (1.0 / fs_signal)
        t_target = np.arange(int(np.ceil(_duration_target * target_fs))) / target_fs
        sig_target = np.interp(t_target, t, sig)
    else:
        t_target = t
        sig_target = sig

    # Apply filter in time domain (zero-phase)
    if sos_target is not None:
        try:
            filtered = signal.sosfiltfilt(sos_target, sig_target)
        except Exception:
            # fallback to no filtering
            filtered = sig_target.copy()
    else:
        filtered = sig_target.copy()

    # If we resampled earlier, bring back to original sampling for comparison
    if target_fs != int(fs_signal):
        sig_filtered_resampled = np.interp(t, t_target, filtered)
        t_out = t
    else:
        sig_filtered_resampled = filtered
        t_out = t_target

    # Plot result (time)
    fig_r = Figure(figsize=(10, 3))
    _ax = fig_r.add_subplot(111)
    _ax.plot(t_out, sig_filtered_resampled, color='tab:purple', linewidth=0.9)
    _ax.set_title('Filtered Signal (Time Domain)')
    _ax.set_xlabel('Time (s)')
    _ax.set_ylabel('Amplitude')
    _ax.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.vstack([mo.md('## Filtered Signal (Time Domain)'), fig_r])

    # Also show spectrum of filtered signal
    fft_filt = np.fft.fft(sig_filtered_resampled)
    _freqs_filt = np.fft.fftfreq(len(fft_filt), 1 / int(fs_signal))
    _idx_filt = (_freqs_filt >= 0) & (_freqs_filt <= min(20000, int(fs_signal) / 2))
    _mag_db_filt = 10 * np.log10(np.abs(fft_filt) ** 2 + 1e-12)

    fig_rf = Figure(figsize=(10, 3))
    _ax2 = fig_rf.add_subplot(111)
    _ax2.plot(_freqs_filt[_idx_filt], _mag_db_filt[_idx_filt], color='tab:purple')
    _ax2.set_title('Filtered Signal Spectrum')
    _ax2.set_xlabel('Frequency (Hz)')
    _ax2.set_ylabel('Power (dB)')
    _ax2.grid(True, alpha=0.25)
    plt.tight_layout()

    mo.vstack([mo.md('## Filtered Signal (Frequency Domain)'), fig_rf])

    return sig_filtered_resampled, t_out


if __name__ == '__main__':
    app.run()
