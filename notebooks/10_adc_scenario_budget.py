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
    return Figure, mo, np, plt, signal


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
    
    # Generate one period with plenty of samples
    fs_display = 44100
    samples = int(fs_display * T)
    t = np.linspace(0, T, samples, endpoint=False)
    x = A * np.sin(2 * np.pi * f0 * t)
    
    return A, T, f0, samples, t, x


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
def _(Figure, plt, t, x):
    fig_time = Figure(figsize=(12, 4))
    ax_time = fig_time.add_subplot(1, 1, 1)
    ax_time.plot(t * 1e3, x, 'b-', linewidth=2)
    ax_time.set_xlabel('Tempo (ms)', fontsize=11)
    ax_time.set_ylabel('Amplitude', fontsize=11)
    ax_time.set_title('Sinal: Um Período', fontsize=12)
    ax_time.grid(True, alpha=0.3)
    fig_time.tight_layout()
    plt.close(fig_time)
    fig_time
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
    cutoff_freq = mo.ui.slider(100, 20000, step=100, value=2000, label="Frequência de Corte (Hz):", show_value=True)
    
    mo.md(f"""
    {filter_order}
    
    {cutoff_freq}
    """)
    return cutoff_freq, filter_order


@app.cell
def _(cutoff_freq, filter_order, signal):
    # Design Butterworth filter using analog design
    order = filter_order.value
    fc = cutoff_freq.value
    
    # Use analog design to avoid Nyquist limitation
    z, p, k = signal.butter(order, fc, 'low', analog=True, output='zpk')
    
    return fc, k, order, p, z


@app.cell
def _(fc, f0, Figure, k, np, order, p, plt, z):
    # Compute frequency response for analog filter
    w = np.logspace(-1, 5, 4096)  # 0.1 Hz to 100 kHz
    from scipy.signal import freqs
    # Construct transfer function: H(s) = wc^n / (s + p[0])(s + p[1])...(s + p[n-1])
    num = np.array([fc**order])
    denom = np.poly(p)
    w_resp, h = freqs(num, denom, w)
    
    # Show from 0 to 4*fc (cutoff at 1/4 of the range)
    # fc is at 25%, remaining 75% shows transition and rejection
    freq_max = 4 * fc
    mask = w_resp <= freq_max
    
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    
    fig_filter = Figure(figsize=(12, 5))
    ax_filter = fig_filter.add_subplot(1, 1, 1)
    
    # Use linear scale to show the proportional spacing
    ax_filter.plot(w_resp[mask], magnitude_db[mask], 'r-', linewidth=2.5, label=f'Butterworth Ordem {order}')
    
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


if __name__ == "__main__":
    app.run()

