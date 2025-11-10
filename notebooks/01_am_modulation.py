"""
Modulação AM (Amplitude Modulation) - Notebook Interativo

Este notebook explora a modulação de amplitude (AM DSB-TC) permitindo
a manipulação interativa de parâmetros fundamentais.

Autor: Fundamentos da Comunicação
Data: 2025-10-30
Python: 3.14+
"""

import marimo

__generated_with = "0.17.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    return Figure, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 📡 Modulação AM (Amplitude Modulation)

    ## Introdução

    A **Modulação de Amplitude (AM)** é uma técnica onde a amplitude da portadora
    varia proporcionalmente ao sinal modulante (mensagem).

    ### Fórmula Matemática

    $$s_{AM}(t) = [1 + k_a \cdot m(t)] \cos(2\pi f_c t)$$

    Onde:
    - $f_c$ = frequência da portadora (Hz)
    - $f_m$ = frequência do sinal modulante (Hz)
    - $k_a$ = índice de modulação (0 a 1 para evitar sobremodulação)
    - $m(t) = \cos(2\pi f_m t)$ = sinal modulante normalizado

    ### Largura de Banda

    $$B_{AM} = 2f_m$$

    A largura de banda ocupada é o dobro da frequência máxima do sinal modulante.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🎛️ Controles Interativos
    """)
    return


@app.cell
def _(mo):
    # Frequência da portadora (carrier)
    fc_slider = mo.ui.slider(
        start=1000,
        stop=10000,
        step=500,
        value=5000,
        label="Frequência Portadora (fc) [Hz]",
        show_value=True
    )

    # Frequência do sinal modulante (message)
    fm_slider = mo.ui.slider(
        start=100,
        stop=1000,
        step=50,
        value=500,
        label="Frequência Modulante (fm) [Hz]",
        show_value=True
    )

    # Índice de modulação
    ka_slider = mo.ui.slider(
        start=0.0,
        stop=1.5,
        step=0.05,
        value=0.8,
        label="Índice de Modulação (ka)",
        show_value=True
    )

    # Duração da janela de tempo (ms)
    duration_ms = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=5,
        label="Duração da Janela (ms)"
    )

    # Botão de reset
    reset_button = mo.ui.button(
        label="🔄 Reset para Defaults"
    )

    mo.md(
        f"""
        {fc_slider}

        {fm_slider}

        {ka_slider}

        {duration_ms}

        {reset_button}
        """
    )
    return duration_ms, fc_slider, fm_slider, ka_slider, reset_button


@app.cell
def _(duration_ms, fc_slider, fm_slider, ka_slider, reset_button):
    # Processamento dos valores dos controles
    # Reset redefine para valores padrão
    if reset_button.value:
        fc = 5000
        fm = 500
        ka = 0.8
        T_window_ms = 5
    else:
        fc = fc_slider.value
        fm = fm_slider.value
        ka = ka_slider.value
        T_window_ms = duration_ms.value

    # Conversão para segundos
    T_window = T_window_ms * 1e-3

    # Taxa de amostragem (pelo menos 10x a frequência da portadora)
    fs = max(fc * 20, 100000)
    return T_window, fc, fm, fs, ka


@app.cell
def _(mo):
    mo.md(r"""
    ## 🧮 Geração do Sinal AM
    """)
    return


@app.cell
def _(T_window, fc, fm, fs, ka, np):
    # Vetor de tempo
    t_am = np.arange(0, T_window, 1/fs)

    # Sinal modulante (mensagem) normalizado
    m_t = np.cos(2 * np.pi * fm * t_am)

    # Portadora
    carrier = np.cos(2 * np.pi * fc * t_am)

    # Sinal AM
    s_am = (1 + ka * m_t) * carrier

    # Cálculo da largura de banda
    bandwidth_am = 2 * fm

    # Detecção de sobremodulação
    max_modulation = np.max(np.abs(ka * m_t))
    is_overmodulated = max_modulation > 1.0
    return (
        bandwidth_am,
        carrier,
        is_overmodulated,
        m_t,
        max_modulation,
        s_am,
        t_am,
    )


@app.cell
def _(is_overmodulated, ka, max_modulation, mo):
    mo.md(f"""
    ## 📊 Métricas Calculadas

    - **Índice de Modulação (ka):** {ka:.2f}
    - **Modulação Máxima:** {max_modulation:.2f}
    - **Status:** {"⚠️ **SOBREMODULAÇÃO DETECTADA!**" if is_overmodulated else "✅ Modulação Normal"}

    {mo.callout(
        "**Atenção:** Sobremodulação causa distorção! Reduza o valor de ka.",
        kind="warn"
    ) if is_overmodulated else "\\"}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📈 Visualização no Domínio do Tempo
    """)
    return


@app.cell
def _(Figure, carrier, m_t, plt, s_am, t_am, ka, np):
    # Gráfico no domínio do tempo
    fig_time_am = Figure(figsize=(12, 8))

    # Sinal modulante
    ax1 = fig_time_am.add_subplot(3, 1, 1)
    ax1.plot(t_am * 1000, m_t, 'b-', linewidth=1.5, label='Sinal Modulante m(t)')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Sinal Modulante (Mensagem)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, t_am[-1] * 1000])

    # Portadora
    ax2 = fig_time_am.add_subplot(3, 1, 2)
    ax2.plot(t_am * 1000, carrier, 'g-', linewidth=0.8, alpha=0.7, label='Portadora')
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Portadora (Carrier)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, t_am[-1] * 1000])

    # Sinal AM
    ax3 = fig_time_am.add_subplot(3, 1, 3)
    ax3.plot(t_am * 1000, s_am, 'r-', linewidth=1, label='Sinal AM')
    # Envelope (use analytic signal via Hilbert when disponível; fallback para teoria)
    try:
        from scipy.signal import hilbert

        analytic = hilbert(s_am)
        envelope = np.abs(analytic)
    except Exception:
        # Fallback: usar a envoltória teórica |1 + ka * m(t)| quando scipy não estiver disponível
        envelope = np.abs(1 + ka * m_t)

    envelope_upper = envelope
    envelope_lower = -envelope
    ax3.plot(t_am * 1000, envelope_upper, 'k--', linewidth=1.5, alpha=0.8, label='Envelope')
    ax3.plot(t_am * 1000, envelope_lower, 'k--', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Tempo (ms)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_title('Sinal Modulado AM', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    ax3.set_xlim([0, t_am[-1] * 1000])

    fig_time_am.tight_layout()
    plt.close(fig_time_am)
    fig_time_am
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 🌊 Visualização no Domínio da Frequência (FFT)
    """)
    return


@app.cell
def _(fc, fm, fs, np, s_am):
    # Cálculo da FFT
    N_am = len(s_am)
    fft_am = np.fft.fft(s_am)
    fft_am_magnitude = np.abs(fft_am) / N_am
    fft_am_magnitude = fft_am_magnitude[:N_am//2]  # Apenas frequências positivas
    freqs_am = np.fft.fftfreq(N_am, 1/fs)
    freqs_am = freqs_am[:N_am//2]

    # Identificação dos picos esperados
    expected_peaks = [fc - fm, fc, fc + fm]
    return expected_peaks, fft_am_magnitude, freqs_am


@app.cell
def _(bandwidth_am, fc, fm, mo):
    mo.md(f"""
    ### Análise Espectral

    **Componentes de Frequência Esperadas:**
    - **Portadora:** {fc} Hz
    - **Banda Lateral Inferior (LSB):** {fc - fm} Hz
    - **Banda Lateral Superior (USB):** {fc + fm} Hz
    - **Largura de Banda Total:** {bandwidth_am} Hz
    """)
    return


@app.cell
def _(Figure, expected_peaks, fc, fft_am_magnitude, freqs_am, plt):
    # Gráfico no domínio da frequência
    fig_freq_am = Figure(figsize=(12, 6))
    ax_freq = fig_freq_am.add_subplot(1, 1, 1)

    ax_freq.plot(freqs_am, fft_am_magnitude, 'b-', linewidth=1, label='Espectro AM')

    # Destacar picos esperados
    for peak_freq in expected_peaks:
        ax_freq.axvline(x=peak_freq, color='r', linestyle='--', alpha=0.5, linewidth=1.5)

    ax_freq.set_xlabel('Frequência (Hz)', fontsize=12)
    ax_freq.set_ylabel('Magnitude Normalizada', fontsize=12)
    ax_freq.set_title('Espectro de Frequência do Sinal AM', fontsize=14, fontweight='bold')
    ax_freq.grid(True, alpha=0.3)
    ax_freq.legend(loc='upper right')
    ax_freq.set_xlim([0, fc * 1.5])

    fig_freq_am.tight_layout()
    plt.close(fig_freq_am)
    fig_freq_am
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 📚 Notas Pedagógicas

    ### O que observar:

    1. **Envelope do Sinal AM**: A envoltória do sinal modulado segue o formato
       do sinal modulante. Quando ka = 1, o envelope toca zero nos pontos mínimos.

    2. **Sobremodulação (ka > 1)**: Causa distorção e inversão de fase,
       resultando em demodulação incorreta. Visualmente, o envelope cruza o eixo zero.

    3. **Espectro AM**: Mostra três componentes principais:
       - Portadora em fc
       - Banda lateral inferior (LSB) em fc - fm
       - Banda lateral superior (USB) em fc + fm

    4. **Eficiência**: AM convencional (DSB-TC) desperdiça energia na portadora,
       que não carrega informação. Apenas 33% da potência (no máximo) carrega informação.

    ### Experimentos Sugeridos:

    - Varie **ka** e observe o envelope e o espectro
    - Compare diferentes valores de **fm** e veja como a largura de banda muda
    - Tente **ka > 1** para ver o efeito da sobremodulação

    ---

    **Referências:**
    - Haykin, S. "Communication Systems" (5th Ed.)
    - Proakis, J. & Salehi, M. "Communication Systems Engineering"
    """)
    return


@app.cell
def _():
    # Informações sobre o notebook
    __notebook_info__ = {
        "title": "Modulação AM Interativa",
        "version": "1.0",
        "date": "2025-10-30",
        "python": "3.14+",
        "dependencies": ["marimo", "numpy>=2.0", "matplotlib>=3.9"]
    }
    return


if __name__ == "__main__":
    app.run()
