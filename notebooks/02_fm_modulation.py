"""
Modulação FM (Frequency Modulation) - Notebook Interativo

Este notebook explora a modulação de frequência (FM) permitindo
a manipulação interativa de parâmetros fundamentais.

Autor: Fundamentos da Comunicação
Data: 2025-10-30
Python: 3.14+
"""

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    return Figure, mo, np, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        # 📻 Modulação FM (Frequency Modulation)

        ## Introdução

        A **Modulação de Frequência (FM)** é uma técnica onde a frequência instantânea
        da portadora varia proporcionalmente ao sinal modulante (mensagem).

        ### Fórmula Matemática

        $$s_{FM}(t) = A_c \cos\left(2\pi f_c t + \beta \sin(2\pi f_m t)\right)$$

        Onde:
        - $f_c$ = frequência da portadora (Hz)
        - $f_m$ = frequência do sinal modulante (Hz)
        - $\beta = \frac{\Delta f}{f_m}$ = índice de modulação
        - $\Delta f$ = desvio máximo de frequência (Hz)

        ### Regra de Carson (Largura de Banda)

        $$B_{FM} \approx 2(\Delta f + f_m) = 2f_m(\beta + 1)$$

        A Regra de Carson fornece uma estimativa da largura de banda que contém
        aproximadamente 98% da potência do sinal FM.

        ### Índice de Modulação β

        - **Banda Estreita (NBFM):** $\beta < 0.3$ → $B \approx 2f_m$
        - **Banda Larga (WBFM):** $\beta > 1$ → $B \approx 2\Delta f$
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Controles Interativos""")
    return


@app.cell
def __(mo):
    # Frequência da portadora (carrier)
    fc_fm_slider = mo.ui.slider(
        start=1000,
        stop=10000,
        step=500,
        value=5000,
        label="Frequência Portadora (fc) [Hz]",
        show_value=True
    )

    # Frequência do sinal modulante (message)
    fm_fm_slider = mo.ui.slider(
        start=100,
        stop=1000,
        step=50,
        value=500,
        label="Frequência Modulante (fm) [Hz]",
        show_value=True
    )

    # Desvio de frequência (frequency deviation)
    delta_f_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
        label="Desvio de Frequência (Δf) [Hz]",
        show_value=True
    )

    # Duração da janela de tempo (ms)
    duration_fm_ms = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=5,
        label="Duração da Janela (ms)"
    )

    # Botão de reset
    reset_fm_button = mo.ui.button(
        label="🔄 Reset para Defaults"
    )

    mo.md(
        f"""
        {fc_fm_slider}

        {fm_fm_slider}

        {delta_f_slider}

        {duration_fm_ms}

        {reset_fm_button}
        """
    )
    return (
        delta_f_slider,
        duration_fm_ms,
        fc_fm_slider,
        fm_fm_slider,
        reset_fm_button,
    )


@app.cell
def __(
    delta_f_slider,
    duration_fm_ms,
    fc_fm_slider,
    fm_fm_slider,
    reset_fm_button,
):
    # Processamento dos valores dos controles
    if reset_fm_button.value:
        fc_fm = 5000
        fm_fm = 500
        delta_f = 1000
        T_window_fm_ms = 5
    else:
        fc_fm = fc_fm_slider.value
        fm_fm = fm_fm_slider.value
        delta_f = delta_f_slider.value
        T_window_fm_ms = duration_fm_ms.value

    # Conversão para segundos
    T_window_fm = T_window_fm_ms * 1e-3

    # Taxa de amostragem
    fs_fm = max(fc_fm * 20, 100000)

    # Cálculo do índice de modulação β
    beta = delta_f / fm_fm

    # Cálculo da largura de banda (Regra de Carson)
    bandwidth_carson = 2 * (delta_f + fm_fm)

    # Classificação: Banda Estreita vs Banda Larga
    fm_type = "Banda Estreita (NBFM)" if beta < 0.3 else "Banda Larga (WBFM)"
    return (
        T_window_fm,
        T_window_fm_ms,
        bandwidth_carson,
        beta,
        delta_f,
        fc_fm,
        fm_fm,
        fm_type,
        fs_fm,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🧮 Geração do Sinal FM""")
    return


@app.cell
def __(T_window_fm, beta, fc_fm, fm_fm, fs_fm, np):
    # Vetor de tempo
    t_fm = np.arange(0, T_window_fm, 1/fs_fm)

    # Sinal modulante (mensagem) normalizado
    m_t_fm = np.cos(2 * np.pi * fm_fm * t_fm)

    # Portadora
    carrier_fm = np.cos(2 * np.pi * fc_fm * t_fm)

    # Fase instantânea do sinal FM
    phase_fm = 2 * np.pi * fc_fm * t_fm + beta * np.sin(2 * np.pi * fm_fm * t_fm)

    # Sinal FM
    s_fm = np.cos(phase_fm)

    # Frequência instantânea
    # f_inst(t) = fc + Δf * cos(2πfm*t)
    f_inst = fc_fm + delta_f * np.cos(2 * np.pi * fm_fm * t_fm)
    return carrier_fm, f_inst, m_t_fm, phase_fm, s_fm, t_fm


@app.cell
def __(bandwidth_carson, beta, delta_f, fm_fm, fm_type, mo):
    mo.md(
        f"""
        ## 📊 Métricas Calculadas

        - **Índice de Modulação (β):** {beta:.3f}
        - **Desvio de Frequência (Δf):** {delta_f} Hz
        - **Frequência Modulante (fm):** {fm_fm} Hz
        - **Largura de Banda (Carson):** {bandwidth_carson:.1f} Hz
        - **Tipo:** {fm_type}

        {mo.callout(
            f"**Banda Estreita:** β < 0.3. O espectro é similar ao AM com largura ~2fm.",
            kind="info"
        ) if beta < 0.3 else mo.callout(
            f"**Banda Larga:** β ≥ 0.3. Múltiplas bandas laterais aparecem no espectro.",
            kind="success"
        )}
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 📈 Visualização no Domínio do Tempo""")
    return


@app.cell
def __(Figure, carrier_fm, f_inst, fc_fm, m_t_fm, plt, s_fm, t_fm):
    # Gráfico no domínio do tempo
    fig_time_fm = Figure(figsize=(12, 9))
    
    # Sinal modulante
    ax1_fm = fig_time_fm.add_subplot(4, 1, 1)
    ax1_fm.plot(t_fm * 1000, m_t_fm, 'b-', linewidth=1.5, label='Sinal Modulante m(t)')
    ax1_fm.set_ylabel('Amplitude', fontsize=11)
    ax1_fm.set_title('Sinal Modulante (Mensagem)', fontsize=13, fontweight='bold')
    ax1_fm.grid(True, alpha=0.3)
    ax1_fm.legend(loc='upper right')
    ax1_fm.set_xlim([0, t_fm[-1] * 1000])

    # Portadora
    ax2_fm = fig_time_fm.add_subplot(4, 1, 2)
    ax2_fm.plot(t_fm * 1000, carrier_fm, 'g-', linewidth=0.8, alpha=0.7, label='Portadora')
    ax2_fm.set_ylabel('Amplitude', fontsize=11)
    ax2_fm.set_title('Portadora (Carrier)', fontsize=13, fontweight='bold')
    ax2_fm.grid(True, alpha=0.3)
    ax2_fm.legend(loc='upper right')
    ax2_fm.set_xlim([0, t_fm[-1] * 1000])

    # Sinal FM
    ax3_fm = fig_time_fm.add_subplot(4, 1, 3)
    ax3_fm.plot(t_fm * 1000, s_fm, 'r-', linewidth=1, label='Sinal FM')
    ax3_fm.set_ylabel('Amplitude', fontsize=11)
    ax3_fm.set_title('Sinal Modulado FM', fontsize=13, fontweight='bold')
    ax3_fm.grid(True, alpha=0.3)
    ax3_fm.legend(loc='upper right')
    ax3_fm.set_xlim([0, t_fm[-1] * 1000])

    # Frequência instantânea
    ax4_fm = fig_time_fm.add_subplot(4, 1, 4)
    ax4_fm.plot(t_fm * 1000, f_inst, 'm-', linewidth=1.5, label='Frequência Instantânea')
    ax4_fm.axhline(y=fc_fm, color='k', linestyle='--', alpha=0.5, label=f'fc = {fc_fm} Hz')
    ax4_fm.set_xlabel('Tempo (ms)', fontsize=11)
    ax4_fm.set_ylabel('Frequência (Hz)', fontsize=11)
    ax4_fm.set_title('Frequência Instantânea do Sinal FM', fontsize=13, fontweight='bold')
    ax4_fm.grid(True, alpha=0.3)
    ax4_fm.legend(loc='upper right')
    ax4_fm.set_xlim([0, t_fm[-1] * 1000])

    fig_time_fm.tight_layout()
    plt.close(fig_time_fm)
    fig_time_fm
    return ax1_fm, ax2_fm, ax3_fm, ax4_fm, fig_time_fm


@app.cell
def __(mo):
    mo.md(r"""## 🌊 Visualização no Domínio da Frequência (FFT)""")
    return


@app.cell
def __(fs_fm, np, s_fm):
    # Cálculo da FFT
    N_fm = len(s_fm)
    fft_fm = np.fft.fft(s_fm)
    fft_fm_magnitude = np.abs(fft_fm) / N_fm
    fft_fm_magnitude = fft_fm_magnitude[:N_fm//2]  # Apenas frequências positivas
    freqs_fm = np.fft.fftfreq(N_fm, 1/fs_fm)
    freqs_fm = freqs_fm[:N_fm//2]
    return N_fm, fft_fm, fft_fm_magnitude, freqs_fm


@app.cell
def __(beta, delta_f, fc_fm, fm_fm, mo):
    mo.md(
        f"""
        ### Análise Espectral

        **Teoria de Bessel:**

        O espectro FM é composto por infinitas bandas laterais, cuja amplitude é
        determinada pelas **Funções de Bessel** de primeira espécie $J_n(\beta)$.

        - **Portadora:** {fc_fm} Hz (amplitude proporcional a $J_0(\beta)$)
        - **Bandas Laterais:** {fc_fm} ± n×{fm_fm} Hz para n = 1, 2, 3, ...

        Para β = {beta:.3f}, esperamos aproximadamente **{int(beta + 1)}** pares
        de bandas laterais significativas.

        **Observação:** A largura de banda cresce com Δf, não com a potência do sinal!
        """
    )
    return


@app.cell
def __(Figure, fc_fm, fft_fm_magnitude, freqs_fm, plt):
    # Gráfico no domínio da frequência
    fig_freq_fm = Figure(figsize=(12, 6))
    ax_freq_fm = fig_freq_fm.add_subplot(1, 1, 1)

    ax_freq_fm.plot(freqs_fm, fft_fm_magnitude, 'b-', linewidth=1, label='Espectro FM')
    ax_freq_fm.axvline(x=fc_fm, color='r', linestyle='--', alpha=0.5, linewidth=2, label=f'Portadora ({fc_fm} Hz)')

    ax_freq_fm.set_xlabel('Frequência (Hz)', fontsize=12)
    ax_freq_fm.set_ylabel('Magnitude Normalizada', fontsize=12)
    ax_freq_fm.set_title('Espectro de Frequência do Sinal FM', fontsize=14, fontweight='bold')
    ax_freq_fm.grid(True, alpha=0.3)
    ax_freq_fm.legend(loc='upper right')
    ax_freq_fm.set_xlim([0, fc_fm * 1.5])

    fig_freq_fm.tight_layout()
    plt.close(fig_freq_fm)
    fig_freq_fm
    return ax_freq_fm, fig_freq_fm


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📚 Notas Pedagógicas

        ### O que observar:

        1. **Amplitude Constante**: Diferentemente do AM, o sinal FM mantém amplitude
           constante. A informação está codificada na variação da frequência.

        2. **Frequência Instantânea**: O gráfico mostra como a frequência da portadora
           varia em torno de fc, com excursão máxima de ±Δf.

        3. **Espectro FM (Bandas Laterais)**: 
           - Para β < 0.3 (NBFM): Espectro similar ao AM, 2 bandas laterais principais
           - Para β > 1 (WBFM): Múltiplas bandas laterais aparecem, espaçadas por fm

        4. **Regra de Carson**: Fornece uma estimativa prática. A banda infinita teórica
           é truncada em ~98% da potência.

        5. **Vantagens do FM**:
           - Imune a variações de amplitude (ruído, interferência)
           - Melhor SNR com aumento de banda (troca banda por SNR)
           - Usado em FM broadcast, comunicações móveis

        ### Experimentos Sugeridos:

        - Varie **Δf** e observe como a largura de banda aumenta
        - Compare **β < 0.3** (NBFM) vs **β > 1** (WBFM) no espectro
        - Mantenha **Δf** fixo e varie **fm** para ver o efeito em β
        - Note que a amplitude do sinal FM é sempre constante

        ### Relação FM vs PM

        FM e PM são intimamente relacionados:
        - **FM:** fase é a integral do sinal modulante
        - **PM:** fase é proporcional ao sinal modulante
        
        Para sinais de banda estreita, são praticamente indistinguíveis.

        ---

        **Referências:**
        - Haykin, S. "Communication Systems" (5th Ed.) - Capítulo 3
        - Regra de Carson: Carson, J.R. (1922) "Notes on the theory of modulation"
        - Funções de Bessel: Abramowitz & Stegun
        """
    )
    return


@app.cell
def __():
    # Informações sobre o notebook
    __notebook_info_fm__ = {
        "title": "Modulação FM Interativa",
        "version": "1.0",
        "date": "2025-10-30",
        "python": "3.14+",
        "dependencies": ["marimo", "numpy>=2.0", "matplotlib>=3.9"]
    }
    return


if __name__ == "__main__":
    app.run()
