"""
Modulação PM (Phase Modulation) - Notebook Interativo

Este notebook explora a modulação de fase (PM) permitindo
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
        # 🎚️ Modulação PM (Phase Modulation)

        ## Introdução

        A **Modulação de Fase (PM)** é uma técnica onde a fase instantânea
        da portadora varia proporcionalmente ao sinal modulante (mensagem).

        ### Fórmula Matemática

        $$s_{PM}(t) = A_c \cos\left(2\pi f_c t + k_p \cdot m(t)\right)$$

        Onde:
        - $f_c$ = frequência da portadora (Hz)
        - $f_m$ = frequência do sinal modulante (Hz)
        - $k_p$ = índice de modulação de fase (radianos por volt)
        - $m(t) = \sin(2\pi f_m t)$ = sinal modulante normalizado

        Para sinal senoidal: $m(t) = \sin(2\pi f_m t)$

        $$s_{PM}(t) = A_c \cos\left(2\pi f_c t + k_p \sin(2\pi f_m t)\right)$$

        ### Relação entre FM e PM

        PM e FM são **duais**:

        - **PM:** fase é **proporcional** ao sinal modulante
          $$\phi(t) = k_p \cdot m(t)$$

        - **FM:** fase é a **integral** do sinal modulante
          $$\phi(t) = 2\pi k_f \int m(t) \, dt$$

        **Consequência:** Um sinal FM pode ser gerado por PM se o sinal modulante
        for primeiro integrado, e vice-versa.

        ### Largura de Banda

        Similar ao FM, usando a Regra de Carson:

        $$B_{PM} \approx 2(k_p + 1)f_m$$

        Para sinal senoidal, onde $\beta_{PM} = k_p$ representa o índice de modulação.
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
    fc_pm_slider = mo.ui.slider(
        start=1000,
        stop=10000,
        step=500,
        value=5000,
        label="Frequência Portadora (fc) [Hz]",
        show_value=True
    )

    # Frequência do sinal modulante (message)
    fm_pm_slider = mo.ui.slider(
        start=100,
        stop=1000,
        step=50,
        value=500,
        label="Frequência Modulante (fm) [Hz]",
        show_value=True
    )

    # Índice de modulação de fase (kp)
    kp_slider = mo.ui.slider(
        start=0.0,
        stop=10.0,
        step=0.1,
        value=2.0,
        label="Índice de Modulação de Fase (kp) [rad]",
        show_value=True
    )

    # Duração da janela de tempo (ms)
    duration_pm_ms = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=5,
        label="Duração da Janela (ms)"
    )

    # Botão de reset
    reset_pm_button = mo.ui.button(
        label="🔄 Reset para Defaults"
    )

    mo.md(
        f"""
        {fc_pm_slider}

        {fm_pm_slider}

        {kp_slider}

        {duration_pm_ms}

        {reset_pm_button}
        """
    )
    return (
        duration_pm_ms,
        fc_pm_slider,
        fm_pm_slider,
        kp_slider,
        reset_pm_button,
    )


@app.cell
def __(
    duration_pm_ms,
    fc_pm_slider,
    fm_pm_slider,
    kp_slider,
    reset_pm_button,
):
    # Processamento dos valores dos controles
    if reset_pm_button.value:
        fc_pm = 5000
        fm_pm = 500
        kp = 2.0
        T_window_pm_ms = 5
    else:
        fc_pm = fc_pm_slider.value
        fm_pm = fm_pm_slider.value
        kp = kp_slider.value
        T_window_pm_ms = duration_pm_ms.value

    # Conversão para segundos
    T_window_pm = T_window_pm_ms * 1e-3

    # Taxa de amostragem
    fs_pm = max(fc_pm * 20, 100000)

    # Índice de modulação (para PM com sinal senoidal)
    beta_pm = kp

    # Cálculo da largura de banda (aproximação usando Carson)
    bandwidth_pm = 2 * (beta_pm + 1) * fm_pm

    # Classificação: Banda Estreita vs Banda Larga
    pm_type = "Banda Estreita (NBPM)" if beta_pm < 0.3 else "Banda Larga (WBPM)"
    return (
        T_window_pm,
        T_window_pm_ms,
        bandwidth_pm,
        beta_pm,
        fc_pm,
        fm_pm,
        fs_pm,
        kp,
        pm_type,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🧮 Geração do Sinal PM""")
    return


@app.cell
def __(T_window_pm, fc_pm, fm_pm, fs_pm, kp, np):
    # Vetor de tempo
    t_pm = np.arange(0, T_window_pm, 1/fs_pm)

    # Sinal modulante (mensagem) - SENO para PM
    m_t_pm = np.sin(2 * np.pi * fm_pm * t_pm)

    # Portadora
    carrier_pm = np.cos(2 * np.pi * fc_pm * t_pm)

    # Fase instantânea do sinal PM
    phase_pm = 2 * np.pi * fc_pm * t_pm + kp * m_t_pm

    # Sinal PM
    s_pm = np.cos(phase_pm)

    # Frequência instantânea para PM
    # f_inst(t) = fc + (kp/(2π)) * d[m(t)]/dt
    # Para m(t) = sin(2πfm*t): d[m(t)]/dt = 2πfm*cos(2πfm*t)
    # Então: f_inst(t) = fc + kp*fm*cos(2πfm*t)
    f_inst_pm = fc_pm + kp * fm_pm * np.cos(2 * np.pi * fm_pm * t_pm)

    # Desvio equivalente de frequência (para comparação com FM)
    delta_f_equiv = kp * fm_pm
    return (
        carrier_pm,
        delta_f_equiv,
        f_inst_pm,
        m_t_pm,
        phase_pm,
        s_pm,
        t_pm,
    )


@app.cell
def __(bandwidth_pm, beta_pm, delta_f_equiv, fm_pm, kp, mo, pm_type):
    mo.md(
        f"""
        ## 📊 Métricas Calculadas

        - **Índice de Modulação (kp):** {kp:.2f} rad
        - **Índice β (para senoidal):** {beta_pm:.2f}
        - **Frequência Modulante (fm):** {fm_pm} Hz
        - **Largura de Banda (estimada):** {bandwidth_pm:.1f} Hz
        - **Tipo:** {pm_type}
        - **Desvio de Freq. Equivalente:** {delta_f_equiv:.1f} Hz

        {mo.callout(
            f"**Banda Estreita:** β < 0.3. Similar ao NBFM.",
            kind="info"
        ) if beta_pm < 0.3 else mo.callout(
            f"**Banda Larga:** β ≥ 0.3. Múltiplas bandas laterais no espectro.",
            kind="success"
        )}

        ### 🔄 Equivalência FM ↔ PM

        Este sinal PM com kp = {kp:.2f} é equivalente a um sinal FM com:
        - **Δf ≈ {delta_f_equiv:.1f} Hz** (desvio de frequência)
        - **β_FM = {beta_pm:.2f}** (índice de modulação FM)

        Isso demonstra a dualidade entre FM e PM!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 📈 Visualização no Domínio do Tempo""")
    return


@app.cell
def __(Figure, carrier_pm, fc_pm, f_inst_pm, m_t_pm, phase_pm, plt, s_pm, t_pm):
    # Gráfico no domínio do tempo
    fig_time_pm = Figure(figsize=(12, 10))
    
    # Sinal modulante
    ax1_pm = fig_time_pm.add_subplot(5, 1, 1)
    ax1_pm.plot(t_pm * 1000, m_t_pm, 'b-', linewidth=1.5, label='Sinal Modulante m(t)')
    ax1_pm.set_ylabel('Amplitude', fontsize=11)
    ax1_pm.set_title('Sinal Modulante (Mensagem) - SENO', fontsize=13, fontweight='bold')
    ax1_pm.grid(True, alpha=0.3)
    ax1_pm.legend(loc='upper right')
    ax1_pm.set_xlim([0, t_pm[-1] * 1000])

    # Portadora
    ax2_pm = fig_time_pm.add_subplot(5, 1, 2)
    ax2_pm.plot(t_pm * 1000, carrier_pm, 'g-', linewidth=0.8, alpha=0.7, label='Portadora')
    ax2_pm.set_ylabel('Amplitude', fontsize=11)
    ax2_pm.set_title('Portadora (Carrier)', fontsize=13, fontweight='bold')
    ax2_pm.grid(True, alpha=0.3)
    ax2_pm.legend(loc='upper right')
    ax2_pm.set_xlim([0, t_pm[-1] * 1000])

    # Sinal PM
    ax3_pm = fig_time_pm.add_subplot(5, 1, 3)
    ax3_pm.plot(t_pm * 1000, s_pm, 'r-', linewidth=1, label='Sinal PM')
    ax3_pm.set_ylabel('Amplitude', fontsize=11)
    ax3_pm.set_title('Sinal Modulado PM', fontsize=13, fontweight='bold')
    ax3_pm.grid(True, alpha=0.3)
    ax3_pm.legend(loc='upper right')
    ax3_pm.set_xlim([0, t_pm[-1] * 1000])

    # Fase instantânea (normalizada para visualização)
    ax4_pm = fig_time_pm.add_subplot(5, 1, 4)
    phase_normalized = (phase_pm % (2 * np.pi))  # Wrap para [0, 2π]
    ax4_pm.plot(t_pm * 1000, phase_normalized, 'c-', linewidth=1.5, label='Fase Instantânea')
    ax4_pm.set_ylabel('Fase (rad)', fontsize=11)
    ax4_pm.set_title('Fase Instantânea do Sinal PM', fontsize=13, fontweight='bold')
    ax4_pm.grid(True, alpha=0.3)
    ax4_pm.legend(loc='upper right')
    ax4_pm.set_xlim([0, t_pm[-1] * 1000])
    ax4_pm.set_ylim([0, 2 * np.pi])

    # Frequência instantânea
    ax5_pm = fig_time_pm.add_subplot(5, 1, 5)
    ax5_pm.plot(t_pm * 1000, f_inst_pm, 'm-', linewidth=1.5, label='Frequência Instantânea')
    ax5_pm.axhline(y=fc_pm, color='k', linestyle='--', alpha=0.5, label=f'fc = {fc_pm} Hz')
    ax5_pm.set_xlabel('Tempo (ms)', fontsize=11)
    ax5_pm.set_ylabel('Frequência (Hz)', fontsize=11)
    ax5_pm.set_title('Frequência Instantânea do Sinal PM', fontsize=13, fontweight='bold')
    ax5_pm.grid(True, alpha=0.3)
    ax5_pm.legend(loc='upper right')
    ax5_pm.set_xlim([0, t_pm[-1] * 1000])

    fig_time_pm.tight_layout()
    plt.close(fig_time_pm)
    fig_time_pm
    return (
        ax1_pm,
        ax2_pm,
        ax3_pm,
        ax4_pm,
        ax5_pm,
        fig_time_pm,
        phase_normalized,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🌊 Visualização no Domínio da Frequência (FFT)""")
    return


@app.cell
def __(fs_pm, np, s_pm):
    # Cálculo da FFT
    N_pm = len(s_pm)
    fft_pm = np.fft.fft(s_pm)
    fft_pm_magnitude = np.abs(fft_pm) / N_pm
    fft_pm_magnitude = fft_pm_magnitude[:N_pm//2]  # Apenas frequências positivas
    freqs_pm = np.fft.fftfreq(N_pm, 1/fs_pm)
    freqs_pm = freqs_pm[:N_pm//2]
    return N_pm, fft_pm, fft_pm_magnitude, freqs_pm


@app.cell
def __(beta_pm, fc_pm, fm_pm, mo):
    mo.md(
        f"""
        ### Análise Espectral

        **Similaridade com FM:**

        O espectro PM é matematicamente **idêntico** ao espectro FM para um sinal
        modulante senoidal, com a mesma estrutura de bandas laterais de Bessel.

        - **Portadora:** {fc_pm} Hz
        - **Bandas Laterais:** {fc_pm} ± n×{fm_pm} Hz para n = 1, 2, 3, ...
        - **Número de bandas significativas:** ~{int(beta_pm + 1)}

        **Diferença chave PM vs FM:**
        - **PM:** A fase varia com m(t) diretamente
        - **FM:** A fase varia com a integral de m(t)
        - Para **sinais senoidais**, os espectros são idênticos!
        - Para **outros sinais**, os espectros diferem
        """
    )
    return


@app.cell
def __(Figure, fc_pm, fft_pm_magnitude, freqs_pm, plt):
    # Gráfico no domínio da frequência
    fig_freq_pm = Figure(figsize=(12, 6))
    ax_freq_pm = fig_freq_pm.add_subplot(1, 1, 1)

    ax_freq_pm.plot(freqs_pm, fft_pm_magnitude, 'b-', linewidth=1, label='Espectro PM')
    ax_freq_pm.axvline(x=fc_pm, color='r', linestyle='--', alpha=0.5, linewidth=2, label=f'Portadora ({fc_pm} Hz)')

    ax_freq_pm.set_xlabel('Frequência (Hz)', fontsize=12)
    ax_freq_pm.set_ylabel('Magnitude Normalizada', fontsize=12)
    ax_freq_pm.set_title('Espectro de Frequência do Sinal PM', fontsize=14, fontweight='bold')
    ax_freq_pm.grid(True, alpha=0.3)
    ax_freq_pm.legend(loc='upper right')
    ax_freq_pm.set_xlim([0, fc_pm * 1.5])

    fig_freq_pm.tight_layout()
    plt.close(fig_freq_pm)
    fig_freq_pm
    return ax_freq_pm, fig_freq_pm


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📚 Notas Pedagógicas

        ### O que observar:

        1. **Amplitude Constante**: Como no FM, o sinal PM mantém amplitude constante.
           A informação está codificada na variação da fase.

        2. **Fase Instantânea**: Varia linearmente com o sinal modulante m(t).
           Para m(t) = sin(2πfmt), a fase oscila senoidalmente.

        3. **Frequência Instantânea**: 
           - f_inst(t) = fc + (kp·fm)·cos(2πfm·t)
           - Note que é a **derivada** da fase
           - Varia de forma cosenoidal (90° defasado do sinal modulante!)

        4. **Espectro PM**: 
           - Para sinal senoidal: idêntico ao FM
           - Estrutura de Bessel com múltiplas bandas laterais
           - Largura de banda aumenta com kp

        5. **Dualidade FM-PM**:
           - PM com m(t) = FM com ∫m(t)dt
           - FM com m(t) = PM com dm(t)/dt
           - Sistemas podem usar PM mas emular FM (e vice-versa)

        ### Diferença Visual: PM vs FM

        Compare os gráficos de **Frequência Instantânea**:
        - **PM com sin(2πfmt):** f_inst varia como **cosseno** (derivada do seno)
        - **FM com sin(2πfmt):** f_inst varia como **cosseno** também

        Para sinais senoidais, **PM e FM são equivalentes**! A diferença aparece
        com sinais não-senoidais.

        ### Experimentos Sugeridos:

        - Varie **kp** e observe o número de bandas laterais no espectro
        - Compare com o notebook FM: mesmo kp e Δf = kp·fm dão espectros similares
        - Note que a **fase** em PM segue diretamente m(t)
        - Observe a defasagem de 90° entre m(t) e f_inst(t)

        ### Aplicações de PM

        - **Sistemas digitais:** PSK (Phase Shift Keying) é uma forma de PM digital
        - **Sincronização:** PM é usada em PLLs (Phase-Locked Loops)
        - **Comunicação por satélite:** QPSK, 8PSK são variantes de PM
        - **Análise:** PM é matematicamente mais simples que FM para certos cálculos

        ---

        **Referências:**
        - Haykin, S. "Communication Systems" (5th Ed.) - Capítulo 3
        - Proakis, J. & Salehi, M. "Communication Systems Engineering"
        - Lathi, B. P. "Modern Digital and Analog Communication Systems"
        """
    )
    return


@app.cell
def __():
    # Comparação PM vs FM para referência
    comparison_table = """
    | Característica | PM | FM |
    |----------------|----|----|
    | Parâmetro modulado | Fase φ(t) | Frequência f(t) |
    | Relação matemática | φ(t) ∝ m(t) | f(t) ∝ m(t) |
    | Fase instantânea | φ(t) = 2πfct + kp·m(t) | φ(t) = 2πfct + 2πkf∫m(t)dt |
    | Freq. instantânea | f(t) = fc + (kp/2π)·dm(t)/dt | f(t) = fc + kf·m(t) |
    | Para m(t) senoidal | Espectros idênticos | Espectros idênticos |
    | Implementação | Varactor na referência | VCO direto |
    | Sensibilidade a ruído | Derivada amplifica ruído HF | Melhor SNR |
    """
    return


@app.cell
def __():
    # Informações sobre o notebook
    __notebook_info_pm__ = {
        "title": "Modulação PM Interativa",
        "version": "1.0",
        "date": "2025-10-30",
        "python": "3.14+",
        "dependencies": ["marimo", "numpy>=2.0", "matplotlib>=3.9"]
    }
    return


if __name__ == "__main__":
    app.run()
