"""
Amostragem, Teorema de Nyquist e Aliasing - Notebook Interativo

Este notebook explora o teorema de Nyquist-Shannon e o fenômeno de aliasing,
permitindo visualização interativa dos efeitos da taxa de amostragem.

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
        # 🎯 Amostragem, Teorema de Nyquist e Aliasing

        ## Introdução

        O **Teorema de Nyquist-Shannon** é fundamental para a conversão entre
        sinais analógicos e digitais, estabelecendo a taxa mínima de amostragem
        necessária para representação perfeita de um sinal.

        ### Teorema de Nyquist-Shannon

        Para um sinal banda-limitada com frequência máxima $f_{max}$:

        $$f_s > 2 \cdot f_{max} = f_{Nyquist}$$

        - $f_s$ = taxa de amostragem (samples/segundo)
        - $f_{Nyquist} = 2 \cdot f_{max}$ = taxa de Nyquist (mínima teórica)
        - $f_{max}$ = frequência máxima no sinal (largura de banda)

        ### Aliasing

        Quando $f_s \leq f_{Nyquist}$, ocorre **aliasing**: frequências acima de
        $f_s/2$ "se dobram" e aparecem como frequências mais baixas no sinal amostrado.

        A frequência aparente (alias) é calculada por:

        $$f_{alias} = \left| f_{sinal} - k \cdot f_s \right|$$

        onde $k$ é escolhido para que $f_{alias} \in [0, f_s/2]$

        ### Reconstrução

        Sob a condição de Nyquist ($f_s > 2f_{max}$), o sinal original pode ser
        **perfeitamente reconstruído** usando um filtro passa-baixas ideal.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Controles Interativos""")
    return


@app.cell
def __(mo):
    # Frequência do sinal
    f_signal_slider = mo.ui.slider(
        start=100,
        stop=5000,
        step=100,
        value=1000,
        label="Frequência do Sinal (f) [Hz]",
        show_value=True
    )

    # Taxa de amostragem
    fs_nyquist_slider = mo.ui.slider(
        start=100,
        stop=15000,
        step=100,
        value=3000,
        label="Taxa de Amostragem (fs) [Hz]",
        show_value=True
    )

    # Duração da janela de tempo (ms)
    duration_nyquist_ms = mo.ui.number(
        start=5,
        stop=100,
        step=5,
        value=20,
        label="Duração da Janela (ms)"
    )

    # Amplitude do sinal
    amplitude_signal = mo.ui.number(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=1.0,
        label="Amplitude do Sinal"
    )

    # Botão de reset
    reset_nyquist_button = mo.ui.button(
        label="🔄 Reset para Defaults"
    )

    mo.md(
        f"""
        {f_signal_slider}

        {fs_nyquist_slider}

        {duration_nyquist_ms}

        {amplitude_signal}

        {reset_nyquist_button}
        """
    )
    return (
        amplitude_signal,
        duration_nyquist_ms,
        f_signal_slider,
        fs_nyquist_slider,
        reset_nyquist_button,
    )


@app.cell
def __(
    amplitude_signal,
    duration_nyquist_ms,
    f_signal_slider,
    fs_nyquist_slider,
    reset_nyquist_button,
):
    # Processamento dos valores dos controles
    if reset_nyquist_button.value:
        f_signal = 1000
        fs_nyquist = 3000
        T_window_nyquist_ms = 20
        A_signal = 1.0
    else:
        f_signal = f_signal_slider.value
        fs_nyquist = fs_nyquist_slider.value
        T_window_nyquist_ms = duration_nyquist_ms.value
        A_signal = amplitude_signal.value

    # Conversão para segundos
    T_window_nyquist = T_window_nyquist_ms * 1e-3

    # Taxa de Nyquist
    f_nyquist_rate = 2 * f_signal

    # Verificação da condição de Nyquist
    nyquist_satisfied = fs_nyquist > f_nyquist_rate

    # Cálculo da frequência de aliasing (se aplicável)
    if not nyquist_satisfied:
        # Encontrar k que minimiza |f_signal - k*fs_nyquist|
        k_values = np.arange(-5, 6)
        aliases = np.abs(f_signal - k_values * fs_nyquist)
        # Escolher o alias que cai em [0, fs_nyquist/2]
        valid_aliases = aliases[(aliases >= 0) & (aliases <= fs_nyquist/2)]
        f_alias = valid_aliases[0] if len(valid_aliases) > 0 else f_signal % fs_nyquist
    else:
        f_alias = None
    return (
        A_signal,
        T_window_nyquist,
        T_window_nyquist_ms,
        f_alias,
        f_nyquist_rate,
        f_signal,
        fs_nyquist,
        k_values,
        nyquist_satisfied,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🧮 Geração e Amostragem do Sinal""")
    return


@app.cell
def __(A_signal, T_window_nyquist, f_signal, fs_nyquist, np):
    # Sinal contínuo (alta taxa de amostragem para visualização)
    fs_continuous = max(f_signal * 50, 50000)
    t_continuous = np.arange(0, T_window_nyquist, 1/fs_continuous)
    signal_continuous = A_signal * np.sin(2 * np.pi * f_signal * t_continuous)

    # Sinal amostrado (na taxa escolhida)
    t_sampled = np.arange(0, T_window_nyquist, 1/fs_nyquist)
    signal_sampled = A_signal * np.sin(2 * np.pi * f_signal * t_sampled)

    # Sinal reconstruído (interpolação ideal - sinc)
    # Para visualização, usamos interpolação simples
    # Em sistema real, seria um filtro passa-baixas ideal
    from scipy.interpolate import interp1d
    
    if len(t_sampled) > 1:
        interp_func = interp1d(t_sampled, signal_sampled, kind='cubic', 
                               bounds_error=False, fill_value=0)
        signal_reconstructed = interp_func(t_continuous)
    else:
        signal_reconstructed = np.zeros_like(t_continuous)
    return (
        fs_continuous,
        interp1d,
        interp_func,
        signal_continuous,
        signal_reconstructed,
        signal_sampled,
        t_continuous,
        t_sampled,
    )


@app.cell
def __(f_alias, f_nyquist_rate, f_signal, fs_nyquist, mo, nyquist_satisfied):
    status_emoji = "✅" if nyquist_satisfied else "⚠️"
    status_text = "Sem Aliasing" if nyquist_satisfied else "ALIASING DETECTADO"
    status_kind = "success" if nyquist_satisfied else "warn"

    mo.md(
        f"""
        ## 📊 Análise da Amostragem

        ### Condição de Nyquist

        - **Frequência do Sinal:** {f_signal} Hz
        - **Taxa de Nyquist (mínima):** {f_nyquist_rate} Hz
        - **Taxa de Amostragem (fs):** {fs_nyquist} Hz
        - **fs/2 (Freq. de Nyquist):** {fs_nyquist/2} Hz
        - **Razão fs/fNyquist:** {fs_nyquist/f_nyquist_rate:.2f}×

        ### Status: {status_emoji} **{status_text}**

        {mo.callout(
            f"**Condição satisfeita!** fs ({fs_nyquist} Hz) > 2×f ({f_nyquist_rate} Hz)\\n\\n"
            f"O sinal pode ser perfeitamente reconstruído.",
            kind="success"
        ) if nyquist_satisfied else mo.callout(
            f"**ALIASING!** fs ({fs_nyquist} Hz) ≤ 2×f ({f_nyquist_rate} Hz)\\n\\n"
            f"Frequência aparente (alias): ~**{f_alias:.1f} Hz**\\n\\n"
            f"O sinal original de {f_signal} Hz aparece como {f_alias:.1f} Hz após amostragem!",
            kind="warn"
        )}
        """
    )
    return status_emoji, status_kind, status_text


@app.cell
def __(mo):
    mo.md(r"""## 📈 Visualização no Domínio do Tempo""")
    return


@app.cell
def __(
    A_signal,
    Figure,
    f_signal,
    nyquist_satisfied,
    plt,
    signal_continuous,
    signal_reconstructed,
    signal_sampled,
    t_continuous,
    t_sampled,
):
    # Gráfico no domínio do tempo
    fig_time_nyquist = Figure(figsize=(14, 10))
    
    # Sinal original (contínuo) vs amostrado
    ax1_nyquist = fig_time_nyquist.add_subplot(3, 1, 1)
    ax1_nyquist.plot(t_continuous * 1000, signal_continuous, 'b-', linewidth=1.5, 
                     label=f'Sinal Original ({f_signal} Hz)', alpha=0.7)
    ax1_nyquist.plot(t_sampled * 1000, signal_sampled, 'ro', markersize=8, 
                     label='Amostras', zorder=5)
    ax1_nyquist.stem(t_sampled * 1000, signal_sampled, linefmt='r-', markerfmt='ro',
                     basefmt='k-', alpha=0.3)
    ax1_nyquist.set_ylabel('Amplitude', fontsize=11)
    ax1_nyquist.set_title('Sinal Original e Amostras', fontsize=13, fontweight='bold')
    ax1_nyquist.grid(True, alpha=0.3)
    ax1_nyquist.legend(loc='upper right')
    ax1_nyquist.set_xlim([0, t_continuous[-1] * 1000])
    ax1_nyquist.set_ylim([-A_signal * 1.2, A_signal * 1.2])

    # Sinal reconstruído
    ax2_nyquist = fig_time_nyquist.add_subplot(3, 1, 2)
    ax2_nyquist.plot(t_continuous * 1000, signal_continuous, 'b--', linewidth=1.5, 
                     label='Original', alpha=0.5)
    ax2_nyquist.plot(t_continuous * 1000, signal_reconstructed, 'g-', linewidth=2, 
                     label='Reconstruído (interpolado)', alpha=0.8)
    ax2_nyquist.plot(t_sampled * 1000, signal_sampled, 'ro', markersize=6, 
                     label='Amostras')
    ax2_nyquist.set_ylabel('Amplitude', fontsize=11)
    ax2_nyquist.set_title('Sinal Reconstruído vs Original', fontsize=13, fontweight='bold')
    ax2_nyquist.grid(True, alpha=0.3)
    ax2_nyquist.legend(loc='upper right')
    ax2_nyquist.set_xlim([0, t_continuous[-1] * 1000])
    ax2_nyquist.set_ylim([-A_signal * 1.2, A_signal * 1.2])

    # Erro de reconstrução
    ax3_nyquist = fig_time_nyquist.add_subplot(3, 1, 3)
    reconstruction_error = signal_continuous - signal_reconstructed
    ax3_nyquist.plot(t_continuous * 1000, reconstruction_error, 'r-', linewidth=1, 
                     label='Erro (Original - Reconstruído)')
    ax3_nyquist.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3_nyquist.set_xlabel('Tempo (ms)', fontsize=11)
    ax3_nyquist.set_ylabel('Erro', fontsize=11)
    
    error_rms = np.sqrt(np.mean(reconstruction_error**2))
    title_error = f'Erro de Reconstrução (RMS: {error_rms:.4f})'
    if nyquist_satisfied:
        title_error += ' - ✅ Erro Baixo'
    else:
        title_error += ' - ⚠️ Erro Alto (Aliasing)'
    
    ax3_nyquist.set_title(title_error, fontsize=13, fontweight='bold')
    ax3_nyquist.grid(True, alpha=0.3)
    ax3_nyquist.legend(loc='upper right')
    ax3_nyquist.set_xlim([0, t_continuous[-1] * 1000])

    fig_time_nyquist.tight_layout()
    plt.close(fig_time_nyquist)
    fig_time_nyquist
    return (
        ax1_nyquist,
        ax2_nyquist,
        ax3_nyquist,
        error_rms,
        fig_time_nyquist,
        reconstruction_error,
        title_error,
    )


@app.cell
def __(mo):
    mo.md(r"""## 🌊 Visualização no Domínio da Frequência""")
    return


@app.cell
def __(fs_continuous, fs_nyquist, np, signal_continuous, signal_sampled):
    # FFT do sinal contínuo (original)
    N_continuous = len(signal_continuous)
    fft_continuous = np.fft.fft(signal_continuous)
    fft_continuous_mag = np.abs(fft_continuous) / N_continuous
    fft_continuous_mag = fft_continuous_mag[:N_continuous//2]
    freqs_continuous = np.fft.fftfreq(N_continuous, 1/fs_continuous)
    freqs_continuous = freqs_continuous[:N_continuous//2]

    # FFT do sinal amostrado
    N_sampled = len(signal_sampled)
    if N_sampled > 1:
        fft_sampled = np.fft.fft(signal_sampled)
        fft_sampled_mag = np.abs(fft_sampled) / N_sampled
        fft_sampled_mag = fft_sampled_mag[:N_sampled//2]
        freqs_sampled = np.fft.fftfreq(N_sampled, 1/fs_nyquist)
        freqs_sampled = freqs_sampled[:N_sampled//2]
    else:
        fft_sampled_mag = np.array([])
        freqs_sampled = np.array([])
    return (
        N_continuous,
        N_sampled,
        fft_continuous,
        fft_continuous_mag,
        fft_sampled,
        fft_sampled_mag,
        freqs_continuous,
        freqs_sampled,
    )


@app.cell
def __(
    Figure,
    f_signal,
    fft_continuous_mag,
    fft_sampled_mag,
    freqs_continuous,
    freqs_sampled,
    fs_nyquist,
    plt,
):
    # Gráfico no domínio da frequência
    fig_freq_nyquist = Figure(figsize=(14, 8))
    
    # Espectro do sinal original
    ax1_freq = fig_freq_nyquist.add_subplot(2, 1, 1)
    ax1_freq.plot(freqs_continuous, fft_continuous_mag, 'b-', linewidth=1.5, 
                  label='Espectro Original')
    ax1_freq.axvline(x=f_signal, color='r', linestyle='--', alpha=0.7, linewidth=2,
                     label=f'Frequência do Sinal ({f_signal} Hz)')
    ax1_freq.axvline(x=fs_nyquist/2, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                     label=f'fs/2 = {fs_nyquist/2} Hz (Nyquist)')
    ax1_freq.set_ylabel('Magnitude', fontsize=11)
    ax1_freq.set_title('Espectro do Sinal Original (Contínuo)', fontsize=13, fontweight='bold')
    ax1_freq.grid(True, alpha=0.3)
    ax1_freq.legend(loc='upper right')
    ax1_freq.set_xlim([0, min(fs_nyquist * 2, freqs_continuous[-1])])

    # Espectro do sinal amostrado (com réplicas)
    ax2_freq = fig_freq_nyquist.add_subplot(2, 1, 2)
    
    if len(freqs_sampled) > 0:
        ax2_freq.plot(freqs_sampled, fft_sampled_mag, 'g-', linewidth=1.5, 
                      label='Espectro Amostrado', marker='o', markersize=4)
    
    ax2_freq.axvline(x=f_signal, color='r', linestyle='--', alpha=0.7, linewidth=2,
                     label=f'Freq. Original ({f_signal} Hz)')
    ax2_freq.axvline(x=fs_nyquist/2, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                     label=f'fs/2 = {fs_nyquist/2} Hz')
    
    # Zona de aliasing
    ax2_freq.axvspan(fs_nyquist/2, fs_nyquist, alpha=0.2, color='red', 
                     label='Zona de Aliasing')
    
    ax2_freq.set_xlabel('Frequência (Hz)', fontsize=11)
    ax2_freq.set_ylabel('Magnitude', fontsize=11)
    ax2_freq.set_title('Espectro do Sinal Amostrado (com réplicas espectrais)', 
                       fontsize=13, fontweight='bold')
    ax2_freq.grid(True, alpha=0.3)
    ax2_freq.legend(loc='upper right')
    ax2_freq.set_xlim([0, fs_nyquist])

    fig_freq_nyquist.tight_layout()
    plt.close(fig_freq_nyquist)
    fig_freq_nyquist
    return ax1_freq, ax2_freq, fig_freq_nyquist


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📚 Notas Pedagógicas

        ### O que observar:

        1. **Condição de Nyquist:**
           - Quando **fs > 2f**: o sinal pode ser perfeitamente reconstruído
           - Quando **fs ≤ 2f**: ocorre aliasing e a reconstrução é incorreta

        2. **Visualização no Tempo:**
           - Com fs adequado: a interpolação das amostras reproduz o sinal original
           - Com fs inadequado: a interpolação cria um sinal de frequência diferente
           - O erro de reconstrução é mínimo quando Nyquist é satisfeito

        3. **Visualização na Frequência:**
           - Espectro original: pico único na frequência do sinal
           - Após amostragem: réplicas espectrais aparecem a cada múltiplo de fs
           - **Zona crítica:** frequências acima de fs/2 "se dobram" para baixo

        4. **Efeito de Aliasing:**
           - Um sinal de frequência f aparece como f_alias após amostragem inadequada
           - f_alias = |f - k·fs| (mais próximo de [0, fs/2])
           - Exemplo: f=3000 Hz, fs=2000 Hz → f_alias=1000 Hz

        5. **Interpretação Prática:**
           - **Audio:** Taxa CD (44.1 kHz) captura até ~20 kHz (limite audível)
           - **Video:** 60 fps captura movimento até ~30 Hz
           - **Telecomunicações:** fs deve ser > 2× largura de banda do canal

        ### Experimentos Sugeridos:

        1. **Demonstrar Nyquist:**
           - Fixe f=1000 Hz
           - Varie fs de 1500 Hz (abaixo) → 3000 Hz (acima)
           - Observe a transição de aliasing → sem aliasing

        2. **Calcular Aliasing:**
           - f=3000 Hz, fs=2000 Hz → alias~1000 Hz
           - f=4500 Hz, fs=3000 Hz → alias~1500 Hz
           - Compare com o valor calculado na métrica

        3. **Erro de Reconstrução:**
           - Note que o RMS do erro é alto quando há aliasing
           - Com fs > 2f, erro tende a zero (limitado apenas pela interpolação)

        4. **Réplicas Espectrais:**
           - No espectro amostrado, veja as réplicas a cada fs
           - Quando f > fs/2, uma réplica entra na banda base ([0, fs/2])

        ### Filtros Anti-Aliasing

        Na prática, um **filtro passa-baixas** é aplicado **antes** da amostragem
        para remover componentes acima de fs/2, evitando aliasing:

        $$H(f) = \begin{cases} 
        1, & |f| \leq f_s/2 \\
        0, & |f| > f_s/2
        \end{cases}$$

        Este é o filtro **anti-aliasing**.

        ### Reconstrução Ideal

        Com a condição de Nyquist satisfeita, a reconstrução perfeita é feita
        por um filtro **passa-baixas ideal** (função sinc):

        $$x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t - nT_s}{T_s}\right)$$

        onde $T_s = 1/f_s$ é o período de amostragem.

        ---

        **Referências:**
        - Shannon, C.E. (1949) "Communication in the Presence of Noise"
        - Nyquist, H. (1928) "Certain Topics in Telegraph Transmission Theory"
        - Oppenheim & Schafer "Discrete-Time Signal Processing"
        """
    )
    return


@app.cell
def __():
    # Informações sobre o notebook
    __notebook_info_nyquist__ = {
        "title": "Amostragem e Teorema de Nyquist",
        "version": "1.0",
        "date": "2025-10-30",
        "python": "3.14+",
        "dependencies": ["marimo", "numpy>=2.0", "matplotlib>=3.9", "scipy"]
    }
    return


if __name__ == "__main__":
    app.run()
