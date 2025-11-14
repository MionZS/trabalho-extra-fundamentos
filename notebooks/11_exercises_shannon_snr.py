import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.figure import Figure
    return Figure, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 📋 Exercícios Interativos: Shannon, Quantização e SNR

    ## Objetivo

    Explorar interativamente os conceitos de:
    - **Teorema de Shannon-Nyquist** (Exercício 0 e 0B)
    - **Amostragem e Quantização** (Exercícios 1-5)
    - **Relação SNR vs Bits** (Exercício 4)

    Use os gráficos e simulações abaixo como apoio para responder às questões.
    """)
    return


@app.cell
def _():
    # Piano notes: A0 to A7
    piano_notes = {}
    for octave in range(0, 8):
        freq = 27.5 * (2 ** octave)
        piano_notes[f"A{octave} ({freq:.2f} Hz)"] = float(freq)
    default_note = "A4 (440.00 Hz)" if "A4 (440.00 Hz)" in piano_notes else next(iter(piano_notes))
    return default_note, piano_notes


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuração do Simulador
    """)
    return


@app.cell
def _(default_note, mo, piano_notes):
    tone = mo.ui.dropdown(options=list(piano_notes.keys()), value=default_note, label="Selecione o Tom:")
    bits = mo.ui.slider(4, 16, step=1, value=8, label="Bits de resolução:")
    sampling_freq = mo.ui.text(value="17600", label="Frequência de Amostragem (Hz):")

    mo.md(f"""
    {tone}

    {bits}

    {sampling_freq}
    """)
    return bits, sampling_freq, tone


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Exercício 0: Teorema de Shannon-Nyquist

    **Conceito**: A frequência de Nyquist é $f_N = 2f_0$. Para amostrar sem perda de informação:
    $$f_s > 2f_0$$

    ### Questões:

    **0.1** Calcule a frequência de Nyquist para os seguintes tons:
    - A0 (27,5 Hz) → $f_N = ?$
    - A3 (220 Hz) → $f_N = ?$
    - A4 (440 Hz) → $f_N = ?$
    - A6 (1760 Hz) → $f_N = ?$
    - A7 (3520 Hz) → $f_N = ?$

    **0.2** Com $f_s = 17600$ Hz, qual é o tom de **maior frequência** que pode ser amostrado sem aliasing?

    **0.3** Observe o gráfico abaixo (6 subplots). Identifique: em qual caso a condição de Nyquist deixa de ser satisfeita?
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt):
    # Exercício 0: Visualizar várias frequências de amostragem
    _fs_values = [f0, 1.5*f0, 2*f0, 3*f0, 4*f0]

    fig_shannon = Figure(figsize=(14, 6))

    _sqnr_theoretical = 6.02 * bits.value + 1.76

    for idx, _fs_test in enumerate(_fs_values, 1):
        ax = fig_shannon.add_subplot(2, 3, idx)

        # Gerar sinal denso
        _t_dense = np.linspace(0, 3/f0, 1000)
        _x_dense = np.sin(2 * np.pi * f0 * _t_dense)

        # Amostrar e quantizar
        _t_sample = np.arange(0, 3/f0, 1/_fs_test)
        _x_sample = np.sin(2 * np.pi * f0 * _t_sample)

        # Quantização
        _levels = 2 ** bits.value
        _delta = 2.0 / (_levels - 1)
        _x_quantized = np.round(_x_sample / _delta) * _delta

        # Plot
        ax.plot(_t_dense * 1e3, _x_dense, 'b:', linewidth=1, alpha=0.4, label='Original')
        ax.plot(_t_sample * 1e3, _x_quantized, 'ro', markersize=5, label='Quantizado')

        # Validar Shannon
        _f_nyquist_req = 2 * f0
        _status = "✓ OK" if _fs_test > _f_nyquist_req else "✗ Aliasing"
        _color_title = 'green' if _fs_test > _f_nyquist_req else 'red'

        ax.set_title(f'$f_s = {_fs_test/f0:.1f} \\cdot f_0$\n{_status}',
                    fontsize=10, color=_color_title, weight='bold')
        ax.set_xlabel('Tempo (ms)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    fig_shannon.suptitle(f'Ex. 0: Comparação de Frequências (tom={f0:.1f} Hz, {bits.value} bits)',
                        fontsize=12, weight='bold')
    fig_shannon.tight_layout()
    plt.close(fig_shannon)
    fig_shannon
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Exercício 0B: Reconstrução de Shannon (Interpolação Sinc)

    **Conceito**: Se Nyquist é satisfeito, podemos **reconstruir perfeitamente** o sinal:
    $$x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t - nT_s}{T_s}\right)$$

    onde $\text{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$.

    ### Questões:

    **0B.1** Observe o gráfico esquerdo abaixo:
    - Linha azul tracejada = Sinal original (referência)
    - Linha vermelha sólida = Sinal reconstruído via sinc
    - Pontos verdes = Amostras quantizadas

    **Pergunta**: Com seu tom e bits selecionados, o sinal reconstruído (vermelho) está próximo do original (azul)?

    **0B.2** Observe o gráfico direito (erro em escala logarítmica):
    - O erro de reconstrução é maior ou menor que $10^{-5}$?

    **0B.3** Agora mude a frequência de amostragem para 1000 Hz (campo acima) e execute novamente:
    - O sinal reconstruído ainda sobrepõe o original?
    - O que mudou no gráfico de erro?
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt, sampling_freq):
    # Exercício 0B: Reconstrução de Shannon
    _fs_recon = float(sampling_freq.value.strip()) if sampling_freq.value.strip() else 17600
    _t_recon_period = 1 / _fs_recon

    # Gerar tempo contínuo para reconstrução
    _t_recon = np.linspace(0, 3 / f0, 5000)

    # Amostras
    _t_samp_recon = np.arange(0, 3 / f0, 1 / _fs_recon)
    _x_samp_recon = np.sin(2 * np.pi * f0 * _t_samp_recon)

    # Quantizar
    _levels = 2 ** bits.value
    _delta = 2.0 / (_levels - 1)
    _x_q_recon = np.round(_x_samp_recon / _delta) * _delta

    # Reconstruir via Sinc
    _x_recon = np.zeros_like(_t_recon)
    for _n, _x_n in enumerate(_x_q_recon):
        _t_shifted = (_t_recon - _n * _t_recon_period) / _t_recon_period
        _sinc_vals = np.sinc(_t_shifted)
        _x_recon += _x_n * _sinc_vals

    # Sinal original
    _x_orig_recon = np.sin(2 * np.pi * f0 * _t_recon)

    # Plotar
    fig_recon = Figure(figsize=(14, 6))

    # Subplot 1: Sinais sobrepostos
    ax1 = fig_recon.add_subplot(1, 2, 1)
    ax1.plot(_t_recon * 1e3, _x_orig_recon, 'b:', linewidth=1, label='Original (ref)', alpha=0.6)
    ax1.plot(_t_recon * 1e3, _x_recon, 'r-', linewidth=2, label=f'Reconstruído ({bits.value}b)', alpha=0.8)
    ax1.plot(_t_samp_recon * 1e3, _x_q_recon, 'go', markersize=4, label='Amostras')
    ax1.set_xlabel('Tempo (ms)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Ex. 0B: Reconstrução Sinc ($f_s=${_fs_recon:.0f} Hz)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim([0, 3 / f0 * 1e3])

    # Subplot 2: Erro
    _error_recon = np.abs(_x_orig_recon - _x_recon)
    ax2 = fig_recon.add_subplot(1, 2, 2)
    ax2.semilogy(_t_recon * 1e3, _error_recon + 1e-15, 'r-', linewidth=2, label='Erro absoluto')
    ax2.set_xlabel('Tempo (ms)', fontsize=11)
    ax2.set_ylabel('Erro (log)', fontsize=11)
    ax2.set_title('Erro de Reconstrução', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim([0, 3 / f0 * 1e3])

    fig_recon.tight_layout()
    plt.close(fig_recon)
    fig_recon
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Exercício 1: Amostragem de Tom Puro

    **Conceito**: Com frequência de amostragem fixa, observamos diferentes números de amostras por período.

    ### Questões:

    **1.1** Selecione A4 (440 Hz). Para cada frequência de amostragem abaixo, calcule:
    $$\text{Amostras/Período} = \frac{f_s}{f_0}$$

    - $f_s = 17600$ Hz → Amostras/período = ?
    - $f_s = 8800$ Hz → Amostras/período = ?
    - $f_s = 4400$ Hz → Amostras/período = ?

    **1.2** Observe o gráfico abaixo para $f_s = 17600$ Hz:
    - O sinal amostrado parece suave ou discreto?
    - Quantos pontos verdes (amostras) você conta em um período?
    """)
    return


@app.cell
def _(Figure, f0, np, plt, sampling_freq):
    # Exercício 1: Amostragem
    _fs_ex1 = float(sampling_freq.value.strip()) if sampling_freq.value.strip() else 17600
    _t_ex1 = np.arange(0, 2/f0, 1/_fs_ex1)  # 2 períodos
    _x_ex1 = np.sin(2 * np.pi * f0 * _t_ex1)

    # Sinal denso para comparação
    _t_dense_ex1 = np.linspace(0, 2/f0, 2000)
    _x_dense_ex1 = np.sin(2 * np.pi * f0 * _t_dense_ex1)

    fig_ex1 = Figure(figsize=(12, 4))
    ax_ex1 = fig_ex1.add_subplot(1, 1, 1)
    ax_ex1.plot(_t_dense_ex1 * 1e3, _x_dense_ex1, 'b-', linewidth=1, alpha=0.5, label='Contínuo')
    ax_ex1.plot(_t_ex1 * 1e3, _x_ex1, 'go', markersize=5, label=f'Amostras ($f_s$={_fs_ex1:.0f} Hz)')
    ax_ex1.set_xlabel('Tempo (ms)', fontsize=11)
    ax_ex1.set_ylabel('Amplitude', fontsize=11)
    ax_ex1.set_title(f'Ex. 1: Amostragem de {f0:.1f} Hz com $f_s=${_fs_ex1:.0f} Hz', fontsize=12, weight='bold')
    ax_ex1.grid(True, alpha=0.3)
    ax_ex1.legend(fontsize=10)
    fig_ex1.tight_layout()
    plt.close(fig_ex1)
    fig_ex1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Exercício 2: Quantização e Ruído

    **Conceito**: Quantização uniforme com $n$ bits introduz ruído. Calculamos:
    - Passo: $\Delta = \frac{2m_{\max}}{2^n}$
    - Potência de ruído: $P_q = \frac{\Delta^2}{12}$
    - SQNR teórico: $\text{SQNR}_q [\text{dB}] = 6.02n + 1.76$

    ### Questões:

    **2.1** Calcule para A = 1 V:

    | n (bits) | $\Delta$ (V) | $P_q$ (V²) | SQNR (dB) |
    |----------|-------------|-----------|----------|
    | 4        | ?           | ?         | ?        |
    | 8        | ?           | ?         | ?        |
    | 12       | ?           | ?         | ?        |
    | 16       | ?           | ?         | ?        |

    **2.2** Observe o gráfico abaixo:
    - Com seus bits selecionados, qual é a amplitude máxima do ruído (linha verde)?
    - O ruído é maior ou menor com mais bits?
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt, sampling_freq):
    # Exercício 2: Quantização e Ruído
    _fs_ex2 = float(sampling_freq.value.strip()) if sampling_freq.value.strip() else 17600
    _num_periods_ex2 = 1
    _t_ex2 = np.arange(0, f0**-1 * _num_periods_ex2, 1/_fs_ex2)
    _x_ex2 = np.sin(2 * np.pi * f0 * _t_ex2)

    # Quantização
    _levels_ex2 = 2 ** bits.value
    _delta_ex2 = 2.0 / (_levels_ex2 - 1)
    _x_q_ex2 = np.round(_x_ex2 / _delta_ex2) * _delta_ex2
    _noise_ex2 = _x_ex2 - _x_q_ex2

    fig_ex2 = Figure(figsize=(14, 5))

    # Subplot 1: Sinal quantizado
    ax1_ex2 = fig_ex2.add_subplot(1, 2, 1)
    ax1_ex2.plot(_t_ex2 * 1e3, _x_ex2, 'b-', linewidth=1, alpha=0.5, label='Original')
    ax1_ex2.plot(_t_ex2 * 1e3, _x_q_ex2, 'r.-', linewidth=1, markersize=3, label=f'Quantizado ({bits.value}b)')
    ax1_ex2.set_xlabel('Tempo (ms)', fontsize=11)
    ax1_ex2.set_ylabel('Amplitude', fontsize=11)
    ax1_ex2.set_title('Ex. 2a: Sinal Quantizado', fontsize=12, weight='bold')
    ax1_ex2.grid(True, alpha=0.3)
    ax1_ex2.legend(fontsize=10)

    # Subplot 2: Ruído
    ax2_ex2 = fig_ex2.add_subplot(1, 2, 2)
    ax2_ex2.plot(_t_ex2 * 1e3, _noise_ex2, 'g.-', linewidth=1, markersize=3, label=f'Ruído ({bits.value}b)')
    ax2_ex2.axhline(_delta_ex2/2, color='r', linestyle='--', linewidth=1, alpha=0.5, label=fr'$\Delta/2$ = {_delta_ex2/2:.4f}')
    ax2_ex2.set_xlabel('Tempo (ms)', fontsize=11)
    ax2_ex2.set_ylabel('Ruído (V)', fontsize=11)
    ax2_ex2.set_title('Ex. 2b: Ruído de Quantização', fontsize=12, weight='bold')
    ax2_ex2.grid(True, alpha=0.3)
    ax2_ex2.legend(fontsize=10)

    fig_ex2.tight_layout()
    plt.close(fig_ex2)
    fig_ex2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Exercício 4: Relação SNR × Bits

    **Conceito**: SQNR teórico é $\text{SQNR}_q = 6.02n + 1.76$ dB. Cada bit adiciona ~6 dB.

    ### Questões:

    **4.1** Plote SQNR teórico para $n = 4$ a $16$ bits. A relação é linear?

    **4.2** Observe o gráfico abaixo à direita (erro de reconstrução vs bits):
    - Com 4 bits, qual é a ordem de magnitude do erro RMS?
    - Com 16 bits, qual é o erro RMS?
    - O erro diminui linearmente ou exponencialmente?

    **4.3** Estime a inclinação (dB/bit) do gráfico de erro. É próxima a 6 dB/bit?
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt):
    # Exercício 4: SNR vs Bits
    fig_snr_ex4 = Figure(figsize=(14, 5))

    # Gráfico 1: SQNR teórico
    _ax1_ex4 = fig_snr_ex4.add_subplot(1, 2, 1)
    _bits_range_ex4 = np.arange(4, 17)
    _sqnr_db_ex4 = 6.02 * _bits_range_ex4 + 1.76
    _ax1_ex4.plot(_bits_range_ex4, _sqnr_db_ex4, 'b-', linewidth=2.5, label='SQNR teórico')
    _ax1_ex4.plot(bits.value, 6.02 * bits.value + 1.76, 'ro', markersize=10, label=f'Atual ({bits.value}b)')
    _ax1_ex4.axhline(y=96, color='r', linestyle='--', linewidth=1, alpha=0.5, label='CD (16-bit)')
    _ax1_ex4.set_xlabel('Número de Bits', fontsize=11)
    _ax1_ex4.set_ylabel('SQNR (dB)', fontsize=11)
    _ax1_ex4.set_title('Ex. 4a: SQNR Teórico vs Bits', fontsize=12, weight='bold')
    _ax1_ex4.grid(True, alpha=0.3)
    _ax1_ex4.legend(fontsize=10)
    _ax1_ex4.set_xlim([4, 16])

    # Gráfico 2: Erro de reconstrução vs bits
    _ax2_ex4 = fig_snr_ex4.add_subplot(1, 2, 2)

    _fs_recon_ex4 = 17600
    _t_recon_period_ex4 = 1 / _fs_recon_ex4
    _t_recon_ex4 = np.linspace(0, 3/f0, 1000)
    _x_orig_ex4 = np.sin(2 * np.pi * f0 * _t_recon_ex4)

    _t_samp_ex4 = np.arange(0, 3/f0, 1/_fs_recon_ex4)
    _x_samp_ex4 = np.sin(2 * np.pi * f0 * _t_samp_ex4)

    _error_by_bits_ex4 = []
    _bits_test_ex4 = np.arange(4, 17)

    for _b in _bits_test_ex4:
        _levels = 2 ** _b
        _delta = 2.0 / (_levels - 1)
        _x_q = np.round(_x_samp_ex4 / _delta) * _delta

        # Reconstruir
        _x_recon = np.zeros_like(_t_recon_ex4)
        for _n, _x_n in enumerate(_x_q):
            _t_shifted = (_t_recon_ex4 - _n * _t_recon_period_ex4) / _t_recon_period_ex4
            _sinc_vals = np.sinc(_t_shifted)
            _x_recon += _x_n * _sinc_vals

        # Erro RMS
        _err_rms = np.sqrt(np.mean((_x_orig_ex4 - _x_recon)**2))
        _error_by_bits_ex4.append(_err_rms)

    _ax2_ex4.semilogy(_bits_test_ex4, _error_by_bits_ex4, 'r-o', linewidth=2.5, markersize=6, label='Erro RMS (Shannon)')
    _ax2_ex4.plot(bits.value, _error_by_bits_ex4[bits.value - 4], 'go', markersize=12, label=f'Atual ({bits.value}b)')
    _ax2_ex4.set_xlabel('Número de Bits', fontsize=11)
    _ax2_ex4.set_ylabel('Erro RMS (log)', fontsize=11)
    _ax2_ex4.set_title('Ex. 4b: Saturação de Shannon por Quantização', fontsize=12, weight='bold')
    _ax2_ex4.grid(True, alpha=0.3, which='both')
    _ax2_ex4.legend(fontsize=10)
    _ax2_ex4.set_xlim([4, 16])

    fig_snr_ex4.tight_layout()
    plt.close(fig_snr_ex4)
    fig_snr_ex4
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📌 Tabelas para Preenchimento

    ### Tabela 0: Exercício 0 --- Teorema de Shannon (Preencha)

    | Tom | $f_0$ (Hz) | $f_N = 2f_0$ (Hz) | $f_s$ (Hz) | $f_s \geq f_N$? | Visualização |
    |-----|-----------|----------------|-----------|-------------|--------------|
    | A0  | 27,5      |                | 17600     |             |              |
    | A3  | 220       |                | 17600     |             |              |
    | A4  | 440       |                | 17600     |             |              |
    | A6  | 1760      |                | 17600     |             |              |
    | A7  | 3520      |                | 17600     |             |              |

    ### Tabela 0B: Exercício 0B --- Reconstrução de Shannon (Preencha)

    | Tom | $f_0$ (Hz) | $f_s$ (Hz) | Erro Máx. (log) | Observação |
    |-----|-----------|-----------|-----------------|------------|
    | A4  | 440       | 17600     |                 |  |
    | A4  | 440       | 8800      |                 |  |
    | A4  | 440       | 1000      |                 |  |

    ### Tabela 2: Exercício 1 --- Amostragem (Preencha)

    | Tom | $f_s$ (Hz) | Amostras/Período | $f_0$ Teórica (Hz) | $f_0$ Observada (Hz) |
    |-----|-----------|------------------|-------------------|----------------------|
    | A4  | 17600     |                  | 440                |                      |
    | A4  | 8800      |                  | 440                |                      |
    | A4  | 4400      |                  | 440                |                      |

    ### Tabela 3: Exercício 2 --- Quantização (Preencha)

    | n (bits) | $\Delta$ (V) | $P_q$ (V²) | SQNR Teórico (dB) |
    |----------|-------------|-----------|-------------------|
    | 4        |             |           |                   |
    | 8        |             |           |                   |
    | 12       |             |           |                   |
    | 16       |             |           |                   |

    ### Tabela 5: Exercício 4 --- SNR vs Bits (Preencha)

    | n (bits) | SQNR Teórico (dB) | Erro RMS (Ex. 4b) | Observação |
    |----------|-------------------|-------------------|------------|
    | 4        |                   |                   |            |
    | 6        |                   |                   |            |
    | 8        |                   |                   |            |
    | 10       |                   |                   |            |
    | 12       |                   |                   |            |
    | 14       |                   |                   |            |
    | 16       |                   |                   |            |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 📋 Resumo de Fórmulas

    ### Teorema de Shannon-Nyquist

    **Critério (Nyquist)**:
    $$f_s > 2f_0$$

    **Reconstrução (Shannon)**:
    $$x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t - nT_s}{T_s}\right), \quad \text{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$$

    ### Quantização Uniforme

    **Passo de quantização**:
    $$\Delta = \frac{2m_{\max}}{L} = \frac{2m_{\max}}{2^n}$$

    **Potência de ruído**:
    $$P_q = \frac{\Delta^2}{12}$$

    **SQNR teórico**:
    $$\text{SQNR}_q [\text{dB}] = 6.02\,n + 1.76$$

    onde $n$ é o número de bits.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 🎓 Instruções

    1. **Configure o simulador** acima (tom, bits, frequência de amostragem)
    2. **Observe os gráficos** para cada exercício
    3. **Preencha as tabelas** com seus resultados
    4. **Responda as questões** em texto
    5. **Compare teórico × simulação** e discuta desvios

    ### Entrega

    - PDF com tabelas preenchidas
    - Capturas dos gráficos do Marimo
    - **Explicações dos conceitos e análises dos resultados** (teórico vs simulação)
    """)
    return


@app.cell
def _(piano_notes, tone):
    # Calcula f0 do tom selecionado
    f0 = piano_notes[tone.value]
    return (f0,)


if __name__ == "__main__":
    app.run()
