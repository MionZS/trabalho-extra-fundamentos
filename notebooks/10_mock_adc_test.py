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
    # Simulador de ADC: Amostragem e Quantização
    """)
    return


@app.cell
def _():
    # Only A notes (Lás): A0 to A7
    piano_notes = {}
    for octave in range(0, 8):
        freq = 27.5 * (2 ** octave)
        piano_notes[f"A{octave} ({freq:.2f} Hz)"] = float(freq)
    default_note = "A5 (880.00 Hz)" if "A5 (880.00 Hz)" in piano_notes else next(iter(piano_notes))
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
def _(mo, f0):
    # Theorema de Shannon
    f_nyquist = 2 * f0
    mo.md(f"""
    ### Teorema de Shannon-Nyquist
    
    Para reconstruir perfeitamente um sinal bandlimitado à frequência máxima **f₀**, a frequência de amostragem deve ser:
    
    $$f_s \\geq 2 \\cdot f_0 = f_{{Nyquist}}$$
    
    **Para este tom:**
    - Frequência máxima (f₀): **{f0:.2f} Hz**
    - Frequência de Nyquist: **{f_nyquist:.2f} Hz**
    - Frequência mínima recomendada: **f_s ≥ {f_nyquist:.2f} Hz**
    
    Se a frequência de amostragem for menor que 2·f₀, ocorre **aliasing** (replicação espectral).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sinal no Domínio do Tempo
    """)
    return


@app.cell
def _(mo, T):
    periods_to_show = mo.ui.slider(1, 10, value=1, label="Períodos para mostrar:", show_value=True)
    return periods_to_show,


@app.cell
def _(Figure, T, periods_to_show, plt, t, x):
    # Plot selected periods for clarity
    mask_periods = t < periods_to_show.value * T
    fig_time = Figure(figsize=(12, 4))
    ax_time = fig_time.add_subplot(1, 1, 1)
    ax_time.plot(t[mask_periods] * 1e3, x[mask_periods], 'b-', linewidth=2)
    ax_time.set_xlabel('Tempo (ms)', fontsize=11)
    ax_time.set_ylabel('Amplitude', fontsize=11)
    ax_time.set_title(f'Sinal: {periods_to_show.value} Período(s) (Alta Densidade)', fontsize=12)
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
    sampling_freq = mo.ui.text(value="17600", label="Frequência de Amostragem (Hz):")
    
    mo.md(f"""
    {bits}
    
    {sampling_freq}
    """)
    return bits, sampling_freq


@app.cell
def _(f0, mo, sampling_freq):
    # Validação do Teorema de Shannon
    _fs_str = sampling_freq.value.strip()
    _fs_current = float(_fs_str) if _fs_str else 17600
    _f_nyquist_required = 2 * f0
    _satisfies_shannon = _fs_current >= _f_nyquist_required
    
    _status_msg = "✅ **SIM** - Teorema de Shannon satisfeito!" if _satisfies_shannon else "❌ **NÃO** - Violação do Teorema de Shannon (aliasing esperado)"
    
    mo.md(f"""
    #### Validação do Teorema de Shannon
    
    - Frequência de amostragem (f_s): **{_fs_current:.2f} Hz**
    - Frequência de Nyquist requerida (2·f₀): **{_f_nyquist_required:.2f} Hz**
    - Razão (f_s / 2·f₀): **{_fs_current / _f_nyquist_required:.2f}**
    - Status: {_status_msg}
    """)
    return


@app.cell
def _(A, bits, f0, np, sampling_freq, T):
    # Sampling and Quantization
    fs_str = sampling_freq.value.strip()
    fs_sampling = float(fs_str) if fs_str else 17600  # Default to 17600 if empty
    num_periods_sample = 1  # Show 1 period for clear visualization
    t_sample = np.arange(0, T * num_periods_sample, 1/fs_sampling)
    x_sample = A * np.sin(2 * np.pi * f0 * t_sample)
    
    levels = 2 ** bits.value
    delta = 2 * A / (levels - 1)  # For bipolar signal -A to A
    x_q = np.round(x_sample / delta) * delta
    noise = x_sample - x_q
    
    return delta, fs_sampling, levels, noise, num_periods_sample, t_sample, x_q, x_sample


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
def _(mo):
    mo.md(r"""
    ### ✨ Teorema de Shannon: Visualização Interativa
    
    **O que Shannon realmente diz:**
    
    Se você tem amostras de um sinal tomadas a frequência $f_s \geq 2f_0$, então é possível **reconstruir perfeitamente** o sinal original contínuo usando a **Fórmula de Interpolação Sinc**:
    
    $$x(t) = \sum_{n=-\infty}^{\infty} x[n] \cdot \text{sinc}\left(\frac{t - nT_s}{T_s}\right)$$
    
    onde $\text{sinc}(u) = \frac{\sin(\pi u)}{\pi u}$ e cada amostra quantizada contribui como uma onda suave que passa exatamente pelo ponto amostrado.
    
    **Observe no gráfico abaixo:**
    
    - **Azul tracejado fino** = O sinal original contínuo (sua "verdade" de referência)
    - **Vermelho sólido** = O sinal reconstruído usando Shannon a partir de amostras quantizadas
    - **Pontos verdes** = As amostras que levam a reconstrução
    - **Gráfico de erro** = Mostra a diferença entre original e reconstruído (mude bits para ver mudar!)
    
    **Como o número de bits afeta Shannon?**
    - Mais bits → Amostras mais precisas → Reconstrução mais fiel ao original
    - Menos bits → Mais "degraus" de quantização → Maior erro de reconstrução
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt):
    # Reconstrução via Interpolação Sinc
    _fs_recon = 17600  # Fixed high sampling rate for reconstruction
    _T_recon = 1 / _fs_recon
    
    # Gerar tempo contínuo para reconstrução
    _t_recon = np.linspace(0, 3 / f0, 5000)  # 3 períodos com alta resolução
    
    # Amostras (calculadas no tempo de amostragem)
    _t_samp_recon = np.arange(0, 3 / f0, 1 / _fs_recon)
    _x_samp_recon = np.sin(2 * np.pi * f0 * _t_samp_recon)
    
    # Quantizar as amostras com base no valor de bits
    _levels = 2 ** bits.value
    _delta = 2.0 / (_levels - 1)  # Para sinal bipolar -1 a 1
    _x_q_recon = np.round(_x_samp_recon / _delta) * _delta
    
    # Reconstruir usando Sinc (Fórmula de Interpolação de Shannon)
    _x_recon = np.zeros_like(_t_recon)
    for _n, _x_n in enumerate(_x_q_recon):
        _t_shifted = (_t_recon - _n * _T_recon) / _T_recon
        _sinc_vals = np.sinc(_t_shifted)  # NumPy sinc already has π factor
        _x_recon += _x_n * _sinc_vals
    
    # Sinal original contínuo para comparação
    _x_orig_recon = np.sin(2 * np.pi * f0 * _t_recon)
    
    # Plotar
    fig_recon = Figure(figsize=(14, 6))
    
    # Subplot 1: Sinais sobrepostos
    ax1 = fig_recon.add_subplot(1, 2, 1)
    ax1.plot(_t_recon * 1e3, _x_orig_recon, 'b:', linewidth=1, label='Sinal Original (referência)', alpha=0.6)
    ax1.plot(_t_recon * 1e3, _x_recon, 'r-', linewidth=2, label=f'Reconstruído (Sinc com {bits.value} bits)', alpha=0.8)
    ax1.plot(_t_samp_recon * 1e3, _x_q_recon, 'go', markersize=4, label='Amostras quantizadas')
    ax1.set_xlabel('Tempo (ms)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Reconstrução de Shannon ({bits.value} bits)', fontsize=12, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim([0, 3 / f0 * 1e3])
    
    # Subplot 2: Erro de reconstrução
    _error_recon = np.abs(_x_orig_recon - _x_recon)
    ax2 = fig_recon.add_subplot(1, 2, 2)
    ax2.semilogy(_t_recon * 1e3, _error_recon + 1e-15, 'r-', linewidth=2, label=f'Erro ({bits.value} bits)')
    ax2.set_xlabel('Tempo (ms)', fontsize=11)
    ax2.set_ylabel('Erro (escala log)', fontsize=11)
    ax2.set_title(f'Erro de Reconstrução ({bits.value} bits)', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim([0, 3 / f0 * 1e3])
    
    fig_recon.tight_layout()
    plt.close(fig_recon)
    fig_recon
    return


@app.cell
def _(mo, f0, bits):
    mo.md(f"""
    #### 🔍 Análise Interativa: Mude o slider de Bits
    
    **O que você está vendo:**
    
    1. **Linha azul tracejada** = Sinal original sem quantização (sua referência)
    2. **Linha vermelha sólida** = Sinal reconstruído via sinc a partir de amostras quantizadas com **{bits.value} bits**
    3. **Pontos verdes** = Amostras quantizadas (cada ponto é uma medição do ADC com {bits.value} bits)
    4. **Gráfico de erro** = Diferença em escala logarítmica
    
    **Teste agora:**
    - Reduza para **4 bits** → Veja o erro grande (azul vs vermelho divergem visualmente)
    - Aumente para **16 bits** → Vermelho fica sobre azul (erro tão pequeno que parecem iguais)
    
    **Por quê?**
    - Com poucos bits, cada amostra é um "degrau" discreto (quantização grosseira)
    - Shannon reconstrói perfeitamente a partir dos dados que tem
    - Mas se os dados são imprecisos, a reconstrução também é!
    
    **Conclusão**: Shannon diz "se suas amostras forem perfeitas, a reconstrução é perfeita". Mas quantização reduz precisão das amostras. Mais bits = amostras mais precisas = reconstrução mais fiel.
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt):
    # Visualizar várias frequências de amostragem com a fs_sampling interativa
    _fs_values = [f0, 1.5*f0, 2*f0, 3*f0, 4*f0]
    
    fig_shannon = Figure(figsize=(14, 6))
    
    # Calcular SNR teórico para quantização (SQNR = 6.02·N + 1.76 dB, onde N = bits)
    _sqnr_theoretical = 6.02 * bits.value + 1.76  # dB
    
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
        ax.plot(_t_dense * 1e3, _x_dense, 'b:', linewidth=1, alpha=0.4, label='Sinal original')
        ax.plot(_t_sample * 1e3, _x_quantized, 'ro', markersize=5, label=f'Amostras ({bits.value}b)')
        ax.plot(_t_sample * 1e3, _x_sample, 'b-', linewidth=0.5, alpha=0.3, label='Antes de quantizar')
        
        # Validar Shannon
        _f_nyquist_req = 2 * f0
        _status = "✓ Shannon OK" if _fs_test >= _f_nyquist_req else "✗ Aliasing"
        _color_title = 'green' if _fs_test >= _f_nyquist_req else 'red'
        
        ax.set_title(f'f_s = {_fs_test/f0:.1f}·f₀\n{_status}', 
                    fontsize=10, color=_color_title, weight='bold')
        ax.set_xlabel('Tempo (ms)', fontsize=9)
        ax.set_ylabel('Amplitude', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    fig_shannon.suptitle(f'Shannon com Quantização ({bits.value} bits, SQNR ≈ {_sqnr_theoretical:.1f} dB)', 
                        fontsize=12, weight='bold')
    fig_shannon.tight_layout()
    plt.close(fig_shannon)
    fig_shannon
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 📊 SNR e os Limites de Shannon
    
    **Fato importante**: O Teorema de Shannon garante reconstrução **perfeita** se os dados forem **perfeitos**. 
    
    Mas quantização introduz ruído! A **Signal-to-Noise Ratio (SNR)** é limitada pela resolução do ADC:
    
    $$\text{SQNR} = 6.02 \cdot N + 1.76 \text{ dB}$$
    
    onde $N$ é o número de bits. Cada bit adicional melhora SNR em ~6 dB.
    
    **O Problema**: Shannon reconstrói fielmente... mas a partir de amostras ruidosas! 
    - Com poucos bits → Reconstrução é fiel ao sinal quantizado (não ao original)
    - Com muitos bits → Reconstrução se aproxima do sinal original
    
    **Teste**: Mude o slider de bits e observe como o erro diminui. Shannon **não é melhor que os dados**!
    """)
    return


@app.cell
def _(Figure, bits, f0, np, plt):
    # SNR analysis
    fig_snr = Figure(figsize=(14, 5))
    
    # Gráfico 1: SQNR teórico vs Bits
    _ax1 = fig_snr.add_subplot(1, 2, 1)
    _bits_range = np.arange(4, 17)
    _sqnr_db = 6.02 * _bits_range + 1.76
    _ax1.plot(_bits_range, _sqnr_db, 'b-', linewidth=2, label='SQNR teórico')
    _ax1.plot(bits.value, 6.02 * bits.value + 1.76, 'ro', markersize=10, label=f'Atual: {bits.value} bits')
    _ax1.axhline(y=96, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Limite CD (16-bit)')
    _ax1.set_xlabel('Número de Bits', fontsize=11)
    _ax1.set_ylabel('SQNR (dB)', fontsize=11)
    _ax1.set_title('SNR Teórico vs Resolução do ADC', fontsize=12, weight='bold')
    _ax1.grid(True, alpha=0.3)
    _ax1.legend(fontsize=10)
    _ax1.set_xlim([4, 16])
    
    # Gráfico 2: Efeito prático - erro de reconstrução Shannon vs bits
    _ax2 = fig_snr.add_subplot(1, 2, 2)
    
    # Gerar reconstruções com diferentes bits
    _fs_recon_snr = 17600
    _T_recon_snr = 1 / _fs_recon_snr
    _t_recon_snr = np.linspace(0, 3/f0, 1000)
    _x_orig_snr = np.sin(2 * np.pi * f0 * _t_recon_snr)
    
    _t_samp_snr = np.arange(0, 3/f0, 1/_fs_recon_snr)
    _x_samp_snr = np.sin(2 * np.pi * f0 * _t_samp_snr)
    
    _error_by_bits = []
    _bits_test = np.arange(4, 17)
    
    for _b in _bits_test:
        _levels = 2 ** _b
        _delta = 2.0 / (_levels - 1)
        _x_q = np.round(_x_samp_snr / _delta) * _delta
        
        # Reconstruir
        _x_recon = np.zeros_like(_t_recon_snr)
        for _n, _x_n in enumerate(_x_q):
            _t_shifted = (_t_recon_snr - _n * _T_recon_snr) / _T_recon_snr
            _sinc_vals = np.sinc(_t_shifted)
            _x_recon += _x_n * _sinc_vals
        
        # Erro RMS
        _err_rms = np.sqrt(np.mean((_x_orig_snr - _x_recon)**2))
        _error_by_bits.append(_err_rms)
    
    _ax2.semilogy(_bits_test, _error_by_bits, 'r-o', linewidth=2, markersize=6, label='Erro RMS (Shannon)')
    _ax2.plot(bits.value, _error_by_bits[bits.value - 4], 'go', markersize=12, label=f'Atual: {bits.value} bits')
    _ax2.set_xlabel('Número de Bits', fontsize=11)
    _ax2.set_ylabel('Erro RMS (escala log)', fontsize=11)
    _ax2.set_title('Saturação de Shannon por Quantização', fontsize=12, weight='bold')
    _ax2.grid(True, alpha=0.3, which='both')
    _ax2.legend(fontsize=10)
    _ax2.set_xlim([4, 16])
    
    fig_snr.tight_layout()
    plt.close(fig_snr)
    fig_snr
    return


if __name__ == "__main__":
    app.run()

