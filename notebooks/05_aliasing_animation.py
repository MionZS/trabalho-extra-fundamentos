"""
Animação de Aliasing - Notebook Interativo com Controle Temporal

Este notebook demonstra o efeito de aliasing através de uma animação
que varia a taxa de amostragem ao longo do tempo.

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
    import time
    return Figure, mo, np, plt, time


@app.cell
def __(mo):
    mo.md(
        r"""
        # 🎬 Animação de Aliasing

        ## Introdução

        Este notebook demonstra **dinamicamente** o efeito de aliasing variando
        a taxa de amostragem ao longo do tempo. Você verá em tempo real como
        o sinal amostrado muda quando cruzamos a fronteira de Nyquist.

        ### Conceito da Animação

        A animação varia **fs (taxa de amostragem)** automaticamente de um valor
        inicial até um valor final, passando pela **taxa de Nyquist (2f)**.

        Você observará:
        - ✅ **Antes de 2f:** Aliasing severo
        - ⚠️ **Próximo a 2f:** Transição crítica
        - ✅ **Depois de 2f:** Amostragem correta

        ### Controles

        - **▶️ Play:** Inicia a animação
        - **⏸️ Stop:** Pausa a animação
        - **🔄 Reset:** Volta ao estado inicial
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## 🎛️ Controles da Animação""")
    return


@app.cell
def __(mo):
    # Frequência fixa do sinal
    f_signal_anim = mo.ui.slider(
        start=500,
        stop=3000,
        step=100,
        value=1000,
        label="Frequência do Sinal (f) [Hz]",
        show_value=True
    )

    # fs inicial (abaixo de Nyquist)
    fs_start = mo.ui.number(
        start=500,
        stop=10000,
        step=100,
        value=1000,
        label="fs Inicial [Hz] (abaixo de 2f)"
    )

    # fs final (acima de Nyquist)
    fs_end = mo.ui.number(
        start=1000,
        stop=15000,
        step=100,
        value=6000,
        label="fs Final [Hz] (acima de 2f)"
    )

    # Duração total da animação
    duration_total = mo.ui.number(
        start=5,
        stop=30,
        step=1,
        value=10,
        label="Duração Total [s]"
    )

    # Intervalo entre frames
    delta_t_frame = mo.ui.number(
        start=0.1,
        stop=2.0,
        step=0.1,
        value=0.5,
        label="Intervalo entre Frames [s]"
    )

    # Janela de tempo para visualização
    window_anim_ms = mo.ui.number(
        start=5,
        stop=50,
        step=5,
        value=20,
        label="Janela de Tempo [ms]"
    )

    mo.md(
        f"""
        ### Parâmetros do Sinal

        {f_signal_anim}

        ### Parâmetros da Animação

        {fs_start}

        {fs_end}

        {duration_total}

        {delta_t_frame}

        {window_anim_ms}
        """
    )
    return (
        delta_t_frame,
        duration_total,
        f_signal_anim,
        fs_end,
        fs_start,
        window_anim_ms,
    )


@app.cell
def __(mo):
    # Botões de controle
    play_button = mo.ui.run_button(label="▶️ Play Animation")
    stop_button = mo.ui.button(label="⏸️ Stop")
    reset_button_anim = mo.ui.button(label="🔄 Reset")

    mo.md(
        f"""
        ### Controles

        {play_button} {stop_button} {reset_button_anim}
        """
    )
    return play_button, reset_button_anim, stop_button


@app.cell
def __(mo):
    # Estado da animação: contador de frames e flag de parada
    animation_state = mo.state({
        'frame': 0,
        'running': False,
        'fs_current': 0,
        'completed': False
    })
    return


@app.cell
def __(
    animation_state,
    delta_t_frame,
    duration_total,
    f_signal_anim,
    fs_end,
    fs_start,
    mo,
    np,
    play_button,
    reset_button_anim,
    stop_button,
    time,
):
    # Lógica da animação
    if reset_button_anim.value:
        animation_state.value = {
            'frame': 0,
            'running': False,
            'fs_current': fs_start.value,
            'completed': False
        }

    if stop_button.value:
        state_copy_stop = animation_state.value.copy()
        state_copy_stop['running'] = False
        animation_state.value = state_copy_stop

    # Executar animação quando Play é pressionado
    if play_button.value and not animation_state.value['completed']:
        # Resetar se for uma nova execução
        if not animation_state.value['running']:
            animation_state.value = {
                'frame': 0,
                'running': True,
                'fs_current': fs_start.value,
                'completed': False
            }

        # Parâmetros da animação
        f_sig = f_signal_anim.value
        fs_ini = fs_start.value
        fs_fim = fs_end.value
        T_total = duration_total.value
        dt_frame = delta_t_frame.value
        
        num_frames = int(T_total / dt_frame)
        
        # Gerar sequência de fs
        fs_sequence = np.linspace(fs_ini, fs_fim, num_frames)
        
        current_frame = animation_state.value['frame']
        
        if current_frame < num_frames and animation_state.value['running']:
            # Atualizar fs atual
            fs_current_val = fs_sequence[current_frame]
            
            # Aguardar o intervalo de tempo
            time.sleep(dt_frame)
            
            # Atualizar estado
            state_copy = animation_state.value.copy()
            state_copy['frame'] = current_frame + 1
            state_copy['fs_current'] = fs_current_val
            
            if current_frame + 1 >= num_frames:
                state_copy['running'] = False
                state_copy['completed'] = True
            
            animation_state.value = state_copy
        
        fs_anim = fs_current_val
        frame_number = current_frame + 1
    else:
        # Modo estático: usar valores iniciais
        fs_anim = fs_start.value if not animation_state.value['running'] else animation_state.value['fs_current']
        frame_number = animation_state.value['frame']
        f_sig = f_signal_anim.value
    return (
        T_total,
        current_frame,
        dt_frame,
        f_sig,
        frame_number,
        fs_anim,
        fs_current_val,
        fs_fim,
        fs_ini,
        fs_sequence,
        num_frames,
        state_copy,
        state_copy_stop,
    )


@app.cell
def __(animation_state, f_sig, frame_number, fs_anim, mo, num_frames):
    # Status da animação
    is_running = animation_state.value['running']
    is_completed = animation_state.value['completed']
    f_nyquist_anim = 2 * f_sig
    nyquist_ok_anim = fs_anim > f_nyquist_anim
    
    progress_pct = (frame_number / num_frames * 100) if num_frames > 0 else 0

    status_emoji_anim = "▶️" if is_running else ("✅" if is_completed else "⏸️")
    status_text_anim = "Rodando..." if is_running else ("Completo!" if is_completed else "Parado")

    mo.md(
        f"""
        ## 📊 Status da Animação

        **Status:** {status_emoji_anim} {status_text_anim}

        **Frame Atual:** {frame_number} / {num_frames} ({progress_pct:.1f}%)

        **fs Atual:** {fs_anim:.0f} Hz

        **Taxa de Nyquist (2f):** {f_nyquist_anim} Hz

        **Condição de Nyquist:** {"✅ Satisfeita" if nyquist_ok_anim else "⚠️ Violada (Aliasing)"}

        ---

        {mo.callout(
            f"**Amostragem Adequada:** fs ({fs_anim:.0f} Hz) > 2f ({f_nyquist_anim} Hz)",
            kind="success"
        ) if nyquist_ok_anim else mo.callout(
            f"**ALIASING ATIVO:** fs ({fs_anim:.0f} Hz) ≤ 2f ({f_nyquist_anim} Hz)",
            kind="warn"
        )}
        """
    )
    return (
        f_nyquist_anim,
        is_completed,
        is_running,
        nyquist_ok_anim,
        progress_pct,
        status_emoji_anim,
        status_text_anim,
    )


@app.cell
def __(mo):
    mo.md(r"""## 📈 Visualização Animada""")
    return


@app.cell
def __(f_sig, fs_anim, np, window_anim_ms):
    # Geração do sinal para o frame atual
    T_window_anim = window_anim_ms.value * 1e-3
    
    # Sinal contínuo de alta resolução
    fs_continuous_anim = max(f_sig * 50, 50000)
    t_cont_anim = np.arange(0, T_window_anim, 1/fs_continuous_anim)
    signal_cont_anim = np.sin(2 * np.pi * f_sig * t_cont_anim)
    
    # Sinal amostrado no fs atual
    if fs_anim > 0:
        t_samp_anim = np.arange(0, T_window_anim, 1/fs_anim)
        signal_samp_anim = np.sin(2 * np.pi * f_sig * t_samp_anim)
    else:
        t_samp_anim = np.array([])
        signal_samp_anim = np.array([])
    return (
        T_window_anim,
        fs_continuous_anim,
        signal_cont_anim,
        signal_samp_anim,
        t_cont_anim,
        t_samp_anim,
    )


@app.cell
def __(
    Figure,
    f_nyquist_anim,
    f_sig,
    frame_number,
    fs_anim,
    nyquist_ok_anim,
    plt,
    signal_cont_anim,
    signal_samp_anim,
    t_cont_anim,
    t_samp_anim,
):
    # Gráfico animado
    fig_anim = Figure(figsize=(14, 10))
    
    # Sinal no tempo
    ax1_anim = fig_anim.add_subplot(3, 1, 1)
    ax1_anim.plot(t_cont_anim * 1000, signal_cont_anim, 'b-', linewidth=1.5, 
                  label=f'Sinal Original ({f_sig} Hz)', alpha=0.7)
    
    if len(t_samp_anim) > 0:
        ax1_anim.plot(t_samp_anim * 1000, signal_samp_anim, 'ro', markersize=8, 
                      label=f'Amostras (fs={fs_anim:.0f} Hz)', zorder=5)
        ax1_anim.stem(t_samp_anim * 1000, signal_samp_anim, linefmt='r-', 
                      markerfmt='ro', basefmt='k-', alpha=0.4)
    
    ax1_anim.set_ylabel('Amplitude', fontsize=11)
    ax1_anim.set_title(f'Frame {frame_number} - Amostragem em fs = {fs_anim:.0f} Hz', 
                       fontsize=13, fontweight='bold')
    ax1_anim.grid(True, alpha=0.3)
    ax1_anim.legend(loc='upper right', fontsize=10)
    ax1_anim.set_xlim([0, t_cont_anim[-1] * 1000])
    ax1_anim.set_ylim([-1.3, 1.3])
    
    # Adicionar indicador visual de Nyquist
    if nyquist_ok_anim:
        ax1_anim.set_facecolor('#e8f5e9')  # Verde claro
    else:
        ax1_anim.set_facecolor('#ffebee')  # Vermelho claro

    # Reconstrução (interpolação linear simples para visualização)
    ax2_anim = fig_anim.add_subplot(3, 1, 2)
    ax2_anim.plot(t_cont_anim * 1000, signal_cont_anim, 'b--', linewidth=1.5, 
                  label='Original', alpha=0.5)
    
    if len(t_samp_anim) > 1:
        # Interpolação linear
        signal_recon_anim = np.interp(t_cont_anim, t_samp_anim, signal_samp_anim)
        ax2_anim.plot(t_cont_anim * 1000, signal_recon_anim, 'g-', linewidth=2, 
                      label='Reconstruído (linear)', alpha=0.8)
        ax2_anim.plot(t_samp_anim * 1000, signal_samp_anim, 'ro', markersize=6)
    
    ax2_anim.set_ylabel('Amplitude', fontsize=11)
    ax2_anim.set_title('Reconstrução do Sinal', fontsize=13, fontweight='bold')
    ax2_anim.grid(True, alpha=0.3)
    ax2_anim.legend(loc='upper right', fontsize=10)
    ax2_anim.set_xlim([0, t_cont_anim[-1] * 1000])
    ax2_anim.set_ylim([-1.3, 1.3])

    # Gráfico de progresso da taxa de amostragem
    ax3_anim = fig_anim.add_subplot(3, 1, 3)
    ax3_anim.axhline(y=f_nyquist_anim, color='orange', linestyle='--', linewidth=3, 
                     label=f'Taxa de Nyquist (2f = {f_nyquist_anim} Hz)', alpha=0.8)
    ax3_anim.axhline(y=fs_anim, color='red', linestyle='-', linewidth=3, 
                     label=f'fs Atual = {fs_anim:.0f} Hz', alpha=0.8)
    
    # Zona de aliasing
    ax3_anim.axhspan(0, f_nyquist_anim, alpha=0.2, color='red', 
                     label='Zona de Aliasing (fs < 2f)')
    ax3_anim.axhspan(f_nyquist_anim, f_nyquist_anim * 3, alpha=0.2, color='green', 
                     label='Zona Segura (fs > 2f)')
    
    ax3_anim.set_xlabel('', fontsize=11)
    ax3_anim.set_ylabel('Frequência [Hz]', fontsize=11)
    ax3_anim.set_title('Posição Atual da Taxa de Amostragem', fontsize=13, fontweight='bold')
    ax3_anim.set_xlim([0, 1])
    ax3_anim.set_ylim([0, f_nyquist_anim * 3])
    ax3_anim.legend(loc='upper right', fontsize=10)
    ax3_anim.grid(True, alpha=0.3, axis='y')
    ax3_anim.set_xticks([])

    fig_anim.tight_layout()
    plt.close(fig_anim)
    fig_anim
    return (
        ax1_anim,
        ax2_anim,
        ax3_anim,
        fig_anim,
        signal_recon_anim,
    )


@app.cell
def __(fs_anim, fs_continuous_anim, np, signal_samp_anim):
    # FFT do sinal amostrado (para visualização espectral opcional)
    N_samp_anim = len(signal_samp_anim)
    if N_samp_anim > 1:
        fft_samp_anim = np.fft.fft(signal_samp_anim)
        fft_samp_mag_anim = np.abs(fft_samp_anim) / N_samp_anim
        fft_samp_mag_anim = fft_samp_mag_anim[:N_samp_anim//2]
        freqs_samp_anim = np.fft.fftfreq(N_samp_anim, 1/fs_anim)
        freqs_samp_anim = freqs_samp_anim[:N_samp_anim//2]
    else:
        fft_samp_mag_anim = np.array([])
        freqs_samp_anim = np.array([])
    return (
        N_samp_anim,
        fft_samp_anim,
        fft_samp_mag_anim,
        freqs_samp_anim,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 🌊 Visualização Espectral (Opcional)

        Abaixo, o espectro do sinal amostrado no frame atual.
        """
    )
    return


@app.cell
def __(
    Figure,
    f_nyquist_anim,
    f_sig,
    fft_samp_mag_anim,
    freqs_samp_anim,
    fs_anim,
    plt,
):
    # Gráfico espectral
    fig_spectrum_anim = Figure(figsize=(12, 5))
    ax_spec_anim = fig_spectrum_anim.add_subplot(1, 1, 1)
    
    if len(freqs_samp_anim) > 0:
        ax_spec_anim.plot(freqs_samp_anim, fft_samp_mag_anim, 'b-', 
                          linewidth=1.5, marker='o', markersize=4, label='Espectro Amostrado')
    
    ax_spec_anim.axvline(x=f_sig, color='r', linestyle='--', linewidth=2, 
                         alpha=0.7, label=f'Freq. Original ({f_sig} Hz)')
    ax_spec_anim.axvline(x=fs_anim/2, color='orange', linestyle='--', linewidth=2, 
                         alpha=0.7, label=f'fs/2 = {fs_anim/2:.0f} Hz (Nyquist)')
    
    # Zona de aliasing
    ax_spec_anim.axvspan(fs_anim/2, fs_anim, alpha=0.2, color='red')
    
    ax_spec_anim.set_xlabel('Frequência [Hz]', fontsize=11)
    ax_spec_anim.set_ylabel('Magnitude', fontsize=11)
    ax_spec_anim.set_title(f'Espectro do Sinal Amostrado (fs = {fs_anim:.0f} Hz)', 
                           fontsize=13, fontweight='bold')
    ax_spec_anim.grid(True, alpha=0.3)
    ax_spec_anim.legend(loc='upper right')
    ax_spec_anim.set_xlim([0, fs_anim])
    
    fig_spectrum_anim.tight_layout()
    plt.close(fig_spectrum_anim)
    fig_spectrum_anim
    return ax_spec_anim, fig_spectrum_anim


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 📚 Notas Pedagógicas - Animação de Aliasing

        ### Objetivo da Animação

        Esta animação permite **visualizar dinamicamente** a transição entre
        amostragem inadequada (com aliasing) e amostragem adequada (sem aliasing).

        ### O que você está vendo:

        1. **Gráfico 1 (Sinal no Tempo):**
           - Linha azul: sinal original contínuo
           - Pontos vermelhos: amostras capturadas em fs
           - Fundo verde: fs > 2f (sem aliasing)
           - Fundo vermelho: fs ≤ 2f (com aliasing)

        2. **Gráfico 2 (Reconstrução):**
           - Linha verde: sinal reconstruído a partir das amostras
           - Note como a reconstrução melhora quando fs cruza 2f

        3. **Gráfico 3 (Indicador de Taxa):**
           - Mostra visualmente se fs está acima ou abaixo da taxa de Nyquist
           - Zona vermelha: aliasing ativo
           - Zona verde: amostragem adequada

        4. **Gráfico 4 (Espectro):**
           - Mostra as componentes de frequência do sinal amostrado
           - Observe as réplicas espectrais e possível aliasing

        ### Fenômenos a Observar:

        **Fase 1: fs < 2f (Aliasing)**
        - As amostras capturam o sinal em pontos esparsos
        - A reconstrução linear cria um sinal de **frequência aparente mais baixa**
        - No espectro, a frequência original "se dobra" para dentro da banda base

        **Fase 2: fs ≈ 2f (Transição Crítica)**
        - A fronteira de Nyquist
        - Pequenas variações em fs causam grandes mudanças na reconstrução
        - Momento mais instrutivo da animação!

        **Fase 3: fs > 2f (Amostragem Adequada)**
        - Amostras suficientes para capturar todas as oscilações
        - Reconstrução aproxima-se do sinal original
        - Espectro mostra separação clara entre banda base e réplicas

        ### Experimentos Sugeridos:

        1. **Transição Lenta:**
           - Configure: fs_inicial=1000, fs_final=6000, duração=15s, Δt=0.5s
           - Observe cuidadosamente o momento em que fs cruza 2f

        2. **Frequência Alta:**
           - Configure: f=2000 Hz (Nyquist=4000 Hz)
           - Veja como é necessário fs muito maior

        3. **Janela Longa:**
           - Aumente "Janela de Tempo" para 30-50ms
           - Observe múltiplos ciclos do sinal

        4. **Velocidade Rápida:**
           - Reduza Δt para 0.2s para animação mais rápida
           - Útil para demonstrações em sala

        ### Aplicações Práticas:

        - **Design de ADCs:** Escolha de fs adequado
        - **Compressão de Audio:** Trade-off entre qualidade e taxa de bits
        - **Processamento de Imagens:** Resolução espacial vs aliasing
        - **Comunicações:** Largura de banda do canal vs taxa de símbolo

        ### Limitações da Visualização:

        - Reconstrução usa interpolação **linear**, não o filtro sinc ideal
        - Número finito de amostras (janela truncada)
        - Resolução espectral limitada pelo número de pontos

        ---

        **Dica Pedagógica:** Execute a animação múltiplas vezes com diferentes
        parâmetros para desenvolver intuição sobre o teorema de Nyquist.

        ---

        **Referências:**
        - Shannon, C.E. (1949) "Communication in the Presence of Noise"
        - Oppenheim & Schafer "Discrete-Time Signal Processing"
        - Lyons, R. "Understanding Digital Signal Processing"
        """
    )
    return


@app.cell
def __():
    # Informações sobre o notebook
    __notebook_info_anim__ = {
        "title": "Animação de Aliasing",
        "version": "1.0",
        "date": "2025-10-30",
        "python": "3.14+",
        "dependencies": ["marimo", "numpy>=2.0", "matplotlib>=3.9"],
        "features": ["animation", "interactive_control", "real_time_visualization"]
    }
    return


if __name__ == "__main__":
    app.run()
