import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

COLORS = ['#2E75C0', '#34A876', '#D94F3D']
BG = '#1a1a1e'
AX_BG = '#222228'
FG = '#eaeaea'
GRID_C = '#3a3a42'
GOLD = '#f0c040'

def style(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.6)

def make_legend(ax, **kwargs):
    leg = ax.legend(fontsize=8.5, facecolor=AX_BG,
                    edgecolor=GRID_C, labelcolor=FG, **kwargs)
    return leg

def plot_results(j, results, T_list, T_labels, V_thermo, save_path, j_exp=None, V_exp=None):

    T_degC = [T - 273.15 for T in T_list]

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.subplots_adjust(hspace=0.40, wspace=0.30,
                        top=0.88, bottom=0.08,
                        left=0.07, right=0.97)
    gs = gridspec.GridSpec(2, 3, figure=fig)


    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(len(T_list)):
        ax1.plot(j, results['V_cell'][i],
                 color=COLORS[i], lw=2.4, label=T_labels[i])
    ax1.axhline(V_thermo, color=GOLD, ls='--', lw=1.5,
                label=f'V_thermo = {V_thermo} V')
    ax1.set_xlim(0, 1.6)
    ax1.set_ylim(0.85, 2.1)
    ax1.set_xlabel('Current Density j (A/cm²)', fontsize=10)
    ax1.set_ylabel('Cell Voltage V_cell (V)', fontsize=10)
    ax1.set_title('1. Polarisation I-V Curves', fontsize=11, 
                  fontweight='bold', pad=8)
    ax1.text(0.03, 0.90, 'Higher T = lower voltage = higher efficiency',
             color='#909090', fontsize=7.5, fontstyle='italic')
    if j_exp is not None and 'V_jensen' in results:
        ax1.plot(j, results['V_jensen'], color='white',
                 lw=2, ls=':', alpha=0.6, label='Model @ Jensen conditions')
    make_legend(ax1, loc='upper left')
    style(ax1)


    ax2 = fig.add_subplot(gs[0, 1])
    t_ref = 1
    jo = results['eta_ohm'][t_ref]
    ja = results['eta_act'][t_ref]
    jc = np.nan_to_num(results['eta_con'][t_ref], nan=0)
    ax2.fill_between(j, 0, jo, color='#E8A030',
                     alpha=0.85, label='Ohmic (YSZ)')
    ax2.fill_between(j, jo, jo+ja, color='#4A9EE0',
                     alpha=0.85, label='Activation (BV)')
    ax2.fill_between(j, jo+ja, jo+ja+jc, color='#D94F3D',
                     alpha=0.85, label='Concentration (Fick)')
    ax2.set_xlim(0, 1.4)
    ax2.set_ylim(0, 0.75)
    ax2.set_xlabel('Current Density j (A/cm²)', fontsize=10)
    ax2.set_ylabel('Overpotential η (V)', fontsize=10)
    ax2.set_title('3. Loss Breakdown at 800°C', fontsize=11, 
                  fontweight='bold', pad=8)
    make_legend(ax2, loc='upper left')
    style(ax2)


    ax3 = fig.add_subplot(gs[1, 0])
    for i in range(len(T_list)):
        ax3.plot(j, results['eff'][i],
                 color=COLORS[i], lw=2.4, label=T_labels[i])
    ax3.axhline(100, color=GOLD, ls='--', lw=1.5,
                label='100% Thermoneutral')
    ax3.fill_between(j, 100, 140, color='#34A876', alpha=0.06)
    ax3.text(0.03, 57,
             'Electrothermal: cell absorbs heat from surroundings',
             color='#909090', fontsize=7.5, fontstyle='italic')
    ax3.set_xlim(0, 1.4)
    ax3.set_ylim(55, 140)
    ax3.set_xlabel('Current Density j (A/cm²)', fontsize=10)
    ax3.set_ylabel('Energy Efficiency (%)', fontsize=10)
    ax3.set_title('2. Efficiency vs. Current', fontsize=11, 
                  fontweight='bold', pad=8)
    make_legend(ax3, loc='upper right')
    style(ax3)


    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(results['T_range'] - 273.15, results['ASR_range'],
             color='#8888dd', lw=2.4)
    for i in range(len(T_list)):
        ax4.scatter(T_degC[i], results['ASR_pts'][i],
                    color=COLORS[i], s=90, zorder=5,
                    edgecolors='white', linewidths=0.8,
                    label=T_labels[i])
    ax4.set_xlabel('Temperature (°C)', fontsize=10)
    ax4.set_ylabel('ASR (Ω·cm²)', fontsize=10)
    ax4.set_title('4. YSZ Resistance vs. Temperature', fontsize=11, 
                  fontweight='bold', pad=8)
    make_legend(ax4, loc='upper right')
    style(ax4)


    ax5 = fig.add_subplot(gs[0, 2])
    positions = np.linspace(0, 100, results['N_nodes'])
    for i in range(len(T_list)):
        ax5.plot(positions, [x * 100 for x in results['x_H2O_profiles'][i]],
                 color=COLORS[i], lw=2.0, ls='--', label=f'H2O {T_labels[i]}')
        ax5.plot(positions, [x * 100 for x in results['x_H2_profiles'][i]],
                 color=COLORS[i], lw=2.0, ls='-', label=f'H2 {T_labels[i]}')
    ax5.set_xlabel('Position along Cell (%)', fontsize=10)
    ax5.set_ylabel('Mole Fraction (%)', fontsize=10)
    ax5.set_title('5. Gas Composition Along Cell', fontsize=11,
                  fontweight='bold', pad=8)
    ax5.text(2, 18, 'Solid = H2   Dashed = H2O',
             color='#909090', fontsize=7.5, fontstyle='italic')
    make_legend(ax5, loc='center right')
    style(ax5)


    ax6 = fig.add_subplot(gs[1, 2])
    positions = np.linspace(0, 100, results['N_nodes'])
    for i in range(len(T_list)):
        ax6.plot(positions, results['V_ocv_profiles'][i],
                 color=COLORS[i], lw=2.4, label=T_labels[i])
    ax6.set_xlabel('Position along Cell (%)', fontsize=10)
    ax6.set_ylabel('Local OCV (V)', fontsize=10)
    ax6.set_title('6. OCV Profile Along Cell', fontsize=11,
                  fontweight='bold', pad=8)
    ax6.text(0.1, 1.07,
             'OCV rises as H2O is consumed and H2 builds up',
             color='#909090', fontsize=7.5, fontstyle='italic')
    make_legend(ax6, loc='lower right')
    style(ax6)


    fig.text(0.5, 0.945,
             'SOEC Cell Model | 1D Thermodynamics + Kinetics | Python Implementation',
             ha='center', fontsize=14, fontweight='bold', color=FG)
    fig.text(0.5, 0.913,
             'Nernst Eq  |  Butler-Volmer Kinetics  |  Fick Diffusion  |  Arrhenius Resistance',
             ha='center', fontsize=9, color='#888888')
    
    fig.add_artist(plt.Line2D(
        [0.672, 0.672], [0.08, 0.88],
        transform=fig.transFigure,
        color='#555566', lw=1.2, ls='--'
    ))
    fig.text(0.65, 0.895, '<-- 0D Model', fontsize=8, color='#888888')
    fig.text(0.65, 0.878, '1D Extension -->', fontsize=8, color='#888888')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'Figure saved to {save_path}')
    plt.close()