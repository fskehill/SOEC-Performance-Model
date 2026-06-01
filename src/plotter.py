import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
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
    gs = gridspec.GridSpec(3, 3, figure=fig)


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
    ax1.text(0.5, 1.06,
             '- Jensen (2007): 850°C, 50% H₂O/50% H₂',
             color='#aaaaaa', fontsize=7, fontstyle='italic')
    if j_exp is not None and 'V_jensen' in results:
        ax1.scatter(j_exp, V_exp, color='white', s=50,
                    zorder=7, marker='o', label='Jensen (2007) exp.')
        ax1.errorbar(j_exp, V_exp,
                     yerr=[v * 0.011 for v in V_exp], 
                     fmt='none', color='white',
                     capsize=3, lw=1.0, alpha=0.6)
    make_legend(ax1, loc='upper right')
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
    ax2.set_title('2. Loss Breakdown at 800°C', fontsize=11, 
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
    ax3.set_title('4. Efficiency vs. Current', fontsize=11, 
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
    ax4.set_title('5. YSZ Resistance vs. Temperature', fontsize=11, 
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
    ax5.set_title('3. Gas Composition Along Cell', fontsize=11,
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
    ax6.text(0.1, 1.05,
             'OCV rises as H2O is consumed and H2 builds up',
             color='#909090', fontsize=7.5, fontstyle='italic')
    make_legend(ax6, loc='lower right')
    style(ax6)

    ax7 = fig.add_subplot(gs[2, 0])
    positions = np.linspace(0, 100, results['N_nodes'])
    for i in range(len(T_list)):
        T_plot = [T - 273.15 for T in results['T_profiles'][i]]
        ax7.plot(positions, T_plot,
                 color=COLORS[i], lw=2.4, label=T_labels[i])
    ax7.set_xlabel('Position along Cell (%)', fontsize=10)
    ax7.set_ylabel('Local Temperature (°C)', fontsize=10)
    ax7.set_title('7. Temperature Profile Along Cell',
                  fontsize=11, fontweight='bold', pad=8)
    ax7.text(1, min(T_plot) - 15,
             'Temperature evolves due to ohmic' \
             '\n heating vs reaction cooling',
             color='#909090', fontsize=7.5, fontstyle='italic')
    ax7.axhline(y=800, color=GOLD, ls=':', lw=1.0, alpha=0.5)
    ax7.text(2, 793, 'Thermoneutral Crossover',
             color=GOLD, fontsize=7, fontstyle='italic')
    make_legend(ax7, loc='upper right')
    style(ax7)


    ax8 = fig.add_subplot(gs[2, 1])
    if 'j_local_profiles' in results:
        for i in range(len(T_list)):
            ax8.plot(positions, results['j_local_profiles'][i], 
                     color=COLORS[i], lw=2.4, label=T_labels[i])
        ax8.text(2, max(results['j_local_profiles'][0]) * 1.01,
                 'Current drops as steam is depleted towards outlet', 
                 color='#909090', fontsize=7.5, fontstyle='italic')
    else:
        j_op = 0.5
        for i in range(len(T_list)):
            ax8.axhline(y=j_op, color=COLORS[i], lw=2.4, 
                        label=T_labels[i])  
    ax8.set_xlabel('Position along Cell (%)', fontsize=10)
    ax8.set_ylabel('Local Current Density (A/cm²)', fontsize=10)
    ax8.set_title('8. Local Current Density Profile', fontsize=11, 
                  fontweight='bold', pad=8)
    ax8.set_xlim(0, 100)
    make_legend(ax8, loc='upper right')
    style(ax8)


    ax9 = fig.add_subplot(gs[2, 2])
    j_hm = np.linspace(0.05, 1.4, 80)
    T_hm = np.linspace(700, 900, 80)
    J_grid, T_grid = np.meshgrid(j_hm, T_hm)

    R_GAS = 8.314
    B_ohm = 2.99e-5
    E_ohm = 8000.0
    ASR = B_ohm * np.exp(E_ohm / (R_GAS * (T_grid + 273.15))) * 1e4
    V_ocv_hm = 1.253 - 2.4516e-4 * (T_grid + 273.15)
    V_cell_hm = V_ocv_hm + J_grid * ASR * 0.4
    eff_hm = (V_thermo / V_cell_hm) * 100
    eff_hm = np.clip(eff_hm, 55, 140)

    cmap = LinearSegmentedColormap.from_list('soec', 
                                             ['#D94F3D', '#f0c040', '#34A876'], N=256)
    im = ax9.contourf(J_grid, T_grid, eff_hm,
                      levels=20, cmap=cmap)
    ax9.contour(J_grid, T_grid, eff_hm,
                levels=[100], colors=[GOLD], linewidths=1.5,
                linestyles='--')
    ax9.text(0.08, 810, 'Thermoneutral\n(100%)', 
             color=GOLD, fontsize=7, fontstyle='italic')
    cb = fig.colorbar(im, ax=ax9, pad=0.02)
    cb.ax.yaxis.set_tick_params(color=FG, labelsize=8)
    cb.outline.set_edgecolor(GRID_C)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG)
    cb.set_label('Efficiency (%)', color=FG, fontsize=10)
    for i, T_c in enumerate(T_degC):
        ax9.axhline(y=T_c, color=COLORS[i], lw=1.2,
                    ls='--', alpha=0.6, label=T_labels[i])
    ax9.set_xlabel('Current Density j (A/cm²)', fontsize=10)
    ax9.set_ylabel('Temperature (°C)', fontsize=10)
    ax9.set_title('9. Efficiency Map (T vs j)', fontsize=11, 
                  fontweight='bold', pad=8)
    ax9.tick_params(colors=FG, labelsize=9)
    ax9.xaxis.label.set_color(FG)
    ax9.yaxis.label.set_color(FG)
    ax9.title.set_color(FG)
    for spine in ax9.spines.values():
        spine.set_edgecolor(GRID_C)
    def _legend(ax, fontsize=8.5, **kwargs):
        leg = ax.legend(fontsize=fontsize, 
                        facecolor=AX_BG, 
                        edgecolor=GRID_C, 
                        labelcolor=FG, **kwargs)
        return leg


    fig.text(0.5, 0.945,
             'SOEC Cell Model | 1.5D Thermodynamics + Kinetics | Python Implementation',
             ha='center', fontsize=14, fontweight='bold', color=FG)
    fig.text(0.5, 0.913,
             'Nernst Eq  |  Butler-Volmer Kinetics  |  Fick Diffusion  |  Arrhenius Resistance',
             ha='center', fontsize=9, color='#888888')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'Figure saved to {save_path}')
    plt.close()