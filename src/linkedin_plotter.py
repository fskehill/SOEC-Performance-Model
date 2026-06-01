import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

COLORS = ['#2E75C0', '#34A876', '#D94F3D']
BG = '#1a1a1e'
AX_BG = '#222228'
FG = '#eaeaea'
GRID_C = '#3a3a42'
GOLD = '#f0c040'
ACCENT = '#7eb8f7'

def _style(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.grid(True, color=GRID_C, alpha=0.5, linewidth=0.6)

def _legend(ax, **kw):
    kw.setdefault('fontsize', 8)
    return ax.legend(facecolor=AX_BG,
                     edgecolor=GRID_C, labelcolor=FG, **kw)

def plot_linkedin(j, results, T_list, T_labels, V_thermo, save_path,
                  j_exp=None, V_exp=None):
    
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    fig.subplots_adjust(hspace=0.42, wspace=0.32,
                        top=0.87, bottom=0.08,
                        left=0.07, right=0.97)
    
    gs = gridspec.GridSpec(2, 2, figure=fig)

    T_degC = [T - 273.15 for T in T_list]

    ax7 = fig.add_subplot(gs[0, 0])
    positions = np.linspace(0, 100, results['N_nodes'])
    for i in range(len(T_list)):
        T_plot = [T - 273.15 for T in results['T_profiles'][i]]
        ax7.plot(positions, T_plot,
                 color=COLORS[i], lw=2.4, label=T_labels[i])
    ax7.set_xlabel('Position along Cell (%)', fontsize=10)
    ax7.set_ylabel('Local Temperature (°C)', fontsize=10)
    ax7.set_title('1. Temperature Profile Along Cell',
                  fontsize=11, fontweight='bold', pad=8)
    ax7.text(1, min(T_plot) - 15,
             'Temperature evolves due to ohmic' \
             '\n heating vs reaction cooling',
             color='#909090', fontsize=7.5, fontstyle='italic')
    ax7.axhline(y=800, color=GOLD, ls=':', lw=1.0, alpha=0.5)
    ax7.text(2, 793, 'Thermoneutral Crossover',
             color=GOLD, fontsize=7, fontstyle='italic')
    _legend(ax7, loc='upper right')
    _style(ax7)


    ax8 = fig.add_subplot(gs[0, 1])
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
    ax8.set_title('2. Local Current Density Profile', fontsize=11, 
                  fontweight='bold', pad=8)
    ax8.set_xlim(0, 100)
    _legend(ax8, loc='upper right')
    _style(ax8)


    ax9 = fig.add_subplot(gs[1, 0])
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
    ax9.set_title('3. Efficiency Map (T vs j)', fontsize=11, 
                  fontweight='bold', pad=8)
    ax9.tick_params(colors=FG, labelsize=9)
    ax9.xaxis.label.set_color(FG)
    ax9.yaxis.label.set_color(FG)
    ax9.title.set_color(FG)
    for spine in ax9.spines.values():
        spine.set_edgecolor(GRID_C)
    _legend(ax9, loc='lower left', fontsize=7.5)


    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(AX_BG)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('4. 1D → 1.5D: What Changed?', fontsize=11,
                  fontweight='bold', pad=8, color=FG)
    
    col0_x, col1_x, col2_x = 0.02, 0.38, 0.70
    header_y = 0.91

    def txt(ax, x, y, s, **kw):
        kw.setdefault('color', FG)
        kw.setdefault('fontsize', 8.5)
        ax.text(x, y, s, transform=ax.transAxes,
                va='top', **kw)
        
    ax4.text(col1_x, 0.96, '1D Model', transform=ax4.transAxes,
             color=GOLD, fontsize=9, fontweight='bold', va='top', ha='center')
    ax4.text(col2_x + 0.14, 0.96, '1.5D Model', transform=ax4.transAxes,
             color=ACCENT, fontsize=9, fontweight='bold', va='top', ha='center')
    
    ax4.axhline(0.87, color=GRID_C, lw=0.8, xmin=0.0, xmax=1.0)

    rows = [
        ('Spatial Resolution',  '20 Nodes Along (Flow)',  '20 x Layers (Through-Plane)'),
        ('Through-Plane T',     'Not Modelled',           'Resolved Anode → Cathode'),
        ('Thermal Coupling',    'Single T per Node',      'T Gradient Per Node'),
        ('Heat Generation',     'Global Ohmic Term',      'Layer-by-Layer Ohmic + React.'),
        ('Gas Diffusion',       'Fick (1D Flow)',         'Fick + Through-Plane GDL'),
        ('ASR Distribution',    'One Value per Node',     'Split: Electrolyte + Electrodes'),
        ('Computation Cost',    'Fast (~ms)',             'Moderate (~s)'),
    ]

    row_h = 0.82 / len(rows)
    tick_0 = '\u2715'
    tick_1 = '\u2713'

    for k, (feature, d0, d1) in enumerate(rows):
        y = header_y - 0.06 - k * row_h

        if k % 2 == 0:
            rect = FancyBboxPatch((0, y - row_h * 0.55), 1, row_h * 0.95,
                                  boxstyle='square,pad=0',
                                  linewidth=0, facecolor='#2a2a32',
                                  transform=ax4.transAxes, zorder=0)
            ax4.add_patch(rect)

        txt(ax4, col0_x, y, feature, color='#bbbbbb', fontsize=8)

        c0 = '#cc5555' if 'Not' in d0 or d0 == '-' else '#aaaaaa'
        txt(ax4, col1_x, y, d0, color=c0, fontsize=7.5, ha='center')

        txt(ax4, col2_x, y, d1, color='#55cc88', fontsize=7.5)

    for spine in ax4.spines.values():
        spine.set_edgecolor(GRID_C)


    fig.text(0.5, 0.945,
             'SOEC Cell Model  |  1.5D Extension  |  Python Implementation',
             ha='center', fontsize=13, fontweight='bold', color=FG)
    fig.text(0.5, 0.913,
             'Nernst Eq  ·  Butler-Volmer Kinetics  ·  Fick Diffusion  ·  Arrhenius Resistance  ·  1D Flow Path',
             ha='center', fontsize=8.5, color='#888888')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'LinkedIn Figure Saved → {save_path}')
    plt.close()