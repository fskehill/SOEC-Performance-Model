import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


    ax1 = fig.add_subplot(gs[0, 0])
    for i, lbl in enumerate(T_labels):
        ax1.plot(j, results['V_cell'][i],
                 color=COLORS[i], lw=2.4, label=lbl)
    ax1.axhline(V_thermo, color=GOLD, ls='--', lw=1.4,
                label=f'V_thermo = {V_thermo} V')
    if j_exp is not None and 'V_jensen' in results:
        ax1.scatter(j_exp, V_exp, color='white', s=50,
                    zorder=7, marker='o', label='Jensen (2007) exp.')
    ax1.set_xlim(0, 1.6); ax1.set_ylim(0.85, 2.1)
    ax1.set_xlabel('Current Density j (A/cm²)', fontsize=10)
    ax1.set_ylabel('Cell Voltage V_cell (V)', fontsize=10)
    ax1.set_title('1. Polarisation I-V Curves', fontsize=11, 
                  fontweight='bold', pad=8)
    ax1.text(0.03, 0.90, 'Higher T → lower voltage → higher efficiency',
             color='#909090', fontsize=7.5, fontstyle='italic')
    ax1.text(0.5, 1.04,
             '- Jensen (2007): 850°C, 50% H₂O/50% H₂',
             color='#aaaaaa', fontsize=7, fontstyle='italic')
    _legend(ax1, loc='upper left')
    _style(ax1)


    ax2 = fig.add_subplot(gs[0, 1])
    positions = np.linspace(0, 100, results['N_nodes'])
    for i, lbl in enumerate(T_labels):
        ax2.plot(positions,
                 [x * 100 for x in results['x_H2O_profiles'][i]],
                 color=COLORS[i], lw=2.0, ls='--', label=f'H₂O {lbl}')
        ax2.plot(positions,
                 [x * 100 for x in results['x_H2_profiles'][i]],
                 color=COLORS[i], lw=2.0, ls='-', label=f'H₂ {lbl}')
    ax2.set_xlabel('Position along Cell (%)', fontsize=10)
    ax2.set_ylabel('Mole Fraction (%)', fontsize=10)
    ax2.set_title('3. Gas Composition Profiles', fontsize=11, 
                  fontweight='bold', pad=8)
    ax2.text(2, 18, 'Solid = H₂  |  Dashed = H₂O',
             color='#909090', fontsize=7.5, fontstyle='italic')
    _legend(ax2, loc='center right', fontsize=7.5)
    _style(ax2)


    ax3 = fig.add_subplot(gs[1, 0])
    for i, lbl in enumerate(T_labels):
        ax3.plot(positions, results['V_ocv_profiles'][i],
                 color=COLORS[i], lw=2.4, label=lbl)
    ax3.set_xlabel('Position along Cell (%)', fontsize=10)
    ax3.set_ylabel('Local OCV (V)', fontsize=10)
    ax3.set_title('2. OCV Profile Along Cell', fontsize=11,
                  fontweight='bold', pad=8)
    ax3.text(2, max(results['V_ocv_profiles'][2]) * 1.002,
             'OCV rises as H₂O is consumed and H₂ builds up',
             color='#909090', fontsize=7.5, fontstyle='italic')
    _legend(ax3, loc='lower right')
    _style(ax3)


    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(AX_BG)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('4. 0D → 1D: What Changed?', fontsize=11,
                  fontweight='bold', pad=8, color=FG)
    
    col0_x, col1_x, col2_x = 0.02, 0.38, 0.70
    header_y = 0.91

    def txt(ax, x, y, s, **kw):
        kw.setdefault('color', FG)
        kw.setdefault('fontsize', 8.5)
        ax.text(x, y, s, transform=ax.transAxes,
                va='top', **kw)
        
    ax4.text(col1_x, 0.96, '0D Model', transform=ax4.transAxes,
             color=GOLD, fontsize=9, fontweight='bold', va='top', ha='center')
    ax4.text(col2_x + 0.14, 0.96, '1D Model', transform=ax4.transAxes,
             color=ACCENT, fontsize=9, fontweight='bold', va='top', ha='center')
    
    ax4.axhline(0.87, color=GRID_C, lw=0.8, xmin=0.0, xmax=1.0)

    rows = [
        ('Spatial Resolution',  'Single Point',        '20 Nodes Along Flow Path'),
        ('Gas Composition',     'Fixed Inlet Values',  'Evolves Inlet → Outlet'),
        ('Nernst Voltage',      'One Global OCV',      'Recalculated at Each Node'),
        ('Fuel Utilisation',    'Not Modelled',        '60-70 % per Temperature'),
        ('OCV Along Cell',      'Not Available',       'Rising Curve (Plot 2)'),
        ('H₂O / H₂ Profile',    'Not Available',       'Fully Resolved (Plot 3)'),
        ('Validation',          '-',                   'Jensen (2007) Figure 21'),
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
             'SOEC Cell Model  |  1D Extension  |  Python Implementation',
             ha='center', fontsize=13, fontweight='bold', color=FG)
    fig.text(0.5, 0.913,
             'Nernst Eq  ·  Butler-Volmer Kinetics  ·  Fick Diffusion  ·  Arrhenius Resistance  ·  1D Flow Path',
             ha='center', fontsize=8.5, color='#888888')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=BG)
    print(f'LinkedIn Figure Saved → {save_path}')
    plt.close()