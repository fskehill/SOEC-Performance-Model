import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transient_solver import soec_ode
from src.overpotentials import CellParams, calc_overpotentials
from src.nernst import calc_ocv
from src.renewable_profile import renewable_j_profile, solar_irradiance_profile, DUBLIN_SEASONS

params = CellParams(x_H2O=0.5, x_H2=0.5)
N_nodes = 20
J_MAX, J_STANDBY = 1.2, 0.05

V_TN = 1.48

def run_season(season_key):
    s = DUBLIN_SEASONS[season_key]

    def j_func(t):
        return renewable_j_profile(t, s['t_sunrise_h'], s['t_sunset_h'],
                                   s['G_peak'], J_MAX, J_STANDBY, g_ref=1000.0)
    
    def g_func(t):
        return solar_irradiance_profile(t, s['t_sunrise_h'], s['t_sunset_h'], s['G_peak'])
    
    y0 = np.concatenate([np.full(N_nodes, 1073.15), np.full(N_nodes, 0.5)])
    sol = solve_ivp(
        soec_ode, t_span=(0, 86400), y0=y0,
        args=(params, j_func, N_nodes),
        method='Radau', t_eval=np.linspace(0, 86400, 1000), max_step=60.0,
    )
    print(f"  [{season_key}] solver success: {sol.success}")

    t = sol.t
    j_vals = np.array([j_func(ti) for ti in t])
    G_vals = np.array([g_func(ti) for ti in t])
    T_outlet = sol.y[N_nodes - 1] - 273.15

    V_actual = np.zeros(len(t))
    for i in range(len(t)):
        T_avg = np.mean(sol.y[:N_nodes, i])
        x_H2O_avg = np.mean(sol.y[N_nodes:, i])
        V_ocv = calc_ocv(T_avg, 1 - x_H2O_avg, x_H2O_avg, params.x_O2, params.P)
        eo, ea, ec = calc_overpotentials(j_vals[i], T_avg, params)
        V_actual[i] = V_ocv + eo + ea + np.nan_to_num(ec, 0)
    efficiency = np.where(j_vals > 1e-6, V_TN / np.maximum(V_actual, 1e-6) * 100, np.nan)

    return t, G_vals, j_vals, T_outlet, efficiency


def plot_season(season_key, t, G_vals, j_vals, T_outlet, efficiency, outpath):
    s = DUBLIN_SEASONS[season_key]
    time_h = t / 3600

    plt.rcParams.update({'font.size': 13})
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), facecolor='#1a1a1e')
    plt.subplots_adjust(hspace=0.38, wspace=0.32, top=0.88)

    panels = [
        (axs[0, 0], G_vals, "Solar Irradiance", "G (W/m²)", '#f0c040'),
        (axs[0, 1], j_vals, "Current Density", "j (A/cm²)", 'cyan'),
        (axs[1, 0], T_outlet, "Outlet Temperature", "T (°C)", '#D94F3D'),
        (axs[1, 1], efficiency, "Electrolysis Efficiency", "Efficiency (%)", '#a29bfe'),
    ]
    for ax, data, title, ylabel, color in panels:
        ax.plot(time_h, data, color=color, lw=2.5)
        ax.set_title(title, color='white', fontsize=14, pad=10)
        ax.set_ylabel(ylabel, color='white', fontsize=12)
        ax.set_xlabel("Time (h)", color='white', fontsize=12)
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_facecolor('#222228')
        ax.tick_params(colors='white', labelsize=11)
        ax.grid(True, color='#3a3a42', alpha=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3a42')

    fig.suptitle(f"SOEC Renewable-Driven Transient - {s['label']}",
                 color='white', fontsize=16, y=0.96)
    
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(outpath, dpi=200, facecolor='#1a1a1e', bbox_inches='tight')
    plt.close(fig)


print("Running Dublin Summer Solstice...")
t_s, G_s, j_s, T_s, eff_s = run_season('summer')
plot_season('summer', t_s, G_s, j_s, T_s, eff_s, "outputs/SOEC_Dublin_Summer.png")

print("Running Dublin Winter Solstice...")
t_w, G_w, j_w, T_w, eff_w = run_season('winter')
plot_season('winter', t_w, G_w, j_w, T_w, eff_w, "outputs/SOEC_Dublin_Winter.png")

print("\nPeak Current Density Reached:")
print(f"   Summer: {j_s.max():.3f} A/cm^2  ({j_s.max()/J_MAX*100:.0f}% of {J_MAX} A/cm^2 rated)")
print(f"   Winter: {j_w.max():.3f} A/cm^2  ({j_w.max()/J_MAX*100:.0f}% of {J_MAX} A/cm^2 rated)")
print("Done.")