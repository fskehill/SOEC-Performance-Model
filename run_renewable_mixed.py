import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transient_solver import soec_ode
from src.overpotentials import CellParams
from src.renewable_profile import renewable_j_profile, solar_irradiance_profile

params = CellParams(x_H2O=0.5, x_H2=0.5)
N_nodes = 20

T_SUNRISE_H = 6.0
T_SUNSET_H = 18.0
G_PEAK = 1000.0
J_MAX = 1.2
J_STANDBY = 0.05

CLOUD_EVENTS = [
    {'start': 36000, 'duration': 2, 'depth': 0.85, 'ramp': 0.5},
    {'start': 52200, 'duration': 300, 'depth': 0.60, 'ramp': 15},
]

def j_mixed_input(t):
    return renewable_j_profile(
        t, t_sunrise_h=T_SUNRISE_H, t_sunset_h=T_SUNSET_H,
        G_peak=G_PEAK, j_max=J_MAX, j_standby=J_STANDBY,
        cloud_events=CLOUD_EVENTS,
    )

def g_mixed_input(t):
    return solar_irradiance_profile(
        t, T_SUNRISE_H, T_SUNSET_H, G_PEAK, cloud_events=CLOUD_EVENTS)

BUF = 30
segments = [
    (0, 35998, 60.0, 300),
    (35998, 36004, 0.02, 300),
    (36004, 52100, 60.0, 400),
    (52100, 52600, 2.0, 250),
    (52600, 86400, 60.0, 400),
]

y0 = np.concatenate([np.full(N_nodes, 1073.15), np.full(N_nodes, 0.5)])

print("Integrating mixed-timescale (daily + cloud transient) response...")

sol_t_parts, sol_y_parts = [], []
for t0, t1, max_step, n_pts in segments:
    sol = solve_ivp(
        soec_ode, t_span=(t0, t1), y0=y0,
        args=(params, j_mixed_input, N_nodes),
        method='Radau', t_eval=np.linspace(t0, t1, n_pts), max_step=max_step,
    )
    if not sol.success:
        print(f"  WARNING: segment ({t0},{t1}) failed: {sol.message}")
    sol_t_parts.append(sol.t)
    sol_y_parts.append(sol.y)
    y0 = sol.y[:, -1]

sol_t = np.concatenate(sol_t_parts)
sol_y = np.hstack(sol_y_parts)
print("Simulation complete.")

time_h = sol_t / 3600
j_values = np.array([j_mixed_input(t) for t in sol_t])
G_values = np.array([g_mixed_input(t) for t in sol_t])
T_outlet = sol_y[N_nodes - 1] - 273.15

fig = plt.figure(figsize=(18, 10), facecolor='#1a1a1e')
gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.45)

panel_specs = [
    (gs[0, 0], G_values, "1. Irradiance (24h)", "G (W/m²)", '#f0c040'),
    (gs[0, 1], j_values, "2. Current Density (24h)", "j (A/cm²)", 'cyan'),
    (gs[0, 2], T_outlet, "3. Outlet Temp (24h)", "T (°C)", '#D94F3D'),
]
for spec, data, title, ylabel, color in panel_specs:
    ax = fig.add_subplot(spec)
    ax.plot(time_h, data, color=color, lw=1.5)
    for ev in CLOUD_EVENTS:
        ax.axvspan(ev['start']/3600, (ev['start']+ev['duration'])/3600,
                   color='white', alpha=0.15, lw=0)
    ax.set_title(title, color='white', fontsize=10)
    ax.set_ylabel(ylabel, color='white')
    ax.set_xlabel("Time (h)", color='white')
    ax.set_xlim(0, 24)
    ax.set_facecolor('#222228')
    ax.tick_params(colors='white')
    ax.grid(True, color='#3a3a42', alpha=0.5)
    for s in ax.spines.values():
        s.set_edgecolor('#3a3a42')

def zoom_panel(gridspec_slot, center_s, half_width_s, label, unit='min'):
    ax = fig.add_subplot(gridspec_slot)
    mask = (sol_t >= center_s - half_width_s) & (sol_t <= center_s + half_width_s)
    divisor = 60.0 if unit == 'min' else 1.0
    t_rel = (sol_t[mask] - center_s) / divisor

    j_norm = j_values[mask] / J_MAX
    G_norm = G_values[mask] / G_PEAK
    T_dev = T_outlet[mask] - T_outlet[mask][0]

    ax.plot(t_rel, G_norm, color='#f0c040', lw=1.5, label='Irradiance (norm.)')
    ax.plot(t_rel, j_norm, color='cyan', lw=1.5, label='Current (norm.)')
    ax2 = ax.twinx()
    ax2.plot(t_rel, T_dev, color='#D94F3D', lw=2, label='Outlet ΔT')
    ax2.set_ylabel("ΔT (°C)", color='#D94F3D', labelpad=10)
    ax2.tick_params(colors='#D94F3D')

    ax.set_title(label, color='white', fontsize=10)
    ax.set_xlabel(f"{'Minutes' if unit == 'min' else 'Seconds'} from event start", color='white')
    ax.set_ylabel("Normalised (0-1)", color='white', labelpad=8)
    ax.set_facecolor('#222228')
    ax.tick_params(colors='white')
    ax.grid(True, color='#3a3a42', alpha=0.4)
    for s in ax.spines.values():
        s.set_edgecolor('#3a3a42')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, facecolor='#1a1a1e', labelcolor='white', fontsize=8, loc='lower left')


zoom_panel(gs[1, 0], 36000, 6, "4. Zoom: 2s flicker (< tau, partially filtered)", unit='sec')
zoom_panel(gs[1, 1:], 52200, 400, "5. Zoom: 5min cloud (>> tau, fully tracked)", unit='min')

plt.suptitle("SOEC Cell Model  |  Mixed Timescale: Daily Solar Cycle + Cloud Transients",
             color='white', fontsize=13)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/SOEC_Mixed_Timescale_Results.png", dpi=300, facecolor='#1a1a1e')
plt.show()