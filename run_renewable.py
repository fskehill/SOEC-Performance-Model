import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transient_solver import soec_ode
from src.overpotentials import CellParams, calc_overpotentials
from src.nernst import calc_ocv
from src.renewable_profile import renewable_j_profile, solar_irradiance_profile

params = CellParams(x_H2O=0.5, x_H2=0.5)
N_nodes = 20

T_SUNRISE_H = 6.0
T_SUNSET_H = 18.0
G_PEAK = 1000.0
J_MAX = 1.2
J_STANDBY = 0.05

def j_renewable_input(t):
    return renewable_j_profile(
        t, t_sunrise_h=T_SUNRISE_H, t_sunset_h=T_SUNSET_H,
        G_peak=G_PEAK, j_max=J_MAX, j_standby=J_STANDBY
    )

y0 = np.concatenate([
    np.full(N_nodes, 1073.15),
    np.full(N_nodes, 0.5)
])

print("Integrating 24h renewable-driven transient response... this will take longer than the step test.")

t_span = (0, 86400)
t_eval = np.linspace(*t_span, 1500)

sol = solve_ivp(
    soec_ode,
    t_span=t_span,
    y0=y0,
    args=(params, j_renewable_input, N_nodes),
    method='Radau',
    t_eval=t_eval,
    max_step=60.0
)

print(f"sol success: {sol.success}, points: {sol.y.shape}")
print(f"sol message: {sol.message}")

sol_t = sol.t
sol_y = sol.y

print("Simulation Complete.")

time_h = sol_t / 3600
j_values = np.array([j_renewable_input(t) for t in sol_t])
G_values = solar_irradiance_profile(sol_t, T_SUNRISE_H, T_SUNSET_H, G_PEAK)

fig, axs = plt.subplots(2, 3, figsize=(16, 10), facecolor='#1a1a1e')
plt.subplots_adjust(hspace=0.35, wspace=0.35)

axs[0, 0].plot(time_h, G_values, color='#f0c040', lw=2)
axs[0, 0].set_title("1. Solar Irradiance", color='white')
axs[0, 0].set_ylabel("G (W/m²)", color='white')

axs[0, 1].plot(time_h, j_values, color='cyan', lw=2)
axs[0, 1].set_title("2. Current Density (MPPT-tracked)", color='white')
axs[0, 1].set_ylabel("j (A/m²)", color='white')

axs[0, 2].plot(time_h, sol_y[0]-273.15, label='Inlet', color='#2E75C0')
axs[0, 2].plot(time_h, sol_y[N_nodes-1]-273.15, label='Outlet', color='#D94F3D')
axs[0, 2].set_title("3. Thermal Response", color='white')
axs[0, 2].set_ylabel("Temperature (°C)", color='white')
axs[0, 2].legend(facecolor='#1a1a1e', labelcolor='white')

x_H2O_outlet = sol_y[N_nodes + N_nodes - 1, :]
FU = np.where(j_values > 1e-6, (0.5 - x_H2O_outlet) / 0.5 * 100, 0.0)
axs[1, 0].plot(time_h, FU, color='#34A876', lw=2)
axs[1, 0].set_title("4. Fuel Utilisation (%)", color='white')
axs[1, 0].set_ylabel("FU (%)", color='white')

V_actual = []
for i in range(len(sol_t)):
    T_avg = np.mean(sol_y[:N_nodes, i])
    x_H2O_avg = np.mean(sol_y[N_nodes:, i])
    x_H2_avg = 1.0 - x_H2O_avg

    V_ocv = calc_ocv(T_avg, x_H2_avg, x_H2O_avg, params.x_O2, params.P)
    eo, ea, ec = calc_overpotentials(j_values[i], T_avg, params)
    V_actual.append(V_ocv + eo + ea + np.nan_to_num(ec, 0))

axs[1, 1].plot(time_h, V_actual, color='#f0c040', lw=2)
axs[1, 1].set_title("5. Voltage Response", color='white')
axs[1, 1].set_ylabel("Cell Voltage (V)", color='white')

V_tn = 1.48
efficiency = np.where(
    np.array(j_values) > 1e-6,
    [V_tn / v * 100 if v > 0 else 0 for v in V_actual],
    0.0,
)
axs[1, 2].plot(time_h, efficiency, color='#a29bfe', lw=2)
axs[1, 2].axhline(100, color='#00b894', ls='--', lw=1.2, label='Thermoneutral')
axs[1, 2].set_title("6. Electrolysis Efficiency", color='white')
axs[1, 2].set_ylabel("Efficiency (%)", color='white')
axs[1, 2].legend(facecolor='#1a1a1e', labelcolor='white')

for ax in axs.flat:
    ax.set_xlim([0, 24])
    ax.axvline(x=T_SUNRISE_H, color='white', ls='--', alpha=0.3)
    ax.axvline(x=T_SUNSET_H, color='white', ls='--', alpha=0.3)
    ax.set_facecolor('#222228')
    ax.tick_params(colors='white')
    ax.grid(True, color='#3a3a42', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a42')
    ax.set_xlabel("Time (h)", color='white')

plt.suptitle("SOEC Cell Model  |  24h Renewable-Driven Transient  |  Python Implementation",
             color='white', fontsize=13)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/SOEC_Renewable_Results.png", dpi=300)
plt.show()