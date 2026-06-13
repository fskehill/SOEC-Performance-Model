import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.transient_solver import soec_ode
from src.overpotentials import CellParams, calc_overpotentials
from src.nernst import calc_ocv

params = CellParams(x_H2O=0.5, x_H2=0.5)
N_nodes = 20

def j_step_input(t):
    if t < 60:
        return 0.2
    elif t < 1800:
        return 0.8
    else:
        return 0.2

y0 = np.concatenate([
    np.full(N_nodes, 1073.15),
    np.full(N_nodes, 0.5)
])

print("Integrating transient response... - this may take a moment.")

t_eval_1 = np.linspace(0, 60, 300)
sol1 = solve_ivp(
    soec_ode,
    t_span=(0, 60),
    y0=y0,
    args=(params, j_step_input, N_nodes),
    method='Radau',
    t_eval=t_eval_1,
    max_step=1.0,
)

t_eval_2 = np.linspace(60, 300, 500)
sol2 = solve_ivp(
    soec_ode,
    t_span=(60, 300),
    y0=sol1.y[:, -1],
    args=(params, j_step_input, N_nodes),
    method='Radau',
    t_eval=t_eval_2,
    max_step=5.0,
)

print(f"sol1 success: {sol1.success}, points: {sol1.y.shape}")
print(f"sol2 success: {sol2.success}, points: {sol2.y.shape}")
print(f"sol2 message: {sol2.message}")

sol_t = np.concatenate([sol1.t, sol2.t])
sol_y = np.hstack([sol1.y, sol2.y])

print("Simulation complete.")

fig, axs = plt.subplots(2, 3, figsize=(16, 10), facecolor='#1a1a1e')
plt.subplots_adjust(hspace=0.35, wspace=0.35)

time_min = sol_t / 60
j_values = [j_step_input(t) for t in sol_t]

axs[0, 0].plot(time_min, j_values, color='cyan', lw=2)
axs[0, 0].set_title("1. Current Input (Step Change)", color='white')
axs[0, 0].set_ylabel("j (A/cm²)", color='white')

axs[0, 1].plot(time_min, sol_y[0]-273.15, label='Inlet', color='#2E75C0')
axs[0, 1].plot(time_min, sol_y[N_nodes-1]-273.15, label='Outlet', color='#D94F3D')
axs[0, 1].set_title("2. Thermal Response (Minutes)", color='white')
axs[0, 1].set_ylabel("Temperature (°C)", color='white')
axs[0, 1].legend()
axs[0, 1].annotate(
    'τ_electrochemical ~ seconds\nτ_thermal ~ minutes',
    xy=(0.30, 0.95), xycoords='axes fraction',
    color='#ffd700', fontsize=8, va='top',
    bbox=dict(boxstyle='round', facecolor='#1a1a1e', alpha=0.7))

x_H2O_outlet = sol_y[N_nodes + N_nodes - 1, :]
FU = (0.5 - x_H2O_outlet) / 0.5 * 100
axs[0, 2].plot(time_min, FU, color='#34A876', lw=2)
axs[0, 2].set_title("3. Fuel Utilisation (%)", color='white')
axs[0, 2].set_ylabel("FU (%)", color='white')

V_actual = []
for i in range(len(sol_t)):

    T_avg = np.mean(sol_y[:N_nodes, i])
    x_H2O_avg = np.mean(sol_y[N_nodes:, i])
    x_H2_avg = 1.0 - x_H2O_avg

    V_ocv = calc_ocv(T_avg, x_H2_avg, x_H2O_avg, params.x_O2, params.P)
    eo, ea, ec = calc_overpotentials(j_values[i], T_avg, params)
    V_actual.append(V_ocv + eo + ea + np.nan_to_num(ec, 0))

axs[1, 0].plot(time_min, V_actual, color='#f0c040', lw=2)
axs[1, 0].set_title("4. Voltage Response (Instant)", color='white')
axs[1, 0].set_ylabel("Cell Voltage (V)", color='white')

V_tn = 1.48
efficiency = [V_tn / v * 100 for v in V_actual]

axs[1, 1].plot(time_min, efficiency, color='#a29bfe', lw=2)
axs[1, 1].axhline(100, color='#00b894', ls='--', lw=1.2, label='Thermoneutral')
axs[1, 1].set_title("5. Electrolysis Efficiency", color='white')
axs[1, 1].set_ylabel("Efficiency (%)", color='white')
axs[1, 1].legend(facecolor='#1a1a1e', labelcolor='white')

axs[1, 2].plot(time_min, sol_y[N_nodes], label='H₂O Inlet', ls='--', color='gray')
axs[1, 2].plot(time_min, sol_y[-1], label='H₂O Outlet', color='#74b9ff')
axs[1, 2].set_title("6. Gas Composition (H₂O)", color='white')
axs[1, 2].set_ylabel("Mole Fraction", color='white')
axs[1, 2].legend(facecolor='#1a1a1e', labelcolor='white')

for ax in axs.flat:
    ax.set_xlim([0, 5])
    ax.axvline(x=1.0, color='white', ls='--', alpha=0.3)
    ax.set_facecolor('#222228')
    ax.tick_params(colors='white')
    ax.grid(True, color='#3a3a42', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3a42')
    ax.set_xlabel("Time (min)", color='white')

plt.suptitle("SOEC Cell Model  |  Transient 1.5D  |  Python Implementation",
             color='white', fontsize=13)

plt.tight_layout()
plt.savefig("outputs/SOEC_Transient_Results.png", dpi=300)
plt.show()