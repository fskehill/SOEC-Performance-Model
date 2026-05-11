import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nernst import calc_ocv
from src.overpotentials import CellParams, calc_overpotentials
from src.plotter import plot_results

T_degC = [750, 800, 850]
T_list = [T + 273.15 for T in T_degC]
T_labels = ['750°C', '800°C', '850°C']
j = np.linspace(0, 1.8, 600)
V_thermo = 1.48
params = CellParams()

print("Running SOEC Model...")

results = {'V_cell':[], 'eta_ohm':[], 'eta_act':[],
           'eta_con':[], 'eff':[], 'V_ocv':[]}

for T in T_list:
    V_ocv = calc_ocv(T, params.x_H2, params.x_H2O,
                     params.x_O2, params.P)
    eta_ohm, eta_act, eta_con = calc_overpotentials(j, T, params)
    V_cell = V_ocv + eta_ohm + eta_act + np.nan_to_num(eta_con, nan=0)
    eff = (V_thermo / V_cell) * 100

    results['V_ocv'].append(V_ocv)
    results['V_cell'].append(V_cell)
    results['eta_ohm'].append(eta_ohm)
    results['eta_act'].append(eta_act)
    results['eta_con'].append(eta_con)
    results['eff'].append(eff)
    print(f"T = {T-273.15:.0f}°C | OCV = {V_ocv:.3f}V")

for key in ['V_cell', 'eta_ohm', 'eta_act', 'eta_con', 'eff']:
    results[key] = np.array(results[key])

R_GAS = 8.314
T_range = np.linspace(580, 920, 300) + 273.15
results['T_range'] = T_range
results['ASR_range'] = params.B_ohm * np.exp(params.E_act_ohm / (R_GAS * T_range)) * 1e4
results['ASR_pts'] = [params.B_ohm * np.exp(params.E_act_ohm / (R_GAS * T)) * 1e4 for T in T_list]

Path("outputs").mkdir(exist_ok=True)
plot_results(j, results, T_list, T_labels,
             V_thermo, "outputs/SOEC_Results.png")

print("Done.")