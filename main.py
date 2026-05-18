import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nernst import calc_ocv
from src.overpotentials import CellParams, calc_overpotentials
from src.plotter import plot_results
from src.fuel_utilisation import calc_inlet_outlet
from src.cell_1d import calc_1d
from src.validator import j_exp, V_exp, calc_error
from src.linkedin_plotter import plot_linkedin

T_degC = [750, 800, 850]
T_list = [T + 273.15 for T in T_degC]
T_labels = ['750°C', '800°C', '850°C']
j = np.linspace(0, 1.8, 600)
V_thermo = 1.48
FU_dict = {750+273.15: 0.60, 800+273.15: 0.65, 850+273.15: 0.70}
params = CellParams()

print("Running SOEC Model...")

results = {'V_cell':[], 'eta_ohm':[], 'eta_act':[],
           'eta_con':[], 'eff':[], 'V_ocv':[]}

x_H2O_profiles = []
x_H2_profiles = []

for T in T_list:
    
    V_cell, eta_ohm, eta_act, eta_con, V_ocv, x_H2O_prof, x_H2_prof, V_ocv_prof = calc_1d(j, T, params, FU_dict[T], N_nodes=20)

    eff = (V_thermo / V_cell) * 100

    results['V_ocv'].append(V_ocv)
    results['V_cell'].append(V_cell)
    results['eta_ohm'].append(eta_ohm)
    results['eta_act'].append(eta_act)
    results['eta_con'].append(eta_con)
    results['eff'].append(eff)

    x_H2O_profiles.append(x_H2O_prof)
    x_H2_profiles.append(x_H2_prof)

    results.setdefault('V_ocv_profiles', []).append(V_ocv_prof)

    print(f"  T={T-273.15:.0f}°C  |  OCV={V_ocv:.3f}V")

results['x_H2O_profiles'] = x_H2O_profiles
results['x_H2_profiles'] = x_H2_profiles
results['N_nodes'] = 20

for key in ['V_cell', 'eta_ohm', 'eta_act', 'eta_con', 'eff']:
    results[key] = np.array(results[key])

R_GAS = 8.314
T_range = np.linspace(580, 920, 300) + 273.15
results['T_range'] = T_range
results['ASR_range'] = params.B_ohm * np.exp(params.E_act_ohm / (R_GAS * T_range)) * 1e4
results['ASR_pts'] = [params.B_ohm * np.exp(params.E_act_ohm / (R_GAS * T)) * 1e4 for T in T_list]

params_jensen = CellParams(
    x_H2=0.50,
    x_H2O=0.50,
    gamma_c=5.5e12,
    gamma_a=2.0e12,
    E_act_c=80000,
    E_act_a=90000,
    B_ohm=1.0e-5,
    E_act_ohm=7200,
    )
T_jensen = 850 + 273.15
V_jensen, _, _, _, _, _, _, _ = calc_1d(j, T_jensen, params_jensen, FU=0.65, N_nodes=20)

errors, V_interp = calc_error(j, V_jensen, j_exp, V_exp)

print("\nValidation vs Jensen (2007) at 850°C:")
print(f"  {'j (A/cm²)':<12} {'Model (V)':<12} {'Exp (V)':<12} {'Error (%)'}")
print("  " + "-" * 48)
for i in range(len(j_exp)):
    print(f"  {j_exp[i]:12.1f} {V_interp[i]:12.3f} "
          f"{V_exp[i]:<12.3f} {errors[i]:+.1f}%")
    
results['V_jensen'] = V_jensen
results['errors'] = errors
results['V_interp'] = V_interp

Path("outputs").mkdir(exist_ok=True)
plot_results(j, results, T_list, T_labels,
             V_thermo, "outputs/SOEC_Results.png",
             j_exp=j_exp, V_exp=V_exp)
plot_linkedin(j, results, T_list, T_labels,
              V_thermo, "outputs/SOEC_LinkedIn.png",
              j_exp=j_exp, V_exp=V_exp)

print("Done.")