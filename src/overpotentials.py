import numpy as np
from dataclasses import dataclass, field

F = 96485
R_GAS = 8.314
N = 2

@dataclass
class CellParams:
    x_H2: float = 0.40
    x_H2O: float = 0.60
    x_O2: float = 0.21
    P: float = 1e5
    P_ref: float = 1e5
    B_ohm: float = 2.94e-5
    E_act_ohm: float = 8500.0
    alpha_c: float = 0.5
    gamma_c: float = 5.5e8
    E_act_c: float = 100000.0
    a_exp: float = 0.5
    b_exp: float = 0.5
    gamma_a: float = 2.0e8
    E_act_a: float = 110000.0
    m_exp: float = 0.25
    D_eff_c: float = 3e-5
    L_cell: float = 0.5e-3

def calc_overpotentials(j, T, p):
    p_H2 = p.x_H2 * p.P
    p_H2O = p.x_H2O * p.P
    p_O2 = p.x_O2 * p.P

    ASR = p.B_ohm * np.exp(p.E_act_ohm / (R_GAS * T)) * 1e4
    eta_ohm = j * ASR

    J0_c = (p.gamma_c
            * (p_H2 / p.P_ref) ** p.a_exp
            * (p_H2O / p.P_ref) ** p.b_exp
            * np.exp(-p.E_act_c / (R_GAS * T))) * 1e-4
    
    J0_a = (p.gamma_a
            * (p_O2 / p.P_ref) ** p.m_exp
            * np.exp(-p.E_act_a / (R_GAS * T))) * 1e-4
    
    J0 = 2 / (1/J0_c + 1/J0_a)
    j_safe = np.maximum(j, 1e-9)
    eta_act = ((R_GAS * T) / (p.alpha_c * N * F)) * np.log(j_safe / J0 + np.sqrt((j_safe / J0)**2 + 1))

    c_H2O = p_H2O / (R_GAS * T)
    j_lim = (N * F * p.D_eff_c * c_H2O) / p.L_cell * 1e-4
    eta_con = -(R_GAS * T / (N * F)) * np.log(1 - j_safe / j_lim)
    eta_con[j >= j_lim * 0.98] = np.nan

    return eta_ohm, eta_act, eta_con