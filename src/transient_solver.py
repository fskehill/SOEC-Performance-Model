import numpy as np
from src.overpotentials import calc_overpotentials
from src.nernst import calc_ocv
import dataclasses

def soec_ode(t, y, params, j_step_func, N_nodes):
    """
    y: Flat array of Temperatures [T1, T2, ..., TN] for each node
    j_func: A function that returns current density at time t
    """
    T_nodes = y[:N_nodes]
    x_H2O_nodes = y[N_nodes:]

    j_val = j_step_func(t)

    THERMAL_MASS = 2.0
    GAS_VOLUME = 1e-5
    M_DOT_CP = 0.35
    P_TOTAL = 101325
    R_GAS = 8.314
    F = 96485
    V_THERMO = 1.48
    A_ELEC_NODE = 5.0
    F_MOLAR = 2.0e-3

    dTdt = np.zeros(N_nodes)
    dCdt = np.zeros(N_nodes)

    for i in range(N_nodes):

        T_local = T_nodes[i]
        x_H2O_local = x_H2O_nodes[i]
        x_H2_local = 1.0 - x_H2O_local

        V_ocv = calc_ocv(T_local, x_H2_local, x_H2O_local, params.x_O2, params.P)
        params_node = dataclasses.replace(params, x_H2=x_H2_local, x_H2O=x_H2O_local)
        eta_o, eta_a, eta_c = calc_overpotentials(j_val, T_local, params_node)
        V_cell = V_ocv + eta_o + eta_a + np.nan_to_num(eta_c, 0)

        Q_gen = j_val * A_ELEC_NODE * (V_cell - V_THERMO)
        T_prev = 1073.15 if i == 0 else T_nodes[i-1]
        Q_conv = M_DOT_CP * (T_local - T_prev)

        dTdt[i] = (Q_gen - Q_conv) / THERMAL_MASS

        x_prev = params.x_H2O if i == 0 else x_H2O_nodes[i-1]
        moles_in_node = (P_TOTAL * GAS_VOLUME) / (R_GAS * T_local)

        n_dot_react = j_val * A_ELEC_NODE / (2 * F)
        dx_H2O_conv = F_MOLAR * (x_prev - x_H2O_local) / moles_in_node
        dx_H2O_rxn = n_dot_react / moles_in_node

        dCdt[i] = dx_H2O_conv - dx_H2O_rxn

        dCdt = np.where(x_H2O_nodes < 0.05, np.maximum(dCdt, 0), dCdt)

    return np.concatenate([dTdt, dCdt])