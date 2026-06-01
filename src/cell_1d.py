import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.nernst import calc_ocv
from src.overpotentials import CellParams, calc_overpotentials
from src.thermal_1d import calc_temperature_profile
import dataclasses

def calc_1d(j, T, params, FU, N_nodes=20):
    """
    1D model - divides cell into N_nodes slices along flow direction.
    At each slice, gas compoisition is updated based on how much H2O 
    has been consumed up to that point.
    
    Parameters:
    ----------
    j             : current density             [A/cm²]
    T             : temperature                 [K]
    params        : CellParams 
    FU            : fuel utilisation            [-]
    N_nodes       : number of slices            default 20
    
    Returns:
    -------
    V_cell        : total cell voltage          [V] array same length as j
    eta_ohm       : averaged ohmic loss         [V]
    eta_act       : averaged activation loss    [V]
    eta_con       : averaged concentration loss [V]
    V_ocv         : averaged OCV                [V] scalar
    x_H2O_profile : H2O along cell              list length N_nodes
    x_H2_profile  : H2 along cell               list length N_nodes
    T_profile     : temperature along cell      list length N_nodes
    """

    positions = np.linspace(0, 1, N_nodes)

    eta_ohm_pass1 = []
    V_ocv_pass1 = []

    for pos in positions:
        local_FU = FU * pos
        x_H2O_local = params.x_H2O - local_FU * params.x_H2O
        x_H2_local = params.x_H2 + local_FU * params.x_H2O
        total = x_H2O_local + x_H2_local
        x_H2O_local /= total
        x_H2_local /= total

        V_ocv_node = calc_ocv(T, x_H2_local, x_H2O_local,
                              params.x_O2, params.P)
        params_node = dataclasses.replace(params,
                                          x_H2=x_H2_local,
                                          x_H2O=x_H2O_local)
        eta_o, _, _ = calc_overpotentials(j, T, params_node)

        eta_ohm_pass1.append(float(np.mean(eta_o)))
        V_ocv_pass1.append(V_ocv_node)

    j_op = float(np.mean(j[j > 0]))

    T_profile = calc_temperature_profile(
        j_op = j_op,
        T_inlet = T,
        V_cell_nodes = [v + eta_ohm_pass1[i] for i, v in enumerate(V_ocv_pass1)],
        V_thermo = 1.48,
        N_nodes = N_nodes,
        m_dot_cp = 0.8
    )
        
    V_cell_nodes = []
    
    x_H2O_profile = []
    x_H2_profile = []
    V_ocv_nodes = []
    eta_ohm_nodes = []
    eta_act_nodes = []
    eta_con_nodes = []

    for i, pos in enumerate(positions):
        T_local = T_profile[i]

        local_FU = FU * pos
        x_H2O_local = params.x_H2O - local_FU * params.x_H2O
        x_H2_local = params.x_H2 + local_FU * params.x_H2O
        total = x_H2O_local + x_H2_local
        x_H2O_local /= total
        x_H2_local /= total

        x_H2O_profile.append(x_H2O_local)
        x_H2_profile.append(x_H2_local)

        V_ocv_node = calc_ocv(T_local, x_H2_local, x_H2O_local,
                              params.x_O2, params.P)
        V_ocv_nodes.append(V_ocv_node)

        params_node = dataclasses.replace(params,
                                          x_H2=x_H2_local,
                                          x_H2O=x_H2O_local)
        eta_o, eta_a, eta_c = calc_overpotentials(j, T_local, params_node)

        eta_ohm_nodes.append(eta_o)
        V_cell_nodes.append(float(np.interp(j_op, j, V_ocv_node + eta_o + eta_a + np.nan_to_num(eta_c, nan=0))))
        eta_act_nodes.append(eta_a)
        eta_con_nodes.append(eta_c)

    V_ocv = float(np.mean(V_ocv_nodes))
    eta_ohm = np.mean(eta_ohm_nodes, axis=0)
    eta_act = np.mean(eta_act_nodes, axis=0)
    eta_con = np.mean(eta_con_nodes, axis=0)

    V_cell = V_ocv + eta_ohm + eta_act + np.nan_to_num(eta_con, nan=0)

    V_ocv_profile = V_ocv_nodes

    return V_cell, eta_ohm, eta_act, eta_con, V_ocv, x_H2O_profile, x_H2_profile, V_ocv_profile, T_profile, V_cell_nodes