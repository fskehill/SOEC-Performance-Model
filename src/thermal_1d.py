"""
thermal_1d.py
-------------
Calculates temperature profile along the cell (x-direction).

Energy balance at each node:
    Heat Generated  =  Ohmic Losses (I²R)
    Heat Consumed   =  Endothermic Reaction (below thermoneutral voltage)
    Net Heat        =  Changes Local Temperature

dT/dx = (Q_ohm - Q_reaction) / (m_dot * Cp)

Where:
    Q_ohm       =  j * eta_ohm                  [W/cm²] Heat from Resistance
    Q_reaction  =  j * (V_thermo - V_ocv_node)  [W/cm²] Reaction Heat
    m_dot * Cp  =  Thermal Mass of Gas Flow
"""

import numpy as np

def calc_temperature_profile(j_op, T_inlet,
                             V_cell_nodes, V_thermo,
                             N_nodes=20, m_dot_cp=0.5):
    """
    Calculates temperature at each node along the flow path.
    
    Parameters
    ----------
    j_op            : operating current density  [A/cm²]
    T_inlet         : inlet temperature          [K]
    eta_ohm_nodes   : ohmic losses at each node  [W/cm²]
    V_ocv_nodes     : open-circuit voltage at each node  [V]
    V_thermo        : thermal voltage            [V]
    N_nodes         : number of nodes            [-]
    m_dot_cp        : mass flow rate * heat capacity  [W/K]
    
    Returns
    -------
    T_profile       : temperature at each node  [K] list length N_nodes
    """

    T_profile = [T_inlet]
    T_current = T_inlet

    for i in range(N_nodes - 1):

        Q_net = j_op * (V_cell_nodes[i] - V_thermo)
        dT = Q_net / m_dot_cp
        T_current += dT
        T_profile.append(T_current)

    return T_profile