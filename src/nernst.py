import numpy as np

F = 96485
R_GAS = 8.314
N = 2

def calc_ocv(T, x_H2, x_H2O, x_O2, P=1e5, P_ref=1e5):
    dG = 237000 - 50 * (T - 298)
    V_standard = dG / (N * F)
    p_H2 = x_H2 * P
    p_H2O = x_H2O * P
    p_O2 = x_O2 * P
    nernst_term = (R_GAS * T) / (N * F) * np.log((p_H2 * p_O2**0.5) / (p_H2O * P_ref**0.5))
    return V_standard + nernst_term