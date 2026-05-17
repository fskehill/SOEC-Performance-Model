import numpy as np

# Digitised from Jensen (2007) Risø-PhD-29
# Conditions: 800°C, 70% H2O + 30% H2 inlet, 1 atm

j_exp = [0.0, 0.2, 0.4, 0.6, 0.8]
V_exp = [1.00, 1.06, 1.12, 1.19, 1.27]

def calc_error(j_model, V_model, j_exp, V_exp):
    V_interp = np.interp(j_exp, j_model, V_model)
    errors = (V_interp - V_exp) / V_exp * 100
    return errors, V_interp