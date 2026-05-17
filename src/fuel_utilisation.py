def calc_inlet_outlet(x_H2_in, x_H2O_in, FU):

    inlet = {
        'x_H2': x_H2_in,
        'x_H2O': x_H2O_in,
    }

    x_H2O_out = x_H2O_in - FU * x_H2O_in
    x_H2_out = x_H2_in + FU * x_H2O_in

    total = x_H2O_out + x_H2O_out

    outlet = {
        'x_H2': x_H2_out / total,
        'x_H2O': x_H2O_out / total,
    }

    return inlet, outlet