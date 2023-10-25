import os
import csv
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt





def initial_height(h, H=13.95, rho=1, P_atm=1030):
    return (1 - P_atm / (rho * h + P_atm)) * H



# Newton model
def DZ_dt_Newton(Z, t, args):
    h = args[0]
    g = args[1]
    b = args[2]
    return [Z[1], -Z[1] ** 2 / Z[0] - g + g * h / Z[0] - b * Z[1] / Z[0]]


def Newton_model(time_list, h, b, g=980, H=13.95, rho=1, P_atm=1030):
    params = (h, g, b)

    z_0 = initial_height(h, H=H, rho=rho, P_atm=P_atm)

    Z_soln_Newton = sp.integrate.odeint(DZ_dt_Newton, [z_0, 0], time_list, args=(params,))
    z_soln_Newton = Z_soln_Newton[:, 0]

    return z_soln_Newton



## Lorenceau
def DZ_dt_Lor(Z, t, args):
    Omeg = args
    if Z[1] > 0:
        return [Z[1], 1 / Z[0] - 1 - Omeg * Z[1] - (Z[1]) ** 2 / Z[0]]
    else:
        return [Z[1], 1 / Z[0] - 1 - Omeg * Z[1]]


def Lorenceu_model(time_list, h, omega):
    params = (omega)

    z_0 = initial_height(h)

    Z_soln = sp.integrate.odeint(DZ_dt_Lor, [z_0, 0.00], time_list, args=(params,))
    z_soln = Z_soln[:, 0] * h

    return z_soln





def chi_square(time_list, obs_data, fitted_data):
    if len(time_list) != len(obs_data) or len(time_list) != len(fitted_data):
        print("You don't have data for all input time frame.")
        return 0
    else:
        chi_sqr = 0
        for i in range(len(time_list)):
            chi_sqr += (obs_data[i] - fitted_data[i]) ** 2 / fitted_data[i]

        return chi_sqr


def least_element(list):
    least_index, least_item = 0, list[0]
    for i in range(len(list) - 1):
        if list[i + 1] < least_item:
            least_index = i + 1
            least_item = list[i + 1]
    return least_index



def read_csv(filename, location=None, poprows=None, delimiter=None):
    if location == 0:
        filepath = filename
    elif location != None and location != 0:
        filepath = str(location) + str(filename)
    else:
        filepath = str(os.getcwd()) + str('\\') + str(filename)

    if delimiter == '\t':
        delim = '\t'
    else:
        delim = ','

    with open(filepath, 'r') as infile:
        data = csv.reader(infile, delimiter=delim)
        datalist, rows = [], []
        for row in data:
            datalist.append(row)
        if poprows != None:
            for i in range(poprows):
                datalist.pop(0)
        for j in range(len(datalist[0])):
            globals()['string%s' % j] = []
            for k in datalist:
                globals()['string%s' % j].append(float(k[j]))
            rows.append(globals()['string%s' % j])
        infile.close()

    return rows




