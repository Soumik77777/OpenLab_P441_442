import os
import csv
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from library import *



## Params
H = 61.5                   # height of straw in cm
g = 984                     # accelaration due to gravity in cm/s^2
R = 1.1                   # diameter of straw in cm

## LOAD DATA
h = 5.5                     # height of submerged liquid

location = 'D:/Study Table/P441 Lab/data_16th/16 th Sep/With Ethanol/'
filename = 'Set-5_(1)Eth.txt'
poptimeframes = 3                 # number of time frames to be deleted

data = read_csv(filename, location=location, poprows=2+poptimeframes, delimiter='\t')

## calibration
z_0 = initial_height(h, H, rho=0.8945)
z_calib_factor = z_0 - (data[1][0] * 100)
time_calib_factor = data[0][0]

time_list, z_list = [], []

for i in range(len(data[0])):
    time_list.append(data[0][i] - time_calib_factor)
    z_list.append(data[1][i] * 100 + z_calib_factor)


## Chi-sqr FITTING

def minimizing_chi_sqr(time_list, obs_data, h, b_guess, limit=10, points=21, H=13.95, g=980):
    b_list = np.linspace(b_guess-limit, b_guess+limit, points)
    chi_list = np.zeros(len(b_list))

    for i in range(len(b_list)):
        fitted_data = Newton_model(time_list, h, b_list[i], g=g, H=H)
        chi_list[i] = chi_square(time_list, obs_data, fitted_data)

    b_opt_index = least_element(chi_list)

    return b_list[b_opt_index], b_list, chi_list


b_final, b_list, chi_list = minimizing_chi_sqr(time_list, z_list, h, b_guess=11.1863, limit=0.001, points=21, g=g)
print(b_final)

b_dash = b_final * 0.8945 * np.pi * (R**2) / 4       # b' = b * rho * A


fitted_data = Newton_model(time_list, h, b_final, g=g, H=H)
plt.plot(time_list, z_list, '.', label='Observed Data')
plt.plot(time_list, fitted_data, c='r', label='Newtonian Model Fitting')


plt.xlabel('Time (sec)')
plt.ylabel('Fluid Level (cm)')
plt.title("h= " +str(h)+" cm, Fitted b= "+str(round(b_final, 4))+" cm s$^{-1}$, b'= "+str(round(b_dash, 4))+" s$^{-1}$")
plt.suptitle('Fluid oscillation fitted with Newtonian Model')
plt.grid()
plt.legend()
plt.show()

'''
## Relation between h and b
h = [7.6, 6.5, 5.8, 5.0, 4.4, 3.9]
b = [25.2517, 21.5224, 21.6668, 17.9412, 21.5641, 18.0105]
plt.clf()
plt.scatter(h, b)
plt.show()
'''




'''
## Scipy-fitting
def newton_dir_fitting(t, b):
    def DZ_dt_Newton(Z, t, args):
        b = args
        h = 5.8
        g = 984
        return [Z[1], -Z[1] ** 2 / Z[0] - g + g * h / Z[0] - b * Z[1] / Z[0]]

    z_0 = initial_height(h)
    z_soln = sp.integrate.odeint(DZ_dt_Newton, [z_0, 0], t, args=(b,))
    z_soln_Newton = z_soln[:, 0]
    return z_soln_Newton

print(curve_fit(newton_dir_fitting, time_list, z_list))
'''



'''
## Plotting chi-sq vs b

x = b_list
y = chi_list
fig, ax = plt.subplots()

ax.set_xlabel('b (cm/s)')
ax.set_ylabel('$\chi ^2$ value (a.u.)')
ax.set_title('Distribution of $\chi ^2$ for different values of b')
ax.set_xticks(x)
ax.set_xticklabels(x)

pps = ax.bar(x, y, width=2, label='population')
for p in pps:
   height = round(p.get_height(), 2)
   ax.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 3), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom')

plt.show()

'''