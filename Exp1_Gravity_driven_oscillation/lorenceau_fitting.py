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

location = 'D:/Study Table/P441 Lab/data_16th/16 th Sep/With glass pipe/'
filename = 'Set-5_(2)Glass.txt'
poptimeframes = 3                 # number of time frames to be deleted

data = read_csv(filename, location=location, poprows=2+poptimeframes, delimiter='\t')

## calibration
z_0 = initial_height(h, H, rho=1)
z_calib_factor = z_0 - (data[1][0] * 100)
time_calib_factor = data[0][0]

time_list, z_list = [], []

for i in range(len(data[0])):
    time_list.append(data[0][i] - time_calib_factor)
    z_list.append(data[1][i] * 100 + z_calib_factor)



def minimizing_chi_sqr(time_list, obs_data, h, omega_guess, limit=1, points=21,):
    omega_list = np.linspace(omega_guess - limit, omega_guess + limit, points)
    chi_list = np.zeros(len(omega_list))

    for i in range(len(omega_list)):
        fitted_data = Lorenceu_model(time_list, h, omega_list[i])
        chi_list[i] = chi_square(time_list, obs_data, fitted_data)

    omega_opt_index = least_element(chi_list)

    return omega_list[omega_opt_index]


time_data_lor = [i * (h*1e-2/9.8)**(-0.5) for i in time_list]
omega_final = minimizing_chi_sqr(time_data_lor, z_list, h, omega_guess=0.1, limit=.1)
print(omega_final)

fitted_data = Lorenceu_model(time_data_lor, h, omega_final)
plt.plot(time_list, z_list, '.', label='Observed Data')
plt.plot(time_list, fitted_data, c='r', label='Lorenceau Model Fitting')

plt.xlabel('Time (sec)')
plt.ylabel('Fluid Level (cm)')
plt.title('h= ' +str(h)+' cm, Fitted $\Omega$= '+str(round(omega_final, 4)))
plt.suptitle('Fluid oscillation fitted with Lorenceau Model')
plt.grid()
plt.legend()
plt.show()


'''
## FFT
plt.clf()
Z_FFT = np.fft.fft(fitted_data - np.mean(fitted_data))
z2 = Z_FFT * np.conjugate(Z_FFT)
pow = abs(z2[1:len(Z_FFT)//2] + z2[:len(Z_FFT)//2:-1])
pow = pow/np.max(pow)
DT = time_list[1]-time_list[0]   # sample time
freq = (np.fft.fftfreq(t_soln_sec.shape[0])/DT)[1:len(Z_FFT)//2]

plt.plot(freq, pow)
plt.show()
'''

