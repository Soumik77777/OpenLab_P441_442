import numpy as np
import scipy as sp
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline


def sq(time_list, obs_data, fitted_data):
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


# Newton's law model
def DZ_dt_Newton(Z, t, args):
    h = args[0]
    g = args[1]
    b = args[2]
    return [Z[1], -Z[1] ** 2 / Z[0] - g + g * h / Z[0] - b * Z[1] / Z[0]]


def Newton_model(time_list, h, b, g=980):
    params = (h, g, b)
    # solve Newton model:
    t_soln = time_list
    Z_soln_Newton = sp.integrate.odeint(DZ_dt_Newton, [0.02, 0], t_soln, args=(params,))

    z_soln_Newton = Z_soln_Newton[:, 0]  # fluid height

    return z_soln_Newton


def DZ_dt_Lor(Z, t, args):
    Omeg = args[2]
    if Z[1] > 0:
        return [Z[1], 1 / Z[0] - 1 - Omeg * Z[1] - (Z[1]) ** 2 / Z[0]]
    else:
        return [Z[1], 1 / Z[0] - 1 - Omeg * Z[1]]


def Lorenceu_model(time_list, h, omega, g=980):
    params = (h, g, omega)

    t_soln = time_list
    Z_soln = sp.integrate.odeint(DZ_dt_Lor, [0.02, 0.00], t_soln, args=(params,))

    z_soln = Z_soln[:, 0] * h  # fluid height

    return z_soln


# Finding optimum b
def minimizing_chi_sqr(time_list, obs_data, h, b_guess, limit=10):
    b_list = np.linspace(b_guess - limit, b_guess + limit, 20 * limit + 10)
    chi_list = np.zeros(len(b_list))

    for i in range(len(b_list)):
        fitted_data = Newton_model(time_list, h, b_list[i])
        chi_list[i] = sq(time_list, obs_data, fitted_data)

    b_opt_index = least_element(chi_list)

    return b_list[b_opt_index]


# Finding optimum omega
def min_chi_sqr(time_list, obs_data, h, o_guess, limit=0.15):
    o_list = np.linspace(o_guess - limit, o_guess + limit + 0.1, 30)
    chi_list = np.zeros(len(o_list))

    for i in range(len(o_list)):
        fitted_data = Lorenceu_model(time_list, h, o_list[i])
        chi_list[i] = sq(time_list, obs_data, fitted_data)

    o_opt_index = least_element(chi_list)

    return o_list[o_opt_index]


# user-modified area: our data:

# filename= 'set7.dat'
filename = 'D:/Study Table/P441 Lab/data_16th/16 th Sep/With Ethanol/Set-7_(1)Eth.txt'
# timeshift = 0.5333333333333333 # --shifted so that t=0 is the time in video that the cap is released.
timeshift = 0.067
data = np.genfromtxt(filename, delimiter='\t', skip_header=2)
time_data = data[:, 0]
z_data = data[:, 1]

time_data_clean = time_data[np.isfinite(
    z_data)] - timeshift  # cleaned data includes the timeshift, and removes any instances of infinite.

z_data_clean = z_data[np.isfinite(z_data)]

# cleansing data to start rise from t=0,z=0
time_data_clean = time_data_clean[4:]  # ----------------
z_data_clean = z_data_clean[4:] * 100  # to cm -----------

plt.plot(time_data_clean, z_data_clean, 'r.')
plt.xlabel('time in s ', fontsize=15)
plt.ylabel('fluid level in cm ', fontsize=15)
#plt.show()

# calculate the initial starting level of the fluid in the submersed straw using fluid statics:
rho_water = 0.8945  # g/cm^3
g = 980
P_atm = 1030  # g/cm^2
h = 7.6  # cm -------------------------
r = 0.265  # cm -----------------
H = 13.9  # length of straw, cm

# fiting newton model
time_axis1 = time_data_clean
z_data1 = z_data_clean

bl = minimizing_chi_sqr(time_axis1, z_data1, h, 24)

plt.plot(time_axis1, z_data1, 'b.', label='Data')
z_soln_Newton = Newton_model(time_axis1, h, bl)
print("The damping co-efficient b'=", bl * rho_water * math.pi * r ** 2)
plt.plot(time_axis1, z_soln_Newton, 'r', label='Fitted Newtonian model')
plt.legend()
#plt.show()

# fiting lorenceu model
plt.plot(time_axis1, z_data1, 'b.', label='Data')
time_Lor = time_axis1 * (h * 1e-2 / 9.8) ** (-0.5)  # dimensionless data

ol = min_chi_sqr(time_Lor, z_data1, h, 0.16)

z_soln_Lor = Lorenceu_model(time_Lor, h, ol)
plt.plot(time_Lor * (h * 1e-2 / 9.8) ** 0.5, z_soln_Lor, 'g', label='Fitted Lorenceau model')
plt.xlabel('time (sec)', fontsize=15)
plt.ylabel('fluid level (cm)', fontsize=15)
plt.legend()
#plt.show()

# plot solution for longer time so as to get denser data points in the frequency spectrum:

t_soln_sec = time_Lor * (h * 1e-2 / 9.8) ** 0.5
plt.clf()
# plt.plot(t_soln,z_soln_Newton,'r',label='ode solution')
plt.plot(time_Lor * (h * 1e-2 / 9.8) ** 0.5, z_soln_Lor, 'g', label='ode solution')

plt.plot(time_axis1, z_data1, 'b*', label='y data')
# plt.xlim([-.1,5])
#plt.show()
plt.clf()
z_soln = z_data1

# perform the discrete Fourier transform of the data

Z_FFT = np.fft.fft(z_soln - np.mean(z_soln))
z2 = Z_FFT * np.conjugate(Z_FFT)
pow = abs(z2[1:len(Z_FFT) // 2] + z2[:len(Z_FFT) // 2:-1])
pow = pow / np.max(pow)
DT = t_soln_sec[1] - t_soln_sec[0]  # sample time
freq = (np.fft.fftfreq(t_soln_sec.shape[0]) / DT)[1:len(Z_FFT) // 2]

# check the power spectrum:

plt.plot(freq, pow, 'r.-')

plt.xlabel('frequency', fontsize=15)
plt.ylabel('Power density', fontsize=15)

plt.title('Power spectrum curve', fontsize=20)
plt.xlim([0, 6])
# plt.savefig('power spectrum.png',dpi=400)
plt.show()

m = np.argmax(pow)
print("Frequency from the curve: ", freq[m], 'Hz')

f_est = 1 / (2 * np.pi) * np.sqrt(g / h)
print('Small oscillation frequency = %2.3f Hz' % (f_est))


def Lorentzian(x_val, h, w, x_c, y_0):
    return ((h * w**2)/((w**2)+(4*(x_val - x_c)**2)) + y_0)
h_g   = 7.6    #h = height
w_g   = 1        #w = fwhm
x_c_g = 1.5        #x_c = x val of peak
y_0_g = 0           #y_0 = y val of asymptote


fit_from_Hz = 0
fit_to_Hz = 4

fit_from = int(np.round(np.interp(fit_from_Hz,freq,np.arange(len(freq)))))
fit_to = int(np.round(np.interp(fit_to_Hz,freq,np.arange(len(freq)))))



#best fit lines (guesses help the process)
p_guess = [h_g, w_g, x_c_g, y_0_g]
peak, pcov = sp.optimize.curve_fit(Lorentzian, freq[fit_from:fit_to],pow[fit_from:fit_to], p0 = p_guess)

perr = np.sqrt(np.diag(pcov))
plt.plot(freq[:10],pow[:10],'bo',label = 'power spectrum')
# plt.plot(freq[:100], Lorentzian(freq[:100], *p_guess), 'g--',label='guess')
plt.plot(freq[:10], Lorentzian(freq[:10], *peak), 'r',label='Lorentzian fit')
plt.grid()

plt.axvline(x=1.61)
# plt.axvline(x=fit_to_Hz)

xaxis_label = 'frequency (Hz)'
yaxis_label = 'signal (a.u.)'
plt.xlabel(xaxis_label,fontsize=15)
plt.ylabel(yaxis_label,fontsize=15)
plt.legend(frameon=False,loc=2)
plt.savefig('fig3 - frequency spectrum.png',dpi=400)
plt.show()
