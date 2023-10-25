import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

'''
def st_line(x, m, c):
 return m*x + c


freq = [1.77, 1.80, 1.93, 2.02, 2.23]
height = [8.0, 7.7, 6.8, 5.8, 5.0]


freq_sq, height_inv = [], []
for i in range(len(freq)):
 freq_sq.append(freq[i]**2)
 height_inv.append(float(1/height[i]))

popt, pcov = curve_fit(st_line, height_inv, freq_sq)
print(popt)
x = np.linspace(0.12, 0.22)
plt.plot(height_inv, freq_sq, '.', label='Datapoints')
plt.plot(x, st_line(x, *popt), label='$f^2= (1/h)*(g/4 \pi^2)$')
plt.plot([ ], [ ], ' ', label='m= g/4 $\pi^2$ = '+ str(round(popt[0], 2))+'cm/ $s^2$')
plt.plot([ ], [ ], ' ', label='g= 949.85 cm/ $s^2$')


plt.xlabel("1/h ($cm^{-1}$)")
plt.ylabel("Frequency$^2$ (Hz$^2$)")
plt.grid()
plt.legend()
#plt.title("Estimation of g from (1/h) vs $f^2$ plot, Ethanol Solution")
plt.show()
'''

height = [3.9, 4.4, 5.8, 7.6]
b_dash = [12.11, 13.94, 16.34, 17.91]

plt.plot(height, b_dash, '.-')
plt.xlabel("h (cm)")
plt.ylabel("b' (sec$^{-1}$)")
plt.grid()
plt.show()

