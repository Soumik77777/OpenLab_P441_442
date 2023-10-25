import os
import csv
import numpy as np
import scipy as sp
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


#RungeKutta4
def RK4(x,y,z,h,range,f1,f2):
    X=[]
    Y=[]
    while x <= range:
        k1= h*f1(z)
        l1= h*f2(x,y,z)

        k2= h*f1(z+l1/2)
        l2= h*f2(x+h/2,y+k1/2, z+l1/2)

        k3= h*f1(z + l2 / 2)
        l3= h*f2(x + h / 2, y + k2 / 2, z + l2 / 2)

        k4= h*f1(z + l3)
        l4= h*f2(x + h, y + k3, z + l3)

        y= y+1/6*(k1 +2*k2 +2*k3 +k4)
        z= z+1/6*(l1 +2*l2 +2*l3 +l4)
        x= x+h


        X.append(x)
        Y.append(y)
    return X,Y
