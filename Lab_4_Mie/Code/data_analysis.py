import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import math
from pprint import pprint

def readRaw(filename):
        data = pd.read_excel("Lab_4_Mie/Data/" + filename)
        return data

def fit(theta, a, b, k, phi):
    return np.abs(a)*np.cos(k*(theta*np.pi/180) + phi)**2+b
    
def analyze(transverse_data, axial_data):
    x_axis = np.array(transverse_data[0]["Polarization (±0.5 deg)"])
    transverse_values = []
    axial_values = []
    for i in transverse_data:
        transverse_values.append(np.array(i["Voltage Reading (±0.001 mV)"]))
    for i in axial_data:
        axial_values.append(np.array(i["Voltage Reading (±0.001 mV)"]))
    transverse_values = np.asarray(transverse_values)
    axial_values = np.asarray(axial_values)
    transverse_avgs = np.mean(transverse_values, axis=0)
    axial_avgs = np.mean(axial_values, axis=0)
    # thalf1 = transverse_avgs[:37]
    # thalf2 = np.flip(transverse_avgs[36:])
    # ahalf1 = axial_avgs[:37]
    # ahalf2 = np.flip(axial_avgs[36:])
    # t = (thalf1 + thalf2)/2
    # a = (ahalf1 + ahalf2)/2
    # pprint(transverse_avgs)
    # pprint(axial_avgs)
    # plt.plot(x_axis[:37], t)
    # plt.plot(x_axis[:37], a)
    plt.figure(figsize=(6,4))
    plt.scatter(x_axis, transverse_avgs, s=5, label="Transverse Data")
    plt.scatter(x_axis, axial_avgs, s=5, label="Axial Data")
    
    params, covariance = curve_fit(fit, x_axis, transverse_avgs, sigma=np.repeat(0.001, len(transverse_avgs)), p0=[0.2, 0.3, 2, np.pi/4])
    ta, tb, tk, tphi = tuple(params)
    taerr, tberr, tkerr, tphierr = np.sqrt(np.diag(covariance))
    params, covariance = curve_fit(fit, x_axis, axial_avgs, sigma=np.repeat(0.001, len(axial_avgs)), p0=[0.2, 0.3, 2, np.pi/4])
    aa, ab, ak, aphi = tuple(params)
    aaerr, aberr, akerr, aphierr = np.sqrt(np.diag(covariance))
    print(ta,tb,tk,tphi*180/np.pi)
    print(taerr, tberr, tkerr, tphierr*180/np.pi)
    print(aa, ab, ak, aphi*180/np.pi)
    print(aaerr, aberr, akerr, aphierr*180/np.pi)
    plt.plot(x_axis, fit(x_axis, ta, tb, tk, tphi), label="Transverse Fit")
    plt.plot(x_axis, fit(x_axis, aa, ab, ak, aphi), label="Axial Fit")
    plt.xlabel("Half-Waveplate Angle (degrees)")
    plt.ylabel("Voltage Reading (mV)")
    plt.legend(
        bbox_to_anchor=(0.15, 1.15), 
        loc="upper left",
        ncol=2,
        # fancybox=True,
        shadow=True
        )
    
    plt.show()
    # plt.savefig("Lab_4_Mie/plot.png")

if __name__ == "__main__":
    transverse_data = []
    axial_data = []
    for i in range(1, 4):
        transverse_data.append(readRaw(f"0 Trial {i}.xlsx"))
        axial_data.append(readRaw(f"90 Trial {i}.xlsx"))
    analyze(transverse_data, axial_data)

    
    

