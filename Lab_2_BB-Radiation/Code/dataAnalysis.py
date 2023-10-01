import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import math
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")


# input: wavelength in microns
def refractiveIndex(wv):
    wvSquared = wv**2
    n = np.sqrt(
        1.33973
        + 0.81070 * wvSquared / (wvSquared - 0.10065**2)
        + 0.19652 * wvSquared / (wvSquared - 29.87**2)
        + 4.52469 * wvSquared / (wvSquared - 53.82**2)
    )
    return n


def angularDeflection(wv):
    n = refractiveIndex(wv)
    phi = np.deg2rad(47.447)
    alpha = np.deg2rad(60)
    d = phi + np.arcsin(n * np.sin(alpha - np.arcsin(np.sin(phi) / n))) - alpha
    d = np.rad2deg(d)
    return d


def analyzeData(filename):
    def readRaw(filename):
        data = pd.read_table("Lab_2_BB-Radiation/Raws/" + filename, delimiter="\t")
        return data

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def planck_law(wv, a, b):
        return a / (wv**5) * 1 / (np.exp(b / wv) - 1)

    data = readRaw(filename)
    x_axis = np.array(data["angle (degrees) - Plot 0"])
    y_axis = np.array(data["thermopile signal (volts) - Plot 0"])

    if "600C" in filename:
        mask = (y_axis <= 0.0007) & (y_axis >= -0.001)

        # Filter both x and y using the mask
        x_axis = x_axis[mask]
        y_axis = y_axis[mask]

    elif "700C" in filename:
        mask = (y_axis <= 0.005) & (y_axis >= -0.001)

        # Filter both x and y using the mask
        x_axis = x_axis[mask]
        y_axis = y_axis[mask]

    else:
        mask = (y_axis <= 0.01) & (y_axis >= -0.001)

        # Filter both x and y using the mask
        x_axis = x_axis[mask]
        y_axis = y_axis[mask]
    # plt.ylim(0, 0.009)
    # plt.scatter(x_axis, y_axis, label=f"$T = ${filename[3:-5]}$\degree$C", s=3)

    wvRange = np.linspace(0.2, 11, 1000)
    deflections = np.vectorize(angularDeflection)(wvRange)
    # deltafunction = interp1d(wvRange, deflections, kind="cubic")
    lambdafunction = interp1d(deflections, wvRange, kind="cubic")
    grad = -1 * np.gradient(deflections, wvRange)
    # gradfunction = interp1d(wvRange, grad, kind="cubic")
    wavelengths = np.vectorize(lambdafunction)(x_axis)
    grads = []
    for i in wavelengths:
        grads.append(grad[find_nearest(wvRange, i)])
    grads = np.asarray(grads)
    bofWV = np.multiply(y_axis, grads)
    # params, covariance = curve_fit(planck_law, np.delete(wavelengths, [80, 81, 91, 92]), np.delete(bofWV, [80, 81, 91, 92]), p0=[10,10])
    params, covariance = curve_fit(planck_law, wavelengths, bofWV, p0=[10, 10])
    params = tuple(params)
    a, b = params
    perr = np.sqrt(np.diag(covariance))
    aerr = perr[0]
    berr = perr[1]
    # print(b*(int(filename[3:-5])+273.15)*1e-6, berr*(int(filename[3:-5])+273.15)*1e-6)
    pl = planck_law(wvRange, *params)
    wvMax = wvRange[np.argmax(pl)]
    integral = simpson(pl, wvRange)

    plt.scatter(wavelengths, bofWV, label=f"$T = ${filename[3:-5]}$\degree$C", s=2)
    plt.plot(wvRange, planck_law(wvRange, *params))
    # plt.plot(wavelengths, planck_law(wavelengths, *(params+perr)), color="green", linestyle='dashed')
    # plt.plot(wavelengths, planck_law(wavelengths, *(params-perr)), color="green", linestyle='dashed')

    # plt.xlim(left = 1)
    # plt.ylim(top = 0.0025)
    plt.legend()
    # plt.xlabel("$\delta$   (degrees)")
    plt.xlabel("$\lambda$ ($\mu$m)")
    # plt.ylabel("$B(\delta, T)$")
    plt.ylabel("$B(\lambda, T)$")
    # plt.title("Relative Spectral Radiance vs. Deflection Angle")
    plt.title("Relative Spectral Radiance vs. Wavelength")
    # plt.show()
    plt.savefig(f"Lab_2_BB-Radiation/Images/{filename}_angleplot.png")
    plt.clf()
    error = 1e-6 * np.sqrt(
        b**2 * 0.1**2 + berr**2 * (int(filename[3:-5]) + 273.15) ** 2
    )
    # return (b*(int(filename[3:-5])+273.15)*1e-6, berr*(int(filename[3:-5])+273.15)*1e-6, wvMax, integral)
    return (b * (int(filename[3:-5]) + 273.15) * 1e-6, error, wvMax, integral)


def wien_law(temp, a):
    return a / temp


def stefanboltzmann_law(temp, a):
    return temp**4 * a


if __name__ == "__main__":
    temps = [600, 650, 700, 800, 900, 1000, 1100]
    bs = []
    berrs = []
    wvMaxes = []
    integrals = []
    for temp in temps:
        if temp == 700:
            output = analyzeData(f"d4_{temp}C.txt")
        else:
            output = analyzeData(f"d3_{temp}C.txt")
        bs.append(output[0])
        berrs.append(output[1])
        wvMaxes.append(output[2])
        integrals.append(output[3])
        # print(output)
    # print(angularDeflection(635e-9))
    # analyzeData(f"d3_600C.txt")

    # get average hc/k
    b = np.mean(bs)
    berr = np.linalg.norm(berrs)
    print(b, berr)

    # graph lambda_max vs temperature
    temps = np.asarray(temps)
    params, covariance = curve_fit(wien_law, temps + 273.15, wvMaxes)
    a = params[0]
    perr = np.sqrt(np.diag(covariance))
    aerr = perr[0]
    plt.scatter(1 / (temps + 273.15), wvMaxes)
    plt.plot(1 / (temps + 273.15), wien_law(temps + 273.15, a), color="red")
    plt.xlabel("$1/T$ (K$^{-1}$)")
    plt.ylabel("$\lambda_{max}$ ($\mu$m)")
    # plt.show()
    plt.savefig(f"Lab_2_BB-Radiation/Images/wienslawplot.png")
    plt.clf()
    print(a, aerr)
    # residuals = wvMaxes - wien_law(temps+273.15, a)
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((wvMaxes-np.mean(wvMaxes))**2)
    # r_squared = 1- ss_res/ss_tot
    # print(r_squared)

    # graph total power vs temperature]
    params, covariance = curve_fit(stefanboltzmann_law, temps + 273.15, integrals)
    a = params[0]
    perr = np.sqrt(np.diag(covariance))
    aerr = perr[0]
    plt.scatter((temps + 273.15) ** 4, integrals)
    plt.plot((temps + 273.15) ** 4, stefanboltzmann_law(temps + 273.15, a), color="red")
    plt.xlabel("$T^4$ (K$^4$)")
    plt.ylabel("$\int B(\lambda,T) d\lambda$")
    # plt.show()
    plt.savefig(f"Lab_2_BB-Radiation/Images/stefanboltzmannplot.png")
    print(a, aerr)
    # residuals = integrals - stefanboltzmann_law(temps+273.15, a)
    # ss_res = np.sum(residuals**2)
    # ss_tot = np.sum((integrals-np.mean(integrals))**2)
    # r_squared = 1- ss_res/ss_tot
    # print(r_squared)
