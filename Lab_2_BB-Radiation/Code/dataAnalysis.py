import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
import math
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")


# input: wavelength in meters
def refractiveIndex(wv):
    wv = wv * 1e6  # convert wavelength to microns
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

    data = readRaw(filename)
    x_axis = np.array(data["angle (degrees) - Plot 0"])
    y_axis = np.array(data["thermopile signal (volts) - Plot 0"])

    if "600C" in filename:
        mask = y_axis <= 0.001
        x_axis = x_axis[mask]
        y_axis = y_axis[mask]

    else:
        mask = y_axis <= 0.01
        x_axis = x_axis[mask]
        y_axis = y_axis[mask]
    plt.ylim(0, 0.009)
    plt.scatter(x_axis, y_axis, label=filename[3:-4])

    wvRange = np.linspace(0.2e-6, 11e-6, 120)
    deflections = np.vectorize(angularDeflection)(wvRange)

    plt.legend()
    plt.xlabel("Angle")
    plt.ylabel("Intensity")
    plt.title("Intensity vs. Angle")
    # plt.savefig(f"Lab_2_BB-Radiation/Images/{filename}_plot.png")
    plt.clf()


if __name__ == "__main__":
    # for temp in [600, 650, 700, 800, 900, 1000, 1100]:
    #     if temp == 700:
    #         (analyzeData(f"d4_{temp}C.txt"))
    #         continue
    #     (analyzeData(f"d3_{temp}C.txt"))

    print(angularDeflection(635e-9))
    # analyzeData(f"d3_600C.txt")
