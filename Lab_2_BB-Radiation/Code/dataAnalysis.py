import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
import math
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")


def analyzeData(filename):
    def readRaw(filename):
        data = pd.read_table("Lab_2_BB-Radiation/Raws/" + filename, delimiter="\t")
        return data

    # def exp_func(x, a, b, c):
    #     return a * math.e ** (-1 * b * x) + c

    data = readRaw(filename)
    x_axis = np.array(data["angle (degrees) - Plot 0"])
    y_axis = np.array(data["thermopile signal (volts) - Plot 0"])
    plt.ylim(0, .009)
    # params, covariance = curve_fit(exp_func, x_axis, y_axis)
    # params = tuple(params)
    # a, b, c = params
    # perr = np.sqrt(np.diag(covariance))
    # berr = perr[1]
    # print(f'bg = {backgroundAverage}')
    # print(f"The equation of the line is: y = -{a:.2f} exp(-{b:.5f}*x) + {c}")
    # print(f"The decay constant is {b:.5f} \u00B1 {berr:.5f}")
    plt.scatter(x_axis, y_axis, label=filename[3:-4])
    # plt.plot(x_axis, exp_func(x_axis, *params), label="Logarithmic Fit", color="red")
    plt.legend()
    plt.xlabel("Angle")
    plt.ylabel("Intensity")
    plt.title("Intensity vs. Angle")
    # plt.text(x_axis.max() / 3 * 2, y_axis.max() / 4 * 3,f'y = -{a:.2E} exp(-{b:.2E}•x) + {c:.2E}', horizontalalignment='center',
    #  verticalalignment='center', color='red')
    # plt.show()
    # halfLife = np.log(2) / b
    # halfLifeErr = np.log(2) / np.power(b, 2) * berr
    # print(f"Half life is {halfLife:.1f} \u00B1 {halfLifeErr:.1f}")
    plt.savefig(f"Lab_2_BB-Radiation/Images/{filename}_plot.png")
    plt.clf()
    # return {"halfLife": halfLife, "error": halfLifeErr}


if __name__ == "__main__":

    for temp in [600, 650, 700, 800, 900, 1000, 1100]:
        if temp == 700:
            (analyzeData(f"d4_{temp}C.txt"))
            continue
        (analyzeData(f"d3_{temp}C.txt"))
