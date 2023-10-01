import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
import math
from pprint import pprint
import warnings

warnings.filterwarnings("ignore")


# def sanniFunction():
#     def makeDataframe(filename) :
#         return pd.read_excel(f"Lab_1_Half-Life/Data/Formatted/{filename}", header=None)

#     def logRegression(trial) :
#         indices = []
#         for i in range(len(trial[1])):
#             if trial[1][i] > 0:
#                 indices.append(i)
#         time = []
#         logs = []
#         for i in indices :
#             time.append(trial[0][i])
#             logs.append(np.log(trial[1][i]))
#         return np.polyfit(time, logs, 1)
#         # return np.polyfit(trial[0], np.log(trial[1], where= trial[1] > 0), 1)

#     def logPlot(trial) :
#         a,b = logRegression(trial)
#         scatterPlot = plt.scatter(trial[0], np.log(trial[1]), color='teal')
#         regressionPlot = plt.plot(trial[0], a*trial[0]+b, color='red')
#         plt.show()
#         return a,b

# trial1 = makeDataframe("LiquidTrial_1_Excel.xlsx")
# trial2 = makeDataframe("LiquidTrial_2_Excel.xlsx")
# trial3 = makeDataframe("LiquidTrial_3_Excel.xlsx")
# trial4 = makeDataframe("LiquidTrial_4_Excel.xlsx")

# plots1 = logPlot(trial1)
# plots2 = logPlot(trial2)
# plots3 = logPlot(trial3)
# plots4 = logPlot(trial4)

# print(np.log(2)/(-plots1[0]))
# print(np.log(2)/(-plots2[0]))
# print(np.log(2)/(-plots3[0]))
# print(np.log(2)/(-plots4[0]))

# trial5 = makeDataframe("day3_liquidtrial1.xlsx")
# a,b = logPlot(trial5)
# print(np.log(2)/(-a)/3)
# sanniFunction()


def analyzeData(filename):
    def readRaw(filename):
        data = pd.read_table(
            "Lab_1_Half-Life/Data/Raws/" + filename, delimiter="\t", header=None
        )
        return data

    def getBackgroundAverage(filename):
        data = readRaw(filename)
        y_axis = np.array(data[1])
        avg = np.average(y_axis)
        std = np.std(y_axis)
        # print(avg, std)
        return avg

    def exp_func(x, a, b, c):
        return a * math.e ** (-1 * b * x) + c

    data = readRaw(filename)
    x_axis = np.array(data[0])
    backgroundAverage = getBackgroundAverage("backgroundradiation")
    y_axis = np.array(data[1])
    params, covariance = curve_fit(exp_func, x_axis, y_axis)
    params = tuple(params)
    a, b, c = params
    perr = np.sqrt(np.diag(covariance))
    aerr = perr[0]
    berr = perr[1]
    cerr= perr[2]
    # print(f'bg = {backgroundAverage}')
    # print(f"The equation of the line is: y = ({a:.3} +- {aerr:.3}) exp(-({b:.3} +- {berr:.3})*x) + {c:.3} +- {cerr:.3}")
    # print(f"The decay constant is {b:.5f} \u00B1 {berr:.5f}")
    plt.scatter(x_axis, y_axis, label="Data")
    plt.plot(x_axis, exp_func(x_axis, *params), label="Logarithmic Fit", color="red")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Gamma Ray Counts (Bq)")
    plt.title("Barium-137 Gamma Ray Counts vs. Time")
    plt.text(x_axis.max() / 3 * 2, y_axis.max() / 4 * 3,f'y = {a:.2E} exp(-{b:.2E}•x) + {c:.2E}', horizontalalignment='center',
     verticalalignment='center', color='red')
    halfLife = np.log(2) / b
    halfLifeErr = np.log(2) / np.power(b, 2) * berr
    print(f"Half life is {halfLife:.1f} \u00B1 {halfLifeErr:.1f}")
    plt.savefig(f"Lab_1_Half-Life/Images/{filename}_plot.png")
    plt.clf()
    return {"decayconstant": b, "dcerror": berr, "halfLife": halfLife, "hferror": halfLifeErr}
        


if __name__ == "__main__":
    (analyzeData("day3_liquidtrial1"))
    (analyzeData("day3_liquidtrial2"))
    (analyzeData("day3_liquidtrial3"))
    (analyzeData("day3_liquidtrial4"))
    (analyzeData("day3_liquidtrial5"))
    (analyzeData("day3_liquidtrial6"))
    # filename = "backgroundradiation"
    # data = pd.read_table("Lab_1_Half-Life/Data/Raws/" + filename, delimiter="\t", header=None)
    # y_axis = np.array(data[1])
    # x_axis = np.array(data[0])
    # plt.scatter(x_axis, y_axis)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Background Radiation Count (Bq)")
    # plt.title("Background Radiation Counts vs. Time")
    # plt.savefig(f"Lab_1_Half-Life/Images/{filename}_plot.png")
    # plt.clf()
    
