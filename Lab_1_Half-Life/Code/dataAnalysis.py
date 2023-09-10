import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root

def analyzeData(filename):

    def neg_ln_func(x, a, b):
        return -a * np.log(x) + b
    data = pd.read_excel("Lab_1_Half-Life/Data/Formatted/" + filename, header=None)
    x_axis = np.array(data[0])
    y_axis = np.array(data[1])
    params, covariance = curve_fit(neg_ln_func, x_axis, y_axis)
    a, b = params
    print(f"The equation of the line is: y = -{a:.2f} ln(x) + {b:.2f}")

    plt.scatter(x_axis, y_axis, label='Data')
    plt.plot(x_axis, neg_ln_func(x_axis, *params), label='Fitted Curve', color='red')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Logarithmic Curve Fit')
    print("The half life is approximately", np.exp((.5 * b - b)/(-1 * a)) )
    plt.show()
    
(analyzeData("LiquidTrial_1_Excel.xlsx"))
# (analyzeData("LiquidTrial_2_Excel.xlsx"))
# (analyzeData("LiquidTrial_3_Excel.xlsx"))
# (analyzeData("LiquidTrial_4_Excel.xlsx"))