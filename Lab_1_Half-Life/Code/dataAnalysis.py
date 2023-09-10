import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def makeDataframe(filename) :
    return pd.read_excel(f"Lab_1_Half-Life/Data/Formatted/{filename}", header=None)

def logRegression(trial) :
    indices = []
    for i in range(len(trial[1])):
        if trial[1][i] > 0:
            indices.append(i)
    time = []
    logs = []
    for i in indices :
        time.append(trial[0][i])
        logs.append(np.log(trial[1][i]))
    return np.polyfit(time, logs, 1)
    # return np.polyfit(trial[0], np.log(trial[1], where= trial[1] > 0), 1)

def logPlot(trial) :
    a,b = logRegression(trial)
    scatterPlot = plt.scatter(trial[0], np.log(trial[1]), color='teal')
    regressionPlot = plt.plot(trial[0], a*trial[0]+b, color='red')
    plt.show()
    return a,b

trial1 = makeDataframe("LiquidTrial_1_Excel.xlsx")
trial2 = makeDataframe("LiquidTrial_2_Excel.xlsx")
trial3 = makeDataframe("LiquidTrial_3_Excel.xlsx")
trial4 = makeDataframe("LiquidTrial_4_Excel.xlsx")

plots1 = logPlot(trial1)
plots2 = logPlot(trial2)
plots3 = logPlot(trial3)
plots4 = logPlot(trial4)

print(np.log(2)/(-plots1[0]))
print(np.log(2)/(-plots2[0]))
print(np.log(2)/(-plots3[0]))
print(np.log(2)/(-plots4[0]))
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
# (analyzeData("Li