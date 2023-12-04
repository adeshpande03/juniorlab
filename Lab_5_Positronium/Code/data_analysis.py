import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import math
from pprint import pprint
import seaborn as sns

def linear_fit(bin, a, b):
    return a*bin+b
    # return a*bin

def exp_fit(time, A, lmbda, B):
    return A*np.exp(-lmbda*time)+B

# read CHN file
filepath = "Lab_5_Positronium/Data/11_15.Chn"
data = np.fromfile(filepath, dtype=np.int32, count=16384, offset=32)
# y = data[data > 0]
calibration_bins = [743, 1086, 1420, 1745, 2079, 2423, 2748, 3760, 6424, 9747, 13089, 13752]
calibration_delays = [10, 20, 30, 40, 50, 60, 70, 100, 180, 280, 380, 400]
calparams, calcovariance = curve_fit(linear_fit, calibration_bins, calibration_delays)
calparams = tuple(calparams)
a, b = calparams
time_space = np.linspace(linear_fit(1, a, b), linear_fit(16384, a, b), 16384)
# x = time_space[data > 0]
print(a,b)
# plt.scatter(calibration_bins, calibration_delays)
# plt.plot(range(1,16385), time_space)
# plt.show()

# para_params, para_covariance = curve_fit(exp_fit, time_space[820:980], data[820:980])
para_params, para_covariance = curve_fit(exp_fit, time_space[126:295], data[126:295], p0=(20, 2, 10))
print("para params", para_params)

# ortho_params, ortho_covariance = curve_fit(exp_fit, time_space[1000:], data[1000:])
# lmbdas = []
# for i in range(400, 1000):
#     ortho_params, ortho_covariance = curve_fit(exp_fit, time_space[i:8500], data[i:8500], p0=(12, .0072, 6))
#     lmbdas.append(ortho_params[1])
# plt.plot(time_space[400:1000], lmbdas)
# plt.show()

# lmbdas = []
# for j in range(3000, 16000, 250):
#     ortho_params, ortho_covariance = curve_fit(exp_fit, time_space[883:j], data[883:j], p0=(12, .0072, 6))
#     lmbdas.append(ortho_params[1])
# plt.plot(time_space[3000:16000:250], lmbdas)
# plt.show()

ortho_params, ortho_covariance = curve_fit(exp_fit, time_space[884:-6000], data[884:-6000], p0=(12, .0072, 6))
print(f"ortho_params: {ortho_params}")
print("ortho cov", np.sqrt(np.diag(ortho_covariance)))
df = {'Time': time_space, 'Counts': data}
df = pd.DataFrame(df)
df['Type'] = 'Excluded'
df.iloc[884:-6000, 2] = 'Ortho-positronium'
df.iloc[126:295, 2] = 'Para-positronium'
df.iloc[-6000:, 2] = 'Background'
# abcd = np.average(df.loc[df['Type'] == 'Background', 'Counts'])
# print(abcd)
df['Counts'] = df['Counts']

# fig = plt.figure()
# fig.set_figwidth(15)
# ax = fig.add_subplot()
# plot = ax.hist(time_space, time_space, weights=data, label="Data")
# plot = ax.hist(time_space, time_space, weights=data-ortho_params[2], label="Data")
# plot = ax.scatter(np.linspace(1,16384, 16384), data, label = "Data", s=2)
# plot = ax.plot(time_space, data, label = "Data", linewidth=1)
# plot = ax.scatter(time_space, data, label = "Data", s=2)
palette = sns.color_palette("Set2")
sns.lmplot(x='Time', y='Counts', data=df, fit_reg=False, hue='Type', palette='Set2', height=12, aspect=15/12, scatter_kws={"s": 5}, legend=False)
plt.plot(time_space, exp_fit(time_space, *para_params), label='Para-positronium fit', color=palette[5])
plt.yscale('log')
plt.xlabel('Time (ns)')
# plt.ylim(bottom=1)
# ax.set_xlim(right=600)
plt.legend()
plt.show()

plt.clf()
n = 10
bins, weights = np.asarray(df.iloc[884:-6000, 0]), np.asarray(df.iloc[884:-6000, 1])
bins = bins.reshape((-1,n))[:,0]
weights = np.sum(weights.reshape((-1,n)), axis=1)
ortho_params, ortho_covariance = curve_fit(exp_fit, bins, weights, p0=(12, .0072, 6))
print(f"ortho_params: {ortho_params}")
print("ortho cov", np.sqrt(np.diag(ortho_covariance)))
bins, weights = np.asarray(df.iloc[884:, 0]), np.asarray(df.iloc[884:, 1])
bins = bins.reshape((-1,n))[:,0]
weights = np.sum(weights.reshape((-1,n)), axis=1)
# new_hist = np.column_stack((bins,weights))
plt.scatter(bins, weights, color=palette[1], s=2, label='Data')
# plt.scatter(x=df.loc[df['Type']=='Ortho-positronium', 'Time'], y=df.loc[df['Type']=='Ortho-positronium', 'Counts'])
plt.plot(time_space, exp_fit(time_space, *ortho_params), label='Ortho-positronium fit', color=palette[0], linewidth=2)
plt.xlim(left=time_space[800],right=time_space[-220])
plt.xlabel('Time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()

print(list(np.sqrt(np.diag(ortho_covariance))))
print(list(ortho_params))
print(math.log(.5)/(-1 * list(ortho_params)[1]))

print(0.006395763216475263/0.0008291609473602737)