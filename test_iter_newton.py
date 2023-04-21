import numpy as np
import matplotlib.pyplot as plt
import time

def divid_diff_calc_iter(x, y, end):
    a = np.empty((end,end))

    for i in range(0, end+1):
        a[i,i] = y[i]

    for i in range(end):
        for j in range(i+1, end):
            a[i,j] = (a[i+1, j]-a[i, j-1])/(x[j]-x[i])
            a[i+1, j+1] = (a[i+2, j+1]-a[i+1, j])/(x[j]-x[i])
    return a[0,end]

x = np.array([0, 1, 2, 3])
y = np.array([-5, -6, -1, 16])
xi = np.array([0.5, 1.5, 2.5])

div_diff_array2 = np.empty(len(x))
for i in range(len(x)):
    div_diff_array2[i] = divid_diff_calc_iter(x, y, i)

