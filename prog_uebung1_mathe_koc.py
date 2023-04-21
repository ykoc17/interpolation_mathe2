import numpy as np
import matplotlib.pyplot as plt
import time

#Funktionsdefinitionen
def naiv_inter(x, y, xi):

    A = np.empty([len(x), len(x)])

    for i in range(len(x)):
        for j in range(len(x)):
            A[i][j] = np.power(x[i], len(x)-(j+1))

    a = np.linalg.solve(A, y)
    poly_to_eval = np.poly1d(a)

    return poly_to_eval(xi)

def lagrange_inter(x, y, xi):
    yi = 0
    for k in range(len(x)):
        temp = 1
        for j in range(len(x)):
            if j != k:          #Dieser Term wird ausgelassen
                temp = temp * ((xi - x[j]) / (x[k] - x[j]))
        yi = yi + temp * y[k]
    return yi

def newton_inter(x, y, xi):
    div_diff_array = np.empty(len(x))
    for i in range(len(x)):
        div_diff_array[i] = divid_diff_calc(x, y, 0, i)

    div_diff_array2 = np.empty(len(x))
    for i in range(len(x)):
        div_diff_array2[i] = divid_diff_calc_iter(x, y, i)

    yi = np.ones(len(xi))*div_diff_array[0]
    for k in range(1, len(x)):
        temp = 1
        for i in range(k):
            temp *= (xi-x[i])
        yi += div_diff_array[k]*temp
    return yi

def divid_diff_calc(x, y, start, end):
    if start == end: return y[end]
    return (divid_diff_calc(x, y, start+1, end)-divid_diff_calc(x, y, start, end-1))/(x[end]-x[start])

def divid_diff_calc_iter(x, y, end):
    a = np.empty((end,end))

    for i in range(0, end+1):
        a[i,i] = y[i]

    for i in range(end):
        for j in range(i+1, end):
            a[i,j] = (a[i+1, j]-a[i, j-1])/(x[j]-x[i])
            a[i+1, j+1] = (a[i+2, j+1]-a[i+1, j])/(x[j]-x[i])
    return a[0,end]


###################################################
print("a) ")
x = np.array([0, 1, 2, 3])
y = np.array([-5, -6, -1, 16])
xi = np.array([0.5, 1.5, 2.5])

print("xi =", xi)
print("yi = p(xi) =", naiv_inter(x, y, xi), "NAIV")
print("yi = p(xi) =", lagrange_inter(x, y, xi), "LAGRANGE")
print("yi = p(xi) =", newton_inter(x, y, xi), "NEWTON")


###################################################
print("\nb) ")
x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
y = np.array([-5, -6, -1, 16, 10, 20, 9])
xi = np.array([3.5, 5])

print("xi =", xi)
print("yi = p(xi) =", naiv_inter(x, y, xi), "NAIV")
print("yi = p(xi) =", lagrange_inter(x, y, xi), "LAGRANGE")
print("yi = p(xi) =", newton_inter(x, y, xi), "NEWTON")


###################################################
print("\nc) ")
#Setup
n = 5          #Anzahl Stützstellen
start = 0       #Intervallbeginn
end = 2         #Intervallende

x = np.empty([n])
y = np.empty([n])

for i in range(n):
    x[i] = start + (((end-start)*i)/(n-1))
    y[i] = np.sin(x[i])


#Interpolation und Laufzeitmessung
resolution = 100
start_interp = 0
end_interp = 2
x_intp = np.linspace(start_interp, end_interp, resolution)

tic = time.perf_counter()
y_intp_naiv = naiv_inter(x, y, x_intp)
toc = time.perf_counter()
print("Laufzeit NAIV:", toc-tic, "s")

tic = time.perf_counter()
y_intp_lagrange = lagrange_inter(x, y, x_intp)
toc = time.perf_counter()
print("Laufzeit LAGRANGE:", toc-tic, "s")

tic = time.perf_counter()
y_intp_newton = newton_inter(x, y, x_intp)
toc = time.perf_counter()
print("Laufzeit NEWTON:", toc-tic, "s")


#Differenz zwischen Funktion und Interpolation berechnen
y_intp_naiv_low_resolution = naiv_inter(x, y, x)
diff_array_naiv = np.array(y_intp_naiv_low_resolution-y)

y_intp_lagrange_low_resolution = lagrange_inter(x, y, x)
diff_array_lagrange = np.array(y_intp_lagrange_low_resolution-y)

y_intp_newton_low_resolution = newton_inter(x, y, x)
diff_array_newton = np.array(y_intp_newton_low_resolution-y)


#Plotten
fig, (ax, axi1, axi2, axi3) = plt.subplots(4, 1)
fig.tight_layout()
ax.plot(x, y, color="C1")
ax.plot(x, y, "o", color="C7")
ax.set_title("f(x)=sin(x)", fontsize = 10)

axi1.plot(x_intp, y_intp_naiv, color="C0")
axi1.plot(x, y, "o", color="C7")
axi1.set_title("p1(x) Naive Interpolation", fontsize = 10)

axi2.plot(x_intp, y_intp_lagrange, color="C0")
axi2.plot(x, y, "o", color="C7")
axi2.set_title("p2(x) Lagrange Interpolation", fontsize = 10)

axi3.plot(x_intp, y_intp_newton, color="C0")
axi3.plot(x, y, "o", color="C7")
axi3.set_title("p3(x) Newton Interpolation", fontsize = 10)

fig2, (axi1diff, axi2diff, axi3diff) = plt.subplots(3, 1)
fig2.tight_layout()
axi1diff.plot(x, diff_array_naiv, color="C3")
axi1diff.set_title("En[f](x) Fehler Naive Interpolation", fontsize = 10)

axi2diff.plot(x, diff_array_lagrange, color="C3")
axi2diff.set_title("En[f](x) Fehler Lagrange Interpolation", fontsize = 10)

axi3diff.plot(x, diff_array_newton, color="C3")
axi3diff.set_title("En[f](x) Fehler Newton Interpolation", fontsize = 10)

plt.show()