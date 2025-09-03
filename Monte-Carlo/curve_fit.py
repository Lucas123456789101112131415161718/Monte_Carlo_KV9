import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

V = np.array([875, 850, 840, 830, 820, 810, 800, 790, 780, 770, 760, 750, 740, 730, 720, 710, 700, 690, 680, 670, 660, 650, 625, 600])
C1 = [0.011799972, 0.012222494, 0.012222494, 0.02838312, 0.01605188, 0.01902838, 0.01350058, 0.02753269, 0.03178484, 0.04666738, 0.07079473, 0.0900393, 0.18741363, 0.2452429, 0.3315616, 0.44594451, 0.5513979, 0.65982779, 0.74742213, 0.81418093, 0.87413628, 0.90390135, 0.97236101, 0.98766876]
C2 = [0.0036, 0.0056, 0.0068, 0.0076, 0.0072, 0.0168, 0.0184, 0.0436, 0.07, 0.106, 0.1808, 0.2672, 0.3336, 0.4456, 0.5392, 0.6532, 0.7412, 0.8052, 0.8504, 0.9032, 0.9316, 0.9488, 0.992, 0.9956]
C3 = [0.0088, 0.0056, 0.0084, 0.008, 0.01, 0.0192, 0.024, 0.044, 0.0784, 0.1232, 0.2288, 0.3112, 0.4104, 0.5112, 0.6256, 0.7148, 0.7848, 0.838, 0.8928, 0.9064, 0.948, 0.9616, 0.9872, 0.9948]
C4 = [0.02243011, 0.0135963, 0.0279579, 0.02115446, 0.04411608, 0.03178484, 0.02583183, 0.02923355, 0.04284044, 0.05772297, 0.05687254, 0.08408232, 0.13086, 0.20442224, 0.27798448, 0.37493356, 0.46933135, 0.56925694, 0.67343468, 0.74614649, 0.81418093, 0.85457638, 0.97534936, 0.97533751]



error_C1 = [0.00544252, 0.00545676, 0.00545676, 0.00596392, 0.00558273, 0.00567807, 0.00549919, 0.00593875, 0.00603631, 0.00647076, 0.00704823, 0.00746665, 0.00902396, 0.00964851, 0.01027249, 0.0106214, 0.01050062, 0.00991787, 0.00904899, 0.00807484, 0.006869, 0.00609664, 0.00338334, 0.00227654]
error_C2 = [0.00119784, 0.00149247, 0.00164363, 0.00173692, 0.00169094, 0.00257043, 0.00268786, 0.00408407, 0.00510294, 0.00615675, 0.00769705, 0.00884995, 0.00942997, 0.00994064, 0.00996922, 0.00951903, 0.00875951, 0.00791476, 0.00713358, 0.0059137, 0.00504862, 0.00440811, 0.0028, 0.00132373]
error_C3 = [0.00186789, 0.00149247, 0.00182532, 0.00178168, 0.00198997, 0.00274455, 0.00306098, 0.0041019, 0.005376, 0.00657333, 0.0084012, 0.00925969, 0.00983813, 0.00999749, 0.00967935, 0.00903019, 0.00821922, 0.00736902, 0.00618735, 0.00582543, 0.00444054, 0.0038432, 0.00224821, 0.00143847]
error_C4 = [0.00580948, 0.00607667, 0.0057692, 0.00576984, 0.00643051, 0.00608897, 0.00591333, 0.00601461, 0.00639646, 0.00677698, 0.00675617, 0.00737354, 0.00824429, 0.0092623, 0.00996256, 0.01050409, 0.01047013, 0.01067623, 0.00984436, 0.00909773, 0.00810424, 0.0073326, 0.00501977, 0.00321218]

def sigmoid_func(x, V0, k):

    return 1 / (1 + np.exp(k * (x - V0)))

p0_guess = [725, -0.1]
popt1, pcov1 = curve_fit(sigmoid_func, V, C1, p0=p0_guess, sigma=error_C1)
popt2, pcov2 = curve_fit(sigmoid_func, V, C2, p0=p0_guess, sigma=error_C2)
popt3, pcov3 = curve_fit(sigmoid_func, V, C3, p0=p0_guess, sigma=error_C3)
popt4, pcov4 = curve_fit(sigmoid_func, V, C4, p0=p0_guess, sigma=error_C4)

V0_opt1, k_opt1 = popt1
V0_opt2, k_opt2 = popt2
V0_opt3, k_opt3 = popt3
V0_opt4, k_opt4 = popt4
print(f'Optimale Parameter C1: V0={V0_opt1:.4f}, k={k_opt1:.4f}')
print(f'Optimale Parameter C2: V0={V0_opt2:.4f}, k={k_opt2:.4f}')
print(f'Optimale Parameter C3: V0={V0_opt3:.4f}, k={k_opt3:.4f}')
print(f'Optimale Parameter C4704.: V0={V0_opt4:.4f}, k={k_opt4:.4f}')

perr1 = np.sqrt(np.diag(pcov1))
perr2 = np.sqrt(np.diag(pcov2))
perr3 = np.sqrt(np.diag(pcov3))
perr4 = np.sqrt(np.diag(pcov4))
print(f'standart deviation of parameter C1: V0_err={perr1[0]:.4f}, k_err={perr1[1]:.4f}')
print(f'standart deviation of parameter C2: V0_err={perr2[0]:.4f}, k_err={perr2[1]:.4f}')
print(f'standart deviation of parameter C3: V0_err={perr3[0]:.4f}, k_err={perr3[1]:.4f}')
print(f'standart deviation of parameter C4: V0_err={perr4[0]:.4f}, k_err={perr4[1]:.4f}')




V_fit = np.linspace(min(V), max(V), 100)
C1_fit = sigmoid_func(V_fit, V0_opt1, k_opt1)
C2_fit = sigmoid_func(V_fit, V0_opt2, k_opt2)
C3_fit = sigmoid_func(V_fit, V0_opt3, k_opt3)
C4_fit = sigmoid_func(V_fit, V0_opt4, k_opt4)

plt.figure(figsize=(12, 8))
plt.errorbar(V, C1, yerr=error_C1, color="blue", fmt='o', label='Channel 1', capsize=3, alpha=0.7, markersize=4)
plt.errorbar(V, C2, yerr=error_C2, color="orange", fmt='o', label='Channel 2', capsize=3, alpha=0.7, markersize=4)
plt.errorbar(V, C3, yerr=error_C3, color="green", fmt='o', label='Channel 3', capsize=3, alpha=0.7, markersize=4)
plt.errorbar(V, C4, yerr=error_C4, color="red", fmt='o', label='Channel 4', capsize=3, alpha=0.7, markersize=4)
plt.plot(V_fit, C1_fit, color='blue')
plt.plot(V_fit, C2_fit, color="orange")
plt.plot(V_fit, C3_fit, color='green')
plt.plot(V_fit, C4_fit, color='red')

plt.title('Failure rates with fitted curves')
plt.xlabel('Voltage [V]')
plt.ylabel('rate of failure')
plt.legend()
plt.grid(True)
plt.show()