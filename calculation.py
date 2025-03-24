import numpy as np
import math
import scipy.signal as signal

def chebyshev_coefficients(n, ripple): #порядок фильтра, уровень пульсаций (0,1 / 0,5 / 1 / 2) дБ
    z, p, k = signal.cheb1ap(n, ripple)
    b, a = signal.zpk2tf(z, p, k)
    g = [a[i] / b[i] for i in range(n)]
    return g


def calculate_chebyshev_filter(f_c, R, n, ripple=0.5, first_element_C=True):
    g = chebyshev_coefficients(n, ripple)
    C = []
    L = []
    for i, g_k in enumerate(g):
        if (first_element_C and i % 2 == 0) or (not(first_element_C) and i % 2 != 0):
            C_k = g_k / (2 * np.pi * f_c * R)
            C.append(C_k)
        else:
            L_k = (R * g_k) / (2 * np.pi * f_c)
            L.append(L_k)
    return C, L


def calculate_rc_low_pass_filter(f_c, Z, n):
    rc_pairs = []
    
    for i in range(n):
        alpha = math.sqrt(2 ** (1 / (i+1)) - 1)
        fc_i = f_c * alpha
        R = Z
        C = 1 / (2 * math.pi * fc_i * R)
        rc_pairs.append((R, C))
    return rc_pairs


def calculate_butterworth_filter(f_c, R, n, first_element_C=True):
    g = [2 * np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)]
    C = []
    L = []
    for i, g_k in enumerate(g):
        if (first_element_C and i % 2 == 0) or (not(first_element_C) and i % 2 != 0):
            C_k = g_k / (2 * np.pi * f_c * R)
            C.append(C_k)
        else:
            L_k = (R * g_k) / (2 * np.pi * f_c)
            L.append(L_k)

    return C, L