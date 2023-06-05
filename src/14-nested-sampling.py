import numpy as np

def f(x, y, sigma_x, sigma_y):
    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-x ** 2 / (2 * sigma_x ** 2) - y ** 2 / (2 * sigma_y ** 2))