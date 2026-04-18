import matplotlib.pyplot as plt
from math import exp, sin, sqrt, pi
from quadratures import *


functions = [
    (lambda x: x ** 2, 0, 3, 9),
    (exp, 0, 1, 1.718281),
    (sin, 0, pi, 2),
    (sqrt, 0, 9, 18),
]

methods = [
    rectangles,
    trapezoids,
    simpson,
    three_eights,
]


def convergence_analysis():
    for f, a, b, true_val in functions:
        for method in methods:
            values = []
            for n in range(6, 121, 6):
                val = method(f, a, b, n)
                values.append(val)
            plt.plot(range(6, 121, 6), values, label=method.__name__)
        plt.xlabel("n")
        plt.ylabel("Значение")
        plt.legend()
        plt.grid()
        plt.show()


def error_analysis():
    for f, a, b, true_val in functions:
        for method in methods:
            errors = []
            for n in range(6, 121, 6):
                val = method(f, a, b, n)
                errors.append(abs(val - true_val))
            plt.plot(range(6, 121, 6), errors, label=method.__name__)
        plt.xlabel("n")
        plt.ylabel("Абсолютная погрешность")
        plt.legend()
        plt.grid()
        plt.show()

error_analysis()