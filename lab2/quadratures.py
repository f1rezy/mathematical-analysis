from typing import Callable, Tuple, Dict

import numpy as np
from scipy.special import roots_legendre, eval_legendre

MathFunction = Callable[[np.ndarray], np.ndarray]


def calculate_integral(
        func: MathFunction,
        nodes: np.ndarray,
        weights: np.ndarray,
        a: float,
        b: float
) -> float:
    rescaled_nodes: np.ndarray = 0.5 * (nodes + 1) * (b - a) + a
    rescaled_weights: np.ndarray = weights * 0.5 * (b - a)
    return float(np.sum(rescaled_weights * func(rescaled_nodes)))


def get_chebyshev(n: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes_map: Dict[int, np.ndarray] = {
        2: np.array([-0.577350, 0.577350]),
        3: np.array([-0.707107, 0, 0.707107]),
        4: np.array([-0.794654, -0.187592, 0.187592, 0.794654]),
        5: np.array([-0.832497, -0.374541, 0, 0.374541, 0.832497]),
    }

    if n not in nodes_map:
        raise ValueError(f"Узлы Чебышева для n={n} не определены.")

    nodes: np.ndarray = nodes_map[n]
    weights: np.ndarray = np.ones(n) * (2 / n)
    return nodes, weights


def get_gauss(n: int) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = roots_legendre(n)
    return nodes, weights


def get_radau(n: int) -> Tuple[np.ndarray, np.ndarray]:
    radau_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        2: (np.array([-1, 1 / 3]),
            np.array([0.5, 1.5])),
        3: (np.array([-1, (1 - np.sqrt(6)) / 5, (1 + np.sqrt(6)) / 5]),
            np.array([2 / 9, (16 + np.sqrt(6)) / 18, (16 - np.sqrt(6)) / 18])),
        4: (np.array([-1, -0.575318923, 0.181066271, 0.822824081]),
            np.array([0.125, 0.657688640, 0.776386938, 0.440924422])),
        5: (np.array([-1, -0.720480271, -0.167180865, 0.446313980, 0.885791601]),
            np.array([0.08, 0.446207802, 0.623653046, 0.562712030, 0.287427121])),
    }

    if n not in radau_map:
        raise ValueError(f"Формула Радо для n={n} не определена в справочнике.")

    nodes, weights = radau_map[n]
    return nodes, weights


def get_lobatto(n: int) -> Tuple[np.ndarray, np.ndarray]:
    lobatto_map: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        2: (np.array([-1, 1]),
            np.array([1, 1])),
        3: (np.array([-1, 0.0, 1]),
            np.array([1 / 3, 4 / 3, 1 / 3])),
        4: (np.array([-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1]),
            np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6])),
        5: (np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1]),
            np.array([0.1, 49 / 90, 32 / 45, 49 / 90, 0.1]))
    }

    if n not in lobatto_map:
        raise ValueError(f"Формула Лобатто для n={n} не определена в справочнике.")

    nodes, weights = lobatto_map[n]
    return nodes, weights
