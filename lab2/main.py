import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

from lab1.quadratures import simpson
from quadratures import calculate_integral, get_chebyshev, get_gauss, get_radau, get_lobatto

functions = [
    ('x^2', lambda x: x ** 2, 0, 3),
    ('e^x', np.exp, 0, 1),
    ('sin(x)', np.sin, 0, np.pi),
    ('\\sqrt(x)', np.sqrt, 0, 9),
]

methods = {
    'Чебышев': get_chebyshev,
    'Гаусс': get_gauss,
    'Радо': get_radau,
    'Лобатто': get_lobatto,
    'Симпсон': simpson,
}


for function_name, f, a, b in functions:
    n_values = np.arange(2, 6)
    errors = {name: [] for name in methods.keys()}

    for name, method in methods.items():
        exact_val, _ = quad(f, a, b, epsabs=1e-15)
        for n in n_values:
            if name != 'Симпсон':
                approx_val = calculate_integral(f, *method(n), a, b)
            else:
                approx_val = method(f, a, b, n)
            err = abs(exact_val - approx_val)

            errors[name].append(max(err, 1e-16))

    plt.figure(figsize=(10, 6))
    markers = ['o', 's', '^', 'D', '.']
    for (name, err), marker in zip(errors.items(), markers):
        plt.plot(n_values, err, label=name, marker=marker, markersize=5, linestyle='-')

    plt.yscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.xlabel('Количество узлов (n)', fontsize=12)
    plt.ylabel('Фактическая погрешность |I - I_h|', fontsize=12)
    plt.title(f'Зависимость погрешности от объема вычислений {function_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'error_{function_name}_plot.png', dpi=300)
    plt.show()

    epsilons = np.logspace(-2, -15, 30)
    required_intervals = {name: [] for name in methods.keys()}
    
    for eps in epsilons:
        for name in methods.keys():
            err_array = np.array(errors[name])
            valid_indices = np.where(err_array < eps)[0]
            if len(valid_indices) > 0:
                required_intervals[name].append(n_values[valid_indices[0]])
            else:
                required_intervals[name].append(None)
    
    plt.figure(figsize=(10, 6))
    for (name, req_inv), marker in zip(required_intervals.items(), markers):
        eps_filtered = [e for e, r in zip(epsilons, req_inv) if r is not None]
        req_filtered = [r for r in req_inv if r is not None]
    
        plt.plot(eps_filtered, req_filtered, label=name, marker=marker, markersize=5, linestyle='-')
    
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.xlabel('Нужный эпсилон (заданная точность)', fontsize=12)
    plt.ylabel('Нужное количество узлов (n)', fontsize=12)
    plt.title(f'Влияние требуемой точности на объем вычислений {function_name}', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'epsilon_{function_name}_plot.png', dpi=300)
    plt.show()
