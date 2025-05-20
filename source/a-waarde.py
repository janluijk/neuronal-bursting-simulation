import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

t_start = 0.0
t_end = 2500.0
stepsize = 0.1

alpha = 1.0
beta = 3.0
gamma = 5.0
delta = 1.0

epsilon = 0.001
b = 0.20
c = 1.00
v1 = -1.61

initial_conditions = [(0.5, 2.0)]
a_values = [0.077, 0.078, 0.079, 0.080, 0.081]

def create_time_array(t_start, t_end, stepsize):
    n_steps = int(np.floor((t_end - t_start) / stepsize)) + 1
    return t_start + stepsize * np.arange(n_steps)

def fast_system(t, y, a):
    v, r = y
    dvdt = -alpha * v**3 + beta * v**2 - r - a
    drdt = gamma * v**2 - delta - r
    return [dvdt, drdt]

def plot_solutions(a_values, initial_conditions):
    fig, axes = plt.subplots(len(a_values), 2, figsize=(12, 3 * len(a_values)))

    for i, a in enumerate(a_values):
        ax_time, ax_phase = axes[i]

        t_eval = create_time_array(t_start, t_end, stepsize)
        t_span = (t_start, t_end)

        sol = solve_ivp(fast_system, t_span, initial_conditions[0], args=(a,), t_eval=t_eval)
        v, r = sol.y

        ax_time.plot(sol.t, v, label='$v(t)$')
        ax_time.plot(sol.t, r, label='$r(t)$')
        ax_time.set_title(f'Tijdreeksen voor $a = {a}$')
        ax_time.set_xlabel('$t$')
        ax_time.set_ylabel('$v$, $r$')
        ax_time.legend()

        ax_phase.plot(v, r, label=f'Traject voor $a = {a}$')
        ax_phase.set_title(f'Fasevlak voor $a = {a}$')
        ax_phase.set_xlabel('$v$')
        ax_phase.set_ylabel('$r$')
        ax_phase.legend()

    plt.tight_layout()
    plt.show()


def main():
    plot_solutions(a_values, initial_conditions)

if __name__ == "__main__":
    main()
