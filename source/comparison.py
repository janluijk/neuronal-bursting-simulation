import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

t_start = 0.0
t_end = 1000.0
stepsize = 0.1

alpha = 1.0
beta = 3.0
gamma = 5.0
delta = 1.0

epsilon = 0.001
b = 0.20
c = 1.00
v1 = -1.61

def create_time_array(t_start, t_end, stepsize):
    n_steps = int(np.floor((t_end - t_start) / stepsize)) + 1
    return t_start + stepsize * np.arange(n_steps)

def model_2d(t, y):
    v, r = y
    dvdt = -alpha * v**3 + beta * v**2 - r
    drdt = gamma * v**2 - delta - r
    return [dvdt, drdt]

def model_3d(t, y):
    v, r, a = y
    dvdt = -alpha * v**3 + beta * v**2 - r - a
    drdt = gamma * v**2 - delta - r
    dadt = epsilon * (b * (v - v1) - c * a)
    return [dvdt, drdt, dadt]

def simulate_models():
    y0_2d = [0.1, 0.1]
    y0_3d = [0.1, 0.1, 0.0]

    t_eval = create_time_array(t_start, t_end, stepsize)
    t_span = (t_start, t_end)

    sol_2d = solve_ivp(model_2d, t_span, y0_2d, t_eval=t_eval)
    sol_3d = solve_ivp(model_3d, t_span, y0_3d, t_eval=t_eval)

    return sol_2d, sol_3d

def plot_comparison(sol_2d, sol_3d):
    plt.figure(figsize=(10, 5))
    plt.plot(sol_2d.t, sol_2d.y[0], 'k', label='2D model ($v$)', lw=2.0)
    plt.plot(sol_3d.t, sol_3d.y[0], 'r', label='3D model ($v$)', lw=2.0)
    plt.xlabel('Tijd $t$')
    plt.ylabel('Membraanpotentiaal $v$')
    plt.title('Vergelijking tussen 2D en 3D model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    sol_2d, sol_3d = simulate_models()
    plot_comparison(sol_2d, sol_3d)

if __name__ == "__main__":
    main()
