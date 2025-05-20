import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

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

v0 = 0.5
r0 = 0.5
a0 = 0.05

def create_time_array(t_start, t_end, stepsize):
    n_steps = int(np.floor((t_end - t_start) / stepsize)) + 1
    return t_start + stepsize * np.arange(n_steps)

def compute_derivatives(_, y):
    v, r, a = y
    dv = -alpha * v**3 + beta * v**2 - r - a
    dr = gamma * v**2 - delta - r
    da = epsilon * (b * (v - v1) - c * a)
    return [dv, dr, da]


def solve_system():
    y0 = [v0, r0, a0]
    t_eval = create_time_array(t_start, t_end, stepsize)
    return solve_ivp(compute_derivatives, [t_start, t_end], y0, t_eval=t_eval)


def plot_3d_trajectory(solution):
    v, r, a = solution.y
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(v, r, a, color='royalblue', lw=1)
    ax.set_xlabel('$v$ (Membraanpotentiaal)')
    ax.set_ylabel('$r$ (Herstelvariabele)')
    ax.set_zlabel('$a$ (Adaptatie)')
    ax.set_title('3D Fasevlak van het Bursting Model')
    plt.tight_layout()
    plt.show()


def main():
    solution = solve_system()
    plot_3d_trajectory(solution)


if __name__ == "__main__":
    main()

