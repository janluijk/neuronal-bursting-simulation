import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from matplotlib.cm import gnuplot
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

alpha = 1.0
beta = 3.0
gamma = 5.0
delta = 1.0

t_start = 0.0
t_end = 600.0
stepsize = 0.1

initial_conditions = [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0), (0.5, -1.5)]
equilibrium_guesses = [[0, 0], [2, 1], [-1.5, 14]]

v_range = np.linspace(-3.0, 3.0, 25)
r_range = np.linspace(-1.0, 15.0, 25)
V, R = np.meshgrid(v_range, r_range)

def create_time_array(t_start, t_end, stepsize):
    n_steps = int(np.floor((t_end - t_start) / stepsize)) + 1
    return t_start + stepsize * np.arange(n_steps)

def compute_dv(v, r):
    return -alpha * v**3 + beta * v**2 - r

def compute_dr(v, r):
    return gamma * v**2 - delta - r

def compute_vector_field(V, R):
    dV = compute_dv(V, R)
    dR = compute_dr(V, R)
    speed = np.sqrt(dV**2 + dR**2)
    norm = Normalize(vmin=np.min(speed), vmax=np.max(speed))
    colors = gnuplot(norm(speed))
    dVn = 0.5 * dV / speed
    dRn = 0.5 * dR / speed
    return dVn, dRn, colors, norm

def draw_vector_field(ax, V, R, dVn, dRn, colors):
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            ax.quiver(V[i, j], R[i, j], dVn[i, j], dRn[i, j], color=colors[i, j], scale=35, headwidth=5, headlength=10)

def draw_nullclines(ax):
    v_null = np.linspace(-2, 4.5, 500)
    r_null_1 = -alpha * v_null**3 + beta * v_null**2
    r_null_2 = gamma * v_null**2 - delta
    ax.plot(v_null, r_null_1, ',k--', lw=1, label='$\\dot{v} = 0$')
    ax.plot(v_null, r_null_2, ',k--', lw=1, label='$\\dot{r} = 0$')

def draw_trajectories(ax, t_eval, initial_conditions):
    for y0 in initial_conditions:
        sol = solve_ivp(lambda t, y: [compute_dv(y[0], y[1]), compute_dr(y[0], y[1])], (t_start, t_end), y0, t_eval=t_eval)
        ax.plot(sol.y[0], sol.y[1], lw=2, label=f'Uit {y0}')

def find_equilibrium_points(guesses):
    def nullcline_equations(p):
        v, r = p
        return [compute_dv(v, r), compute_dr(v, r)]
    return [fsolve(nullcline_equations, guess) for guess in guesses]

def mark_equilibrium_points(ax, equilibria):
    for eq in equilibria:
        ax.plot(eq[0], eq[1], 'ko', markersize=5)

def add_labels_and_legend(ax):
    ax.set_xlabel('$v$ (membraanpotentiaal)', fontsize=14)
    ax.set_ylabel('$r$ (herstelvariabele)', fontsize=14)
    ax.set_title('2D-fasevlak', fontsize=16)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-1.0, 15.0)
    ax.grid(True)
    ax.legend(loc='upper right')

def add_colorbar(fig, ax, norm):
    sm = ScalarMappable(cmap=gnuplot, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Snelheid')

def plot_phase_plane():
    dVn, dRn, colors, norm = compute_vector_field(V, R)
    t_eval = create_time_array(t_start, t_end, stepsize)
    equilibria = find_equilibrium_points(equilibrium_guesses)

    fig, ax = plt.subplots(figsize=(10, 8))

    draw_vector_field(ax, V, R, dVn, dRn, colors)
    draw_nullclines(ax)
    draw_trajectories(ax, t_eval, initial_conditions)

    mark_equilibrium_points(ax, equilibria)

    add_labels_and_legend(ax)
    add_colorbar(fig, ax, norm)

    plt.show()

def main():
    plot_phase_plane()


if __name__ == "__main__":
    main()

