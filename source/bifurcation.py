import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

alpha = 1.0
beta = 3.0
gamma = 5.0
delta = 1.0

t_start = 0.0
t_end = 1000.0
stepsize = 0.1

def create_time_array(t_start, t_end, stepsize):
    n_steps = int(np.floor((t_end - t_start) / stepsize)) + 1
    return t_start + stepsize * np.arange(n_steps)

def simulate_bifurcation(a_values, v0=0.5, r0=0.5, t_transient=100, t_end=200, stepsize=0.005):
    v_mins = []
    v_maxs = []
    a_record = []

    for a_val in a_values:
        def dvdt(_, y):
            v, r = y
            dv = -alpha * v**3 + beta * v**2 - r - a_val
            dr = gamma * v**2 - delta - r
            return [dv, dr]

        t_eval = create_time_array(0, t_end, stepsize)
        sol = solve_ivp(dvdt, (0, t_end), [v0, r0], t_eval=t_eval, method='RK45')

        v = sol.y[0]
        r = sol.y[1]
        t = sol.t

        transient_mask = t >= t_transient
        v_ss = v[transient_mask]
        r_ss = r[transient_mask]

        if np.allclose(v_ss[-1], -1.61, atol=0.05) and np.allclose(r_ss[-1], 14.0, atol=0.2):
            v_mins.append(v_ss[-1])
            v_maxs.append(v_ss[-1])
        else:
            v_mins.append(np.min(v_ss))
            v_maxs.append(np.max(v_ss))

        a_record.append(a_val)

    return np.array(a_record), np.array(v_mins), np.array(v_maxs)

def plot_bifurcation_diagram(a_values, v_mins, v_maxs):
    plt.figure(figsize=(10, 6))
    plt.plot(a_values, v_mins, 'k.', markersize=2, label='Min $v(t)$')
    plt.plot(a_values, v_maxs, 'r.', markersize=2, label='Max $v(t)$')
    plt.xlabel('$a$ (adaptatieparameter)', fontsize=14)
    plt.ylabel('Asymptotisch gedrag van $v$', fontsize=14)
    plt.title('Bifurcatiediagram', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    a_values = np.linspace(0.078, 0.082, 300)
    a_vals, v_mins, v_maxs = simulate_bifurcation(a_values)
    plot_bifurcation_diagram(a_vals, v_mins, v_maxs)

if __name__ == "__main__":
    main()

