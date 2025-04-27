import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec


def simulate_heston_qe(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
        T=3, dt=1 / 252, n_paths=30_000, seed=42
):
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0] = S0
    v[0] = v0

    for t in range(n_steps):
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        Z_v = Z1
        Z_s = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2

        vt = v[t]
        m = theta + (vt - theta) * np.exp(-kappa * dt)
        s2 = (
                vt * xi ** 2 * np.exp(-kappa * dt) * (1 - np.exp(-kappa * dt)) / kappa
                + theta * xi ** 2 * (1 - np.exp(-kappa * dt)) ** 2 / (2 * kappa)
        )
        psi = s2 / m ** 2

        v_next = np.zeros(n_paths)
        mask1 = psi <= 1.5
        mask2 = ~mask1

        # Cas 1 : psi <= 1.5
        if np.any(mask1):
            b2 = 2 / psi[mask1] - 1 + np.sqrt(2 / psi[mask1] * (2 / psi[mask1] - 1))
            a = m[mask1] / (1 + b2)
            v_next[mask1] = a * (np.sqrt(b2) + Z_v[mask1]) ** 2

        # Cas 2 : psi > 1.5
        if np.any(mask2):
            p = (psi[mask2] - 1) / (psi[mask2] + 1)
            beta = (1 - p) / m[mask2]
            u = np.random.rand(mask2.sum())
            v_temp = np.zeros_like(u)
            u_gt_p = u > p
            v_temp[u_gt_p] = -np.log((1 - u[u_gt_p]) / (1 - p[u_gt_p])) / beta[u_gt_p]
            v_next[mask2] = v_temp

        v[t + 1] = v_next
        S[t + 1] = S[t] * np.exp((r - 0.5 * vt) * dt + np.sqrt(vt * dt) * Z_s)

    # Plot
    time_grid = np.linspace(0, T, n_steps + 1)
    n_plot_paths = 10  # Only a few paths for clarity

    pastel_colors = [
        "#1f77b4",  # Bleu franc
        "#ff7f0e",  # Orange vif
        "#2ca02c",  # Vert franc
        "#d62728",  # Rouge vif
        "#9467bd",  # Violet saturé
        "#8c564b",  # Marron intense
        "#e377c2",  # Rose punchy
        "#7f7f7f",  # Gris foncé neutre
        "#bcbd22",  # Vert olive saturé
        "#17becf",  # Bleu turquoise éclatant
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[2.5, 1.5])

    # --- Simulated Paths of S_t ---
    for i in range(n_plot_paths):
        axs[0, 0].plot(time_grid, S[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[0, 0].set_title('Simulated Paths of $S_t$', fontsize=14)
    axs[0, 0].set_ylabel('Price $S_t$', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Distribution of log-returns ---
    log_returns = np.log(S[1:] / S[:-1])
    flat_log_returns = log_returns.flatten()
    axs[0, 1].hist(flat_log_returns, bins=300, color='black', edgecolor='black')
    axs[0, 1].set_title('Distribution of log-returns $\\log(S_{t+1}/S_t)$', fontsize=14)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].set_xlim(-0.05, 0.05)
    axs[0, 1].set_ylabel('Frequency', fontsize=12)

    # --- Simulated Paths of v_t ---
    for i in range(n_plot_paths):
        axs[1, 0].plot(time_grid, v[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[1, 0].set_title('Simulated Paths of $v_t$', fontsize=14)
    axs[1, 0].set_ylabel('Variance $v_t$', fontsize=12)
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Distribution of v_T ---
    v_differences = (v[1:] - v[:-1]).flatten()
    axs[1, 1].hist(v_differences, bins=300, color='black', edgecolor='black')
    axs[1, 1].set_title('Distribution of $v_{t+1} - v_t$', fontsize=14)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].set_xlim(-0.02, 0.02)
    axs[1, 1].set_ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.savefig('NEWPlotFIXED_PARAMS.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Compute statistics
    from scipy.stats import skew, kurtosis

    # Log-returns stats
    mean_log_returns = np.mean(flat_log_returns)
    std_log_returns = np.std(flat_log_returns)
    skew_log_returns = skew(flat_log_returns)
    kurt_log_returns = kurtosis(flat_log_returns)

    # v differences stats
    mean_v_diff = np.mean(v_differences)
    std_v_diff = np.std(v_differences)
    skew_v_diff = skew(v_differences)
    kurt_v_diff = kurtosis(v_differences)

    # Display nicely
    print("\n--- Statistics ---")
    print("Log-returns:")
    print(f"  Mean: {mean_log_returns:.5f}")
    print(f"  Std Dev: {std_log_returns:.5f}")
    print(f"  Skewness: {skew_log_returns:.5f}")
    print(f"  Kurtosis: {kurt_log_returns:.5f}")

    print("\nv(t+1) - v(t):")
    print(f"  Mean: {mean_v_diff:.5f}")
    print(f"  Std Dev: {std_v_diff:.5f}")
    print(f"  Skewness: {skew_v_diff:.5f}")
    print(f"  Kurtosis: {kurt_v_diff:.5f}")


    return S, v

def simulate_heston_qe_with_stochastic_params(
    S0=100.0, v0=0.04, r=0.0,
    kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
    T=3, dt=1 / 252, n_paths=10_000, seed=42,
    shock_std={"kappa": 0.05, "theta": 0.002, "xi": 0.002, "rho": 0.002},
    reversion_speed=0.95, t_time=50, nb_of_plots=5
):
    import numpy as np
    import matplotlib.pyplot as plt

    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0] = S0
    v[0] = v0

    # Paramètres dynamiques initiaux pour chaque chemin
    kappa_t = np.full(n_paths, kappa)
    theta_t = np.full(n_paths, theta)
    xi_t = np.full(n_paths, xi)
    rho_t = np.full(n_paths, rho)

    kappa_path = []
    theta_path = []
    xi_path = []
    rho_path = []

    for t in range(n_steps):

        # Chocs stochastiques indépendants chemin par chemin (OU style)
        if t > 0:
            kappa_t = kappa + reversion_speed * (kappa_t - kappa) + np.random.normal(0, shock_std["kappa"], n_paths)
            theta_t = theta + reversion_speed * (theta_t - theta) + np.random.normal(0, shock_std["theta"], n_paths)
            xi_t = xi + reversion_speed * (xi_t - xi) + np.random.normal(0, shock_std["xi"], n_paths)
            rho_t = rho + reversion_speed * (rho_t - rho) + np.random.normal(0, shock_std["rho"], n_paths)

        kappa_path.append(kappa_t.copy())
        theta_path.append(theta_t.copy())
        xi_path.append(xi_t.copy())
        rho_path.append(rho_t.copy())

        # Brownian motions indépendants par chemin
        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        Z_v = Z1
        Z_s = rho_t * Z1 + np.sqrt(1 - rho_t**2) * Z2

        vt = v[t]

        m = theta_t + (vt - theta_t) * np.exp(-kappa_t * dt)
        s2 = (
            vt * xi_t**2 * np.exp(-kappa_t * dt) * (1 - np.exp(-kappa_t * dt)) / kappa_t
            + theta_t * xi_t**2 * (1 - np.exp(-kappa_t * dt))**2 / (2 * kappa_t)
        )
        psi = s2 / m**2

        v_next = np.zeros_like(vt)
        mask1 = psi <= 1.5
        mask2 = ~mask1

        if np.any(mask1):
            b2 = 2 / psi[mask1] - 1 + np.sqrt(2 / psi[mask1] * (2 / psi[mask1] - 1))
            a = m[mask1] / (1 + b2)
            v_next[mask1] = a * (np.sqrt(b2) + Z_v[mask1])**2

        if np.any(mask2):
            p = (psi[mask2] - 1) / (psi[mask2] + 1)
            beta = (1 - p) / m[mask2]
            u = np.random.rand(mask2.sum())
            v_temp = np.zeros_like(u)
            u_gt_p = u > p
            v_temp[u_gt_p] = -np.log((1 - u[u_gt_p]) / (1 - p[u_gt_p])) / beta[u_gt_p]
            v_next[mask2] = v_temp

        v[t + 1] = np.maximum(v_next, 0)
        S[t + 1] = S[t] * np.exp((r - 0.5 * vt) * dt + np.sqrt(vt * dt) * Z_s)

    # Pour le plotting
    # Pastel colors palette
    pastel_colors = [
        "#1f77b4",  # Bleu franc
        "#ff7f0e",  # Orange vif
        "#2ca02c",  # Vert franc
        "#d62728",  # Rouge vif
        "#9467bd",  # Violet saturé
        "#8c564b",  # Marron intense
        "#e377c2",  # Rose punchy
        "#7f7f7f",  # Gris foncé neutre
        "#bcbd22",  # Vert olive saturé
        "#17becf",  # Bleu turquoise éclatant
    ]

    time_grid = np.linspace(0, T, n_steps + 1)
    n_plot_paths = nb_of_plots # Only a few paths for clarity

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[2.5, 1.5])

    # --- Simulated Paths of S_t ---
    for i in range(n_plot_paths):
        axs[0, 0].plot(time_grid, S[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[0, 0].set_title('Simulated Paths of $S_t$', fontsize=14)
    axs[0, 0].set_ylabel('Price $S_t$', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Distribution of log-returns ---
    log_returns = np.log(S[1:] / S[:-1])
    flat_log_returns = log_returns.flatten()
    print("len de log_returns", len(flat_log_returns))
    axs[0, 1].hist(flat_log_returns, bins=300, color='black', edgecolor='black')
    axs[0, 1].set_title('Distribution of log-returns $\\log(S_{t+1}/S_t)$', fontsize=14)
    axs[0, 1].set_ylabel('Frequency', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].set_xlim(-0.05, 0.05)


    # --- Simulated Paths of v_t ---
    for i in range(n_plot_paths):
        axs[1, 0].plot(time_grid, v[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[1, 0].set_title('Simulated Paths of $v_t$', fontsize=14)
    axs[1, 0].set_ylabel('Variance $v_t$', fontsize=12)
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Distribution of v_T ---
    v_differences = (v[1:] - v[:-1]).flatten()
    axs[1, 1].hist(v_differences, bins=300, color='black', edgecolor='black')
    axs[1, 1].set_title('Distribution of $v_{t+1} - v_t$', fontsize=14)
    axs[0, 1].set_ylabel('Frequency', fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].set_xlim(-0.02, 0.02)

    plt.tight_layout()
    plt.savefig("SandVPATHNEWSHOCKED", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Display some statistics ---

    print("=== Statistics on log-returns ===")
    print(f"Mean: {flat_log_returns.mean():.6f}")
    print(f"Std: {flat_log_returns.std():.6f}")
    print(f"Skewness: {scipy.stats.skew(flat_log_returns):.6f}")
    print(f"Kurtosis: {scipy.stats.kurtosis(flat_log_returns):.6f}")
    print()

    print("=== Statistics on v_{t+1} - v_t ===")
    print(f"Mean: {v_differences.mean():.6f}")
    print(f"Std: {v_differences.std():.6f}")
    print(f"Skewness: {scipy.stats.skew(v_differences):.6f}")
    print(f"Kurtosis: {scipy.stats.kurtosis(v_differences):.6f}")

    # --- New figure for stochastic parameters ---
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    kappa_path = np.array(kappa_path)  # (n_steps, n_paths)
    theta_path = np.array(theta_path)
    xi_path = np.array(xi_path)
    rho_path = np.array(rho_path)

    print(S)
    # --- Plot Kappa ---
    for i in range(n_plot_paths):
        axs[0, 0].plot(time_grid[:-1], kappa_path[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[0, 0].set_title('Paths of $\\kappa_t$', fontsize=14)
    axs[0, 0].set_ylabel('$\\kappa_t$', fontsize=12)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot Theta ---
    for i in range(n_plot_paths):
        axs[0, 1].plot(time_grid[:-1], theta_path[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[0, 1].set_title('Paths of $\\theta_t$', fontsize=14)
    axs[0, 1].set_ylabel('$\\theta_t$', fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # --- Plot Xi ---
    for i in range(n_plot_paths):
        axs[1, 0].plot(time_grid[:-1], xi_path[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[1, 0].set_title('Paths of $\\sigma_t$', fontsize=14)
    axs[1, 0].set_ylabel('$\\sigma_t$', fontsize=12)
    axs[1, 0].set_xlabel('Time (years)', fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # --- Plot Rho ---
    for i in range(n_plot_paths):
        axs[1, 1].plot(time_grid[:-1], rho_path[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[1, 1].set_title('Paths of $\\rho_t$', fontsize=14)
    axs[1, 1].set_ylabel('$\\rho_t$', fontsize=12)
    axs[1, 1].set_xlabel('Time (years)', fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("NEWParamsPaths.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Prints pour debug sur les chemins (exemple sur le chemin 0)
    print(f"kappa[0][{t_time}] = {kappa_path[t_time][0]:.6f}, kappa[0][{t_time+1}] = {kappa_path[t_time+1][0]:.6f}")
    print(f"theta[0][{t_time}] = {theta_path[t_time][0]:.6f}, theta[0][{t_time+1}] = {theta_path[t_time+1][0]:.6f}")
    print(f"xi   [0][{t_time}] = {xi_path[t_time][0]:.6f}, xi   [0][{t_time+1}] = {xi_path[t_time+1][0]:.6f}")
    print(f"rho  [0][{t_time}] = {rho_path[t_time][0]:.6f}, rho  [0][{t_time+1}] = {rho_path[t_time+1][0]:.6f}")

    print(kappa_path)

    return S, v, (
        kappa_path[t_time], kappa_path[t_time+1],
        theta_path[t_time], theta_path[t_time+1],
        xi_path[t_time], xi_path[t_time+1],
        rho_path[t_time], rho_path[t_time+1]
    )







def simulate_heston_qe_with_stochastic_params_2(
    S0=100.0, v0=0.04, r=0.0,
    kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
    T=3, dt=1 / 252, n_paths=30_000, seed=42,
    shock_std={"kappa": 0.05, "theta": 0.002, "xi": 0.002, "rho": 0.002},
    reversion_speed=0.95, t_time=100
):
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(T / dt)
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))

    S[0] = S0
    v[0] = v0

    # Paramètres dynamiques initialisés à leur valeur cible
    kappa_t = kappa
    theta_t = theta
    xi_t = xi
    rho_t = rho

    kappa_path = []
    theta_path = []
    xi_path = []
    rho_path = []

    for t in range(n_steps):

        if t>0:
        # Chocs stochastiques sur les paramètres (Ornstein-Uhlenbeck style)
            kappa_t = kappa + reversion_speed * (kappa_t - kappa) + np.random.normal(0, shock_std["kappa"])
            theta_t = theta + reversion_speed * (theta_t - theta) + np.random.normal(0, shock_std["theta"])
            xi_t    = xi    + reversion_speed * (xi_t - xi)       + np.random.normal(0, shock_std["xi"])
            rho_t   = rho   + reversion_speed * (rho_t - rho)     + np.random.normal(0, shock_std["rho"])

        kappa_path.append(kappa_t)
        theta_path.append(theta_t)
        xi_path.append(xi_t)
        rho_path.append(rho_t)

        Z1 = np.random.randn(n_paths)
        Z2 = np.random.randn(n_paths)
        Z_v = Z1
        Z_s = rho_t * Z1 + np.sqrt(1 - rho_t ** 2) * Z2

        vt = v[t]
        m = theta_t + (vt - theta_t) * np.exp(-kappa_t * dt)
        s2 = (
            vt * xi_t ** 2 * np.exp(-kappa_t * dt) * (1 - np.exp(-kappa_t * dt)) / kappa_t
            + theta_t * xi_t ** 2 * (1 - np.exp(-kappa_t * dt)) ** 2 / (2 * kappa_t)
        )
        psi = s2 / m ** 2

        v_next = np.zeros(n_paths)
        mask1 = psi <= 1.5
        mask2 = ~mask1

        if np.any(mask1):
            b2 = 2 / psi[mask1] - 1 + np.sqrt(2 / psi[mask1] * (2 / psi[mask1] - 1))
            a = m[mask1] / (1 + b2)
            v_next[mask1] = a * (np.sqrt(b2) + Z_v[mask1]) ** 2

        if np.any(mask2):
            p = (psi[mask2] - 1) / (psi[mask2] + 1)
            beta = (1 - p) / m[mask2]
            u = np.random.rand(mask2.sum())
            v_temp = np.zeros_like(u)
            u_gt_p = u > p
            v_temp[u_gt_p] = -np.log((1 - u[u_gt_p]) / (1 - p[u_gt_p])) / beta[u_gt_p]
            v_next[mask2] = v_temp

        v[t + 1] = v_next
        S[t + 1] = S[t] * np.exp((r - 0.5 * vt) * dt + np.sqrt(vt * dt) * Z_s)

    plt.figure(figsize=(14, 8))

    # kappa
    plt.subplot(2, 2, 1)
    plt.plot(kappa_path, color='black')
    plt.title("Evolution of $\\kappa$", fontsize=12)
    plt.xlabel("Time (days)")
    plt.ylabel("$\\kappa$ value")
    plt.grid(True)

    # theta
    plt.subplot(2, 2, 2)
    plt.plot(theta_path, color='black')
    plt.title("Evolution of $\\theta$", fontsize=12)
    plt.xlabel("Time (days)")
    plt.ylabel("$\\theta$ value")
    plt.grid(True)

    # xi (vol of vol)
    plt.subplot(2, 2, 3)
    plt.plot(xi_path, color='black')
    plt.title("Evolution of $\\sigma$", fontsize=12)
    plt.xlabel("Time (days)")
    plt.ylabel("$\\sigma$ value")
    plt.grid(True)

    # rho
    plt.subplot(2, 2, 4)
    plt.plot(rho_path, color='black')
    plt.title("Evolution of $\\rho$", fontsize=12)
    plt.xlabel("Time (days)")
    plt.ylabel("$\\rho$ value")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("stochastic_params_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"kappa[100] = {kappa_path[0]:.6f}, kappa[101] = {kappa_path[101]:.6f}")
    print(f"theta[100] = {theta_path[0]:.6f}, theta[101] = {theta_path[101]:.6f}")
    print(f"xi   [100] = {xi_path[0]:.6f}, xi   [101] = {xi_path[101]:.6f}")
    print(f"rho  [100] = {rho_path[0]:.6f}, rho  [101] = {rho_path[101]:.6f}")

    return S, v, kappa_path, theta_path, xi_path, rho_path












def plot_terminal_distributions(S, v):
    """
    Affiche les distributions de S_T et v_T à partir des trajectoires simulées.

    Paramètres
    ----------
    S : ndarray
        Matrice (n_steps+1, n_paths) contenant les trajectoires de prix de l'actif.
    v : ndarray
        Matrice (n_steps+1, n_paths) contenant les trajectoires de la variance instantanée.
    """

    S_T = S[-1]
    v_T = v[-1]

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme de S_T
    axs[0].hist(S_T, bins=100, density=False, color='black', alpha=0.7)
    # axs[0].axvline(np.mean(S_T), color='gray', linestyle='dashed', label='Mean')
    axs[0].set_title("Distribution of $S_T$")
    axs[0].set_xlabel("$S_T$")
    axs[0].set_ylabel("Nb of observations")
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].legend()

    # Histogramme de v_T
    axs[1].hist(v_T, bins=100, density=False, color='black', alpha=0.7)
    # axs[1].axvline(np.mean(v_T), color='gray', linestyle='dashed', label='Mean')
    axs[1].set_title("Distribution of $v_T$ (Terminal Variance)")
    axs[1].set_xlabel("$v_T$")
    axs[1].set_ylabel("Nb of observations")
    axs[1].grid(True, linestyle='--', alpha=0.3)
    axs[1].legend()

    plt.tight_layout()

    plt.show()


def extract_snapshots(S, v, t):
    """
    Extrait les valeurs S_t, S_{t+1}, v_t, v_{t+1} pour chaque trajectoire simulée.

    Paramètres
    ----------
    S : ndarray
        Matrice (n_steps+1, n_paths) des prix simulés.
    v : ndarray
        Matrice (n_steps+1, n_paths) des volatilités simulées.
    t : int
        Indice temporel (0 <= t < n_steps)

    Retours
    -------
    S_t : ndarray
        Valeurs de S_t pour chaque chemin.
    S_tp1 : ndarray
        Valeurs de S_{t+1} pour chaque chemin.
    v_t : ndarray
        Valeurs de v_t pour chaque chemin.
    v_tp1 : ndarray
        Valeurs de v_{t+1} pour chaque chemin.
    """
    if t < 0 or t + 1 >= S.shape[0]:
        raise ValueError(f"t doit être dans l'intervalle [0, {S.shape[0] - 2}]")

    S_t = S[t]
    S_tp1 = S[t + 1]
    v_t = v[t]
    v_tp1 = v[t + 1]

    return S_t, S_tp1, v_t, v_tp1
