import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    n_plot_paths = 1000  # Only a few paths for clarity

    # Pastel colors palette
    pastel_colors = [
        "#F0F0F0",  # Gris ultra clair
        "#D9D9D9",  # Gris perle
        "#C0C0C0",  # Argent doux
        "#A9A9A9",  # Gris moyen
        "#999999",  # Gris plus dense
        "#8C8C8C",  # Graphite léger
        "#808080",  # Gris standard
        "#737373",  # Slate doux
        "#666666",  # Asphalte clair
        "#595959",  # Gris anthracite doux
    ]

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot S_t (prices)
    for i in range(n_plot_paths):
        axs[0].plot(time_grid, S[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[0].set_title('Simulated Paths of $S_t$', fontsize=14)
    axs[0].set_ylabel('Price $S_t$', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # Plot v_t (variance)
    for i in range(n_plot_paths):
        axs[1].plot(time_grid, v[:, i], color=pastel_colors[i % len(pastel_colors)], alpha=0.9)
    axs[1].set_title('Simulated Paths of $v_t$', fontsize=14)
    axs[1].set_ylabel('Variance $v_t$', fontsize=12)
    axs[1].set_xlabel('Time (years)', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    return S, v


def simulate_heston_qe_with_stochastic_params(
    S0=100.0, v0=0.04, r=0.0,
    kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
    T=3, dt=1 / 252, n_paths=30_000, seed=42,
    shock_std={"kappa": 0.005, "theta": 0.0002, "xi": 0.0002, "rho": 0.0002},
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

    plt.subplot(2, 2, 1)
    plt.plot(kappa_path)
    plt.title("Évolution de kappa")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(theta_path)
    plt.title("Évolution de theta")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(xi_path)
    plt.title("Évolution de xi (vol of vol)")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(rho_path)
    plt.title("Évolution de rho")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"kappa[100] = {kappa_path[0]:.6f}, kappa[101] = {kappa_path[101]:.6f}")
    print(f"theta[100] = {theta_path[0]:.6f}, theta[101] = {theta_path[101]:.6f}")
    print(f"xi   [100] = {xi_path[0]:.6f}, xi   [101] = {xi_path[101]:.6f}")
    print(f"rho  [100] = {rho_path[0]:.6f}, rho  [101] = {rho_path[101]:.6f}")

    return S, v, (
        kappa_path[t_time], kappa_path[t_time+1],
        theta_path[t_time], theta_path[t_time+1],
        xi_path[t_time], xi_path[t_time+1],
        rho_path[t_time], rho_path[t_time+1]
    )




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
