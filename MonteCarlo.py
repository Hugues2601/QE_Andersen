import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_heston_qe(
        S0=100.0, v0=0.04, r=0.0,
        kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
        T=3, dt=1 / 252, n_paths=10_000, seed=42
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

    return S, v



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

    # Plot de S_T
    plt.figure(figsize=(10, 5))
    plt.hist(S_T, bins=100, density=True, alpha=0.7)
    plt.axvline(np.mean(S_T), color='red', linestyle='dashed', label='Moyenne')
    plt.title("Distribution de $S_T$ (prix de l'actif à maturité)")
    plt.xlabel("$S_T$")
    plt.ylabel("Densité")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot de v_T
    plt.figure(figsize=(10, 5))
    plt.hist(v_T, bins=100, density=True, color='orange', alpha=0.7)
    plt.axvline(np.mean(v_T), color='red', linestyle='dashed', label='Moyenne')
    plt.title("Distribution de $v_T$ (variance instantanée à maturité)")
    plt.xlabel("$v_T$")
    plt.ylabel("Densité")
    plt.grid(True)
    plt.legend()
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
