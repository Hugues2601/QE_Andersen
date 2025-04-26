from config import CONFIG
import torch
from ForwardStart import ForwardStart

def compute_pathwise_pnl(S_t, S_tp1, v_t, v_tp1, base_model, batch_size=1000, sub_batch_size=100):
    """
    Calcule le PnL pathwise entre t et t+1 en utilisant les paramètres simulés et le modèle ForwardStart.

    Paramètres
    ----------
    S_t, S_tp1 : ndarray or tensor
        Valeurs de S_t et S_{t+1} pour chaque chemin.
    v_t, v_tp1 : ndarray or tensor
        Valeurs de v_t et v_{t+1} pour chaque chemin.
    base_model : ForwardStart
        Modèle ForwardStart instancié avec les bons paramètres fixes (T0, T1, T2, etc.)
    batch_size : int
        Nombre de chemins traités à la fois dans un batch principal.
    sub_batch_size : int
        Nombre de chemins traités à la fois dans un sous-batch interne.

    Retourne
    --------
    pnl : torch.Tensor
        Différence de prix entre t+1 et t pour chaque chemin (shape: n_paths,)
    """
    device = CONFIG.device

    # Convert to torch tensors if necessary
    S_t = torch.tensor(S_t, device=device, dtype=torch.float32) if not torch.is_tensor(S_t) else S_t.to(device)
    S_tp1 = torch.tensor(S_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(S_tp1) else S_tp1.to(device)
    v_t = torch.tensor(v_t, device=device, dtype=torch.float32) if not torch.is_tensor(v_t) else v_t.to(device)
    v_tp1 = torch.tensor(v_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(v_tp1) else v_tp1.to(device)

    # Tenseur du strike
    k = base_model.k.item()

    # Constantes fixes
    T0 = base_model.T0.item()
    T1 = base_model.T1.item()
    T2 = base_model.T2.item()
    r = base_model.r.item()
    kappa = base_model.kappa.item()
    theta = base_model.theta.item()
    sigma = base_model.sigma.item()
    rho = base_model.rho.item()

    n_paths = S_t.shape[0]
    pnl_list = []

    with torch.no_grad():
        for i in range(0, n_paths, batch_size):
            i_end = min(i + batch_size, n_paths)

            # batch slice
            S_t_batch = S_t[i:i_end]
            S_tp1_batch = S_tp1[i:i_end]
            v_t_batch = v_t[i:i_end]
            v_tp1_batch = v_tp1[i:i_end]

            price_t_list = []
            price_tp1_list = []

            for j in range(0, i_end - i, sub_batch_size):
                j_end = min(j + sub_batch_size, i_end - i)

                # sub-batch slice
                S_t_sub = S_t_batch[j:j_end]
                S_tp1_sub = S_tp1_batch[j:j_end]
                v_t_sub = v_t_batch[j:j_end]
                v_tp1_sub = v_tp1_batch[j:j_end]

                # Modèle t
                model_t = ForwardStart(
                    S0=S_t_sub,
                    k=k,
                    T0=T0,
                    T1=T1,
                    T2=T2,
                    r=r,
                    kappa=kappa,
                    v0=v_t_sub,
                    theta=theta,
                    sigma=sigma,
                    rho=rho
                )
                price_t_sub = model_t.heston_price()
                price_t_list.append(price_t_sub)

                # Modèle t+1
                model_tp1 = ForwardStart(
                    S0=S_tp1_sub,
                    k=k,
                    T0=T0,
                    T1=T1 - (1/252),
                    T2=T2 - (1/252),
                    r=r,
                    kappa=kappa,
                    v0=v_tp1_sub,
                    theta=theta,
                    sigma=sigma,
                    rho=rho
                )
                price_tp1_sub = model_tp1.heston_price()
                price_tp1_list.append(price_tp1_sub)

            # Concaténer
            price_t = torch.cat(price_t_list, dim=0)
            price_tp1 = torch.cat(price_tp1_list, dim=0)

            pnl_batch = price_tp1 - price_t
            pnl_list.append(pnl_batch)

    pnl = torch.cat(pnl_list, dim=0)
    return pnl


def compute_pathwise_pnl_choc(S_t, S_tp1, v_t, v_tp1, base_model, new_params, batch_size=1000, sub_batch_size=100):
    device = CONFIG.device

    # Convert to torch tensors if necessary
    S_t = torch.tensor(S_t, device=device, dtype=torch.float32) if not torch.is_tensor(S_t) else S_t.to(device)
    S_tp1 = torch.tensor(S_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(S_tp1) else S_tp1.to(device)
    v_t = torch.tensor(v_t, device=device, dtype=torch.float32) if not torch.is_tensor(v_t) else v_t.to(device)
    v_tp1 = torch.tensor(v_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(v_tp1) else v_tp1.to(device)

    kappa_t, kappa_tp1, theta_t, theta_tp1, sigma_t, sigma_tp1, rho_t, rho_tp1 = new_params

    # Ensure all params are on the right device
    kappa_t = torch.tensor(kappa_t, device=device, dtype=torch.float32) if not torch.is_tensor(kappa_t) else kappa_t.to(device)
    kappa_tp1 = torch.tensor(kappa_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(kappa_tp1) else kappa_tp1.to(device)
    theta_t = torch.tensor(theta_t, device=device, dtype=torch.float32) if not torch.is_tensor(theta_t) else theta_t.to(device)
    theta_tp1 = torch.tensor(theta_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(theta_tp1) else theta_tp1.to(device)
    sigma_t = torch.tensor(sigma_t, device=device, dtype=torch.float32) if not torch.is_tensor(sigma_t) else sigma_t.to(device)
    sigma_tp1 = torch.tensor(sigma_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(sigma_tp1) else sigma_tp1.to(device)
    rho_t = torch.tensor(rho_t, device=device, dtype=torch.float32) if not torch.is_tensor(rho_t) else rho_t.to(device)
    rho_tp1 = torch.tensor(rho_tp1, device=device, dtype=torch.float32) if not torch.is_tensor(rho_tp1) else rho_tp1.to(device)

    # Strike (fixed)
    k = base_model.k.item()
    # Time settings
    T0 = base_model.T0.item()
    T1 = base_model.T1.item()
    T2 = base_model.T2.item()
    r = base_model.r.item()

    n_paths = S_t.shape[0]
    pnl_list = []

    with torch.no_grad():  # Pas besoin de gradients ici
        for i in range(0, n_paths, batch_size):
            i_end = min(i + batch_size, n_paths)

            # batch slice
            S_t_batch = S_t[i:i_end]
            S_tp1_batch = S_tp1[i:i_end]
            v_t_batch = v_t[i:i_end]
            v_tp1_batch = v_tp1[i:i_end]
            kappa_t_batch = kappa_t[i:i_end]
            kappa_tp1_batch = kappa_tp1[i:i_end]
            theta_t_batch = theta_t[i:i_end]
            theta_tp1_batch = theta_tp1[i:i_end]
            sigma_t_batch = sigma_t[i:i_end]
            sigma_tp1_batch = sigma_tp1[i:i_end]
            rho_t_batch = rho_t[i:i_end]
            rho_tp1_batch = rho_tp1[i:i_end]

            # --- Sous-batch loop ---
            price_t_list = []
            price_tp1_list = []

            for j in range(0, i_end - i, sub_batch_size):
                j_end = min(j + sub_batch_size, i_end - i)

                # Sub-batch slice
                S_t_sub = S_t_batch[j:j_end]
                S_tp1_sub = S_tp1_batch[j:j_end]
                v_t_sub = v_t_batch[j:j_end]
                v_tp1_sub = v_tp1_batch[j:j_end]
                kappa_t_sub = kappa_t_batch[j:j_end]
                kappa_tp1_sub = kappa_tp1_batch[j:j_end]
                theta_t_sub = theta_t_batch[j:j_end]
                theta_tp1_sub = theta_tp1_batch[j:j_end]
                sigma_t_sub = sigma_t_batch[j:j_end]
                sigma_tp1_sub = sigma_tp1_batch[j:j_end]
                rho_t_sub = rho_t_batch[j:j_end]
                rho_tp1_sub = rho_tp1_batch[j:j_end]

                # Modèle ForwardStart pour t
                model_t = ForwardStart(
                    S0=S_t_sub,
                    k=k,
                    T0=T0,
                    T1=T1,
                    T2=T2,
                    r=r,
                    kappa=kappa_t_sub,
                    v0=v_t_sub,
                    theta=theta_t_sub,
                    sigma=sigma_t_sub,
                    rho=rho_t_sub
                )
                price_t_sub = model_t.heston_price()
                price_t_list.append(price_t_sub)

                # Modèle ForwardStart pour t+1
                model_tp1 = ForwardStart(
                    S0=S_tp1_sub,
                    k=k,
                    T0=T0,
                    T1=T1 - (1/252),
                    T2=T2 - (1/252),
                    r=r,
                    kappa=kappa_tp1_sub,
                    v0=v_tp1_sub,
                    theta=theta_tp1_sub,
                    sigma=sigma_tp1_sub,
                    rho=rho_tp1_sub
                )
                price_tp1_sub = model_tp1.heston_price()
                price_tp1_list.append(price_tp1_sub)

            # Concaténer tous les sous-batchs
            price_t = torch.cat(price_t_list, dim=0)
            price_tp1 = torch.cat(price_tp1_list, dim=0)

            # PnL sur ce batch
            pnl_batch = price_tp1 - price_t
            pnl_list.append(pnl_batch)

    pnl = torch.cat(pnl_list, dim=0)
    return pnl




import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def analyze_pnl_numpy(pnl_tot, pnl_explained, bins=100):
    """
    Analyse complète du PnL inexpliqué à partir du PnL total et expliqué.

    Paramètres
    ----------
    pnl_tot : np.ndarray
        PnL total (par chemin)
    pnl_explained : np.ndarray
        PnL expliqué (par chemin)
    bins : int
        Nombre de bins pour les histogrammes

    Affiche :
    --------
    - Histogramme du PnL inexpliqué
    - Q-Q Plot
    - PnL cumulé (total / expliqué / inexpliqué)
    - Ratio inexpliqué / total (%)
    - Statistiques descriptives
    """

    pnl_unexplained = pnl_tot - pnl_explained
    ratio = np.sum(np.abs(pnl_unexplained)) / np.sum(np.abs(pnl_tot))
    rmse = np.sqrt(np.mean((pnl_unexplained) ** 2))
    corr = np.corrcoef(pnl_tot, pnl_unexplained)[0, 1]

    print("Unexplained PnL Analysis:")
    print(f"→ Unexplained / Total Ratio        : {ratio:.2%}")
    print(f"→ Mean Unexplained PnL             : {np.mean(pnl_unexplained):.6f}")
    print(f"→ Std Dev of Unexplained PnL       : {np.std(pnl_unexplained):.6f}")
    print(f"→ Skewness                         : {stats.skew(pnl_unexplained):.4f}")
    print(f"→ Kurtosis                         : {stats.kurtosis(pnl_unexplained):.4f}")
    print("")
    print("Additional Info:")
    print(f"→ Mean Total PnL                   : {np.mean(pnl_tot):.6f}")
    print(f"→ Std Dev of Total PnL             : {np.std(pnl_tot):.6f}")
    print(f"→ Mean Explained PnL               : {np.mean(pnl_explained):.6f}")
    print(f"→ Std Dev of Explained PnL         : {np.std(pnl_explained):.6f}")

    # Histogramme
    plt.figure(figsize=(10, 5))
    plt.hist(pnl_unexplained, bins=bins, alpha=0.75, color='orange', edgecolor='black')
    plt.title("Distribution du PnL inexpliqué")
    plt.xlabel("PnL inexpliqué")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()

    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

    # Histogram of unexplained PnL with xlim
    axs[0].hist(pnl_unexplained, bins=150, alpha=0.75, color='black', edgecolor='white')
    axs[0].set_title("Distribution of Unexplained PnL")
    axs[0].set_xlabel("Unexplained PnL")
    axs[0].set_ylabel("Frequency")
    axs[0].grid(True, linestyle='--', alpha=0.3)
    axs[0].set_xlim(-1, 1)  # 💡 Limite l'axe horizontal entre -1 et 1

    # Scatter plot of unexplained PnL per path
    axs[1].scatter(np.arange(len(pnl_unexplained)), pnl_unexplained, alpha=0.6, color='black', s=10)
    axs[1].axhline(0, color='gray', linestyle='dashed', linewidth=1)

    # 💡 Ajoute lignes moyenne et médiane
    mean_pnl = np.mean(pnl_unexplained)
    median_pnl = np.median(pnl_unexplained)

    axs[1].axhline(mean_pnl, color='red', linestyle='--', linewidth=1, label=f'Mean = {mean_pnl:.2e}')
    axs[1].legend(fontsize=9, loc='upper right')

    axs[1].set_title("Unexplained PnL by Path")
    axs[1].set_xlabel("Path Index")
    axs[1].set_ylabel("Unexplained PnL")
    axs[1].grid(True, linestyle='--', alpha=0.3)

    # Layout
    plt.tight_layout()
    plt.savefig("unexplained_pnl_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()








