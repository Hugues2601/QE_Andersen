import numpy as np
import pandas as pd
from scipy.stats import stats

from ForwardStart import ForwardStart
import torch
from config import CONFIG
from MonteCarlo import simulate_heston_qe, plot_terminal_distributions, extract_snapshots, \
    simulate_heston_qe_with_stochastic_params
from PnL_Calculator import compute_pathwise_pnl, analyze_pnl_numpy, compute_pathwise_pnl_choc


calibrated_params = {'kappa': 2.41300630569458, 'v0': 0.029727613553404808, 'theta': 0.04138144478201866,
                     'sigma': 0.3084869682788849, 'rho': -0.8905978202819824}

def simulate_forward_pnl_life(S, v, forward_model, T1, T2, dt=1/252):
    import matplotlib.pyplot as plt
    import numpy as np

    n_steps = 186

    delta = forward_model.compute_greek("delta")
    vega = forward_model.compute_greek("vega")
    theta = forward_model.compute_greek("theta")
    vanna = forward_model.compute_greek("vanna")
    volga = forward_model.compute_greek("volga")

    pnl_total_list = []
    pnl_explained_list = []
    pnl_unexplained_list = []

    for t in range(n_steps):
        St = S[t]
        St1 = S[t + 1]
        vt = v[t]
        vt1 = v[t + 1]

        T0_t = t * dt
        T1_t = T1 - t * dt
        T2_t = T2 - t * dt

        forward_model.T0 = torch.tensor(T0_t, device=CONFIG.device, requires_grad=True)
        forward_model.T1 = torch.tensor(T1_t, device=CONFIG.device)
        forward_model.T2 = torch.tensor(T2_t, device=CONFIG.device)

        forward_model_t = ForwardStart(S0=St, k=1, r=0.03927, T0=0.0, T1=T1_t, T2=T2_t,
                               kappa=calibrated_params['kappa'], v0=vt,
                               theta=calibrated_params["theta"],
                               sigma=calibrated_params['sigma'],
                               rho=calibrated_params['rho'])

        forward_model_t1 = ForwardStart(S0=St1, k=1, r=0.03927, T0=0.0, T1=T1_t - dt, T2=T2_t - dt,
                                kappa=calibrated_params['kappa'], v0=vt1,
                                theta=calibrated_params["theta"],
                                sigma=calibrated_params['sigma'],
                                rho=calibrated_params['rho'])

        price_t = forward_model_t.heston_price()

        price_t1 = forward_model_t1.heston_price()

        delta = forward_model_t.compute_greek("delta")
        vega = forward_model_t.compute_greek("vega")
        theta = forward_model_t.compute_greek("theta")
        vanna = forward_model_t.compute_greek("vanna")
        volga = forward_model_t.compute_greek("volga")

        price_t = price_t.detach().cpu().numpy()
        price_t1 = price_t1.detach().cpu().numpy()
        pnl_tot = price_t1 - price_t

        delta_c = delta * (St1 - St)
        vega_c = vega * (np.sqrt(vt1) - np.sqrt(vt))
        theta_c = theta * dt
        vanna_c = vanna * (St1 - St) * (np.sqrt(vt1) - np.sqrt(vt))
        volga_c = 0.5 * volga * (np.sqrt(vt1) - np.sqrt(vt))**2

        pnl_explained = delta_c + vega_c + theta_c + vanna_c + volga_c
        pnl_unexplained = pnl_tot - pnl_explained

        pnl_total_list.append(pnl_tot)
        pnl_explained_list.append(pnl_explained)
        pnl_unexplained_list.append(pnl_unexplained)

    # 📊 Plot de l’évolution moyenne des PnL
    avg_tot = [np.mean(p) for p in pnl_total_list]
    avg_expl = [np.mean(p) for p in pnl_explained_list]
    avg_unexpl = [np.mean(p) for p in pnl_unexplained_list]
    steps = np.arange(n_steps)

    # 📊 Cumul du PnL par jour
    cum_tot = np.cumsum([np.mean(p) for p in pnl_total_list])
    cum_expl = np.cumsum([np.mean(p) for p in pnl_explained_list])
    cum_unexpl = np.cumsum([np.mean(p) for p in pnl_unexplained_list])
    steps = np.arange(n_steps)

    plt.figure(figsize=(12, 5))
    plt.plot(steps, cum_tot, label="PnL total cumulé", linewidth=2)
    plt.plot(steps, cum_expl, label="PnL expliqué cumulé", linewidth=2)
    plt.plot(steps, cum_unexpl, label="PnL inexpliqué cumulé", linewidth=2)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.title("PnL cumulé (moyenne sur tous les chemins)")
    plt.xlabel("Temps (jours)")
    plt.ylabel("PnL cumulé")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pnl_total_list, pnl_explained_list, pnl_unexplained_list



S, v = simulate_heston_qe(5667.65,
                   calibrated_params["v0"],
                   0.03927,
                   calibrated_params["kappa"],
                   calibrated_params["theta"],
                   calibrated_params["sigma"],
                   calibrated_params["rho"],
                          n_paths=1)

forward_model = ForwardStart(S0=5667.65,
                             k=1,
                             T0=0.0,
                             T1=0.75,
                             T2=1.5,
                             r=0.03927,
                             kappa=calibrated_params["kappa"],
                             v0=calibrated_params["v0"],
                             theta=calibrated_params["theta"],
                             sigma=calibrated_params["sigma"],
                             rho=calibrated_params["rho"])

print("PRIX OPTION A T0", forward_model.heston_price())

pnl_totals, pnl_explained, pnl_unexplained = simulate_forward_pnl_life(S, v, forward_model, T1=0.75, T2=1.5)
import numpy as np

flat_pnl_total = np.concatenate(pnl_totals)
flat_pnl_explained = np.concatenate(pnl_explained)
flat_pnl_unexplained = np.concatenate(pnl_unexplained)

for i in range(len(pnl_totals)):
    if np.any(np.isnan(pnl_totals[i])):
        print(f"⚠️ NaN détecté dans pnl_totals à l’étape {i}")


print("🔍 Résumé de l’analyse du PnL de l’option forward start (sur toute sa durée de vie) :")
print("----------------------------------------------------------")
print(f"Nombre total de points analysés  : {len(flat_pnl_total)}")
print("")

print("📊 Statistiques globales :")
print(f"→ PnL total      : Moy = {np.mean(flat_pnl_total):.6f} | Std = {np.std(flat_pnl_total):.6f}")
print(f"→ PnL expliqué   : Moy = {np.mean(flat_pnl_explained):.6f} | Std = {np.std(flat_pnl_explained):.6f}")
print(f"→ PnL inexpliqué : Moy = {np.mean(flat_pnl_unexplained):.6f} | Std = {np.std(flat_pnl_unexplained):.6f}")
print("")

# Ratio inexpliqué
ratio_inexplique = np.sum(np.abs(flat_pnl_unexplained)) / np.sum(np.abs(flat_pnl_total))
print("📉 Ratio inexpliqué / total :", f"{ratio_inexplique:.2%}")

# RMSE + skew + kurt
rmse = np.sqrt(np.mean(flat_pnl_unexplained**2))
skew = stats.skew(flat_pnl_unexplained)
kurt = stats.kurtosis(flat_pnl_unexplained)
print(f"→ RMSE (inexpliqué)           : {rmse:.6f}")
print(f"→ Skewness (inexpliqué)       : {skew:.4f}")
print(f"→ Kurtosis (inexpliqué)       : {kurt:.4f}")
print("----------------------------------------------------------")
