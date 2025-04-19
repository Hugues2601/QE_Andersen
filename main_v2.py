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

def simulate_forward_pnl_life_dual(S, v, forward_model, T1, T2, dt=1/252):
    import matplotlib.pyplot as plt
    import numpy as np

    n_steps = 186

    def compute_pnl(greeks_dynamic=False):
        pnl_total_list, pnl_explained_list, pnl_unexplained_list = [], [], []

        if not greeks_dynamic:
            delta = forward_model.compute_greek("delta")
            vega = forward_model.compute_greek("vega")
            theta = forward_model.compute_greek("theta")
            vanna = forward_model.compute_greek("vanna")
            volga = forward_model.compute_greek("volga")

        for t in range(n_steps):
            St, St1 = S[t], S[t+1]
            vt, vt1 = v[t], v[t+1]

            T0_t = t * dt
            T1_t = T1 - t * dt
            T2_t = T2 - t * dt

            forward_model_t = ForwardStart(S0=St, k=1, r=0.03927, T0=0.0, T1=T1_t, T2=T2_t,
                                           kappa=calibrated_params['kappa'], v0=vt,
                                           theta=calibrated_params["theta"],
                                           sigma=calibrated_params['sigma'],
                                           rho=calibrated_params['rho'])
            forward_model_t1 = ForwardStart(S0=St1, k=1, r=0.03927, T0=0.0, T1=T1_t-dt, T2=T2_t-dt,
                                            kappa=calibrated_params['kappa'], v0=vt1,
                                            theta=calibrated_params["theta"],
                                            sigma=calibrated_params['sigma'],
                                            rho=calibrated_params['rho'])

            price_t = forward_model_t.heston_price().detach().cpu().numpy()
            price_t1 = forward_model_t1.heston_price().detach().cpu().numpy()
            pnl_tot = price_t1 - price_t

            if greeks_dynamic:
                delta = forward_model_t.compute_greek("delta")
                vega = forward_model_t.compute_greek("vega")
                theta = forward_model_t.compute_greek("theta")
                vanna = forward_model_t.compute_greek("vanna")
                volga = forward_model_t.compute_greek("volga")

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

        return pnl_total_list, pnl_explained_list, pnl_unexplained_list

    # 🎯 Calcul des deux scénarios
    fixed_tot, fixed_expl, fixed_unexpl = compute_pnl(greeks_dynamic=False)
    dyn_tot, dyn_expl, dyn_unexpl = compute_pnl(greeks_dynamic=True)

    def compute_avg_and_cumsum(pnl_list):
        avg = [np.mean(p) for p in pnl_list]
        cumsum = np.cumsum(avg)
        return avg, cumsum

    avg_f_tot, cum_f_tot = compute_avg_and_cumsum(fixed_tot)
    avg_f_expl, cum_f_expl = compute_avg_and_cumsum(fixed_expl)
    avg_f_unexpl, cum_f_unexpl = compute_avg_and_cumsum(fixed_unexpl)

    avg_d_tot, cum_d_tot = compute_avg_and_cumsum(dyn_tot)
    avg_d_expl, cum_d_expl = compute_avg_and_cumsum(dyn_expl)
    avg_d_unexpl, cum_d_unexpl = compute_avg_and_cumsum(dyn_unexpl)

    steps = np.arange(n_steps)

    # 🔥 Plot comparatif
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex='col',
                            gridspec_kw={"height_ratios": [3, 1]})

    # Fixed Greeks
    axs[0, 0].plot(steps, cum_f_tot, label="Cumulative Total PnL", linewidth=2, color='#D28A76')
    axs[0, 0].plot(steps, cum_f_expl, label="Cumulative Explained PnL", linewidth=2, color='#9DC3C3')
    axs[0, 0].plot(steps, cum_f_unexpl, label="Cumulative Unexplained PnL", linewidth=2, color='black')
    axs[0, 0].axhline(0, color='black', linestyle='--')
    axs[0, 0].set_title("Greeks Fixed and computed at step t=0")
    axs[0, 0].set_ylabel("Cumulative PnL")
    axs[0, 0].grid(True, linestyle='--', alpha=0.3)
    axs[0, 0].legend()

    axs[1, 0].bar(steps, avg_f_unexpl, width=1.0, color='black', alpha=0.8)
    axs[1, 0].axhline(0, color='gray', linestyle='--')
    axs[1, 0].set_xlabel("Time (days)")
    axs[1, 0].set_ylabel("Unexpl. PnL")

    # Dynamic Greeks
    axs[0, 1].plot(steps, cum_d_tot, label="Cumulative Total PnL", linewidth=2, color='#D28A76')
    axs[0, 1].plot(steps, cum_d_expl, label="Cumulative Explained PnL", linewidth=2, color='#9DC3C3')
    axs[0, 1].plot(steps, cum_d_unexpl, label="Cumulative Unexplained PnL", linewidth=2, color='black')
    axs[0, 1].axhline(0, color='black', linestyle='--')
    axs[0, 1].set_title("Dynamic Greeks computed at each time step")
    axs[0, 1].grid(True, linestyle='--', alpha=0.3)
    axs[0, 1].legend()

    axs[1, 1].bar(steps, avg_d_unexpl, width=1.0, color='black', alpha=0.8)
    axs[1, 1].axhline(0, color='gray', linestyle='--')
    axs[1, 1].set_xlabel("Time (days)")

    plt.tight_layout()
    plt.savefig("fixed_v_dynamicgreeks", dpi=300)
    plt.show()

    return fixed_tot, fixed_expl, fixed_unexpl, dyn_tot, dyn_expl, dyn_unexpl




S, v = simulate_heston_qe(5667.65,
                   calibrated_params["v0"],
                   0.03927,
                   calibrated_params["kappa"],
                   calibrated_params["theta"],
                   calibrated_params["sigma"],
                   calibrated_params["rho"],
                          n_paths=1, seed=1579)



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

fixed_tot, fixed_expl, fixed_unexpl, dyn_tot, dyn_expl, dyn_unexpl = simulate_forward_pnl_life_dual(S, v, forward_model, T1=0.75, T2=1.5)

print("PRIX OPTION A T0", forward_model.heston_price())

# pnl_totals, pnl_explained, pnl_unexplained = simulate_forward_pnl_life(S, v, forward_model, T1=0.75, T2=1.5)
import numpy as np
from scipy.stats import skew, kurtosis

# Flatten
flat_fixed_tot = np.concatenate(fixed_tot)
flat_fixed_expl = np.concatenate(fixed_expl)
flat_fixed_unexpl = np.concatenate(fixed_unexpl)

flat_dyn_tot = np.concatenate(dyn_tot)
flat_dyn_expl = np.concatenate(dyn_expl)
flat_dyn_unexpl = np.concatenate(dyn_unexpl)

def print_stats(label, tot, expl, unexpl):
    ratio_inexplique = np.sum(np.abs(unexpl)) / np.sum(np.abs(tot))
    rmse = np.sqrt(np.mean(unexpl**2))
    print(f"🔍 {label} PnL Analysis")
    print("----------------------------------------------------------")
    print(f"Nombre total de points analysés : {len(tot)}\n")
    print("📊 Statistiques globales :")
    print(f"→ PnL total      : Moy = {np.mean(tot):.6f} | Std = {np.std(tot):.6f}")
    print(f"→ PnL expliqué   : Moy = {np.mean(expl):.6f} | Std = {np.std(expl):.6f}")
    print(f"→ PnL inexpliqué : Moy = {np.mean(unexpl):.6f} | Std = {np.std(unexpl):.6f}")
    print("")
    print(f"📉 Ratio inexpliqué / total : {ratio_inexplique:.2%}")
    print(f"→ RMSE (inexpliqué)         : {rmse:.6f}")
    print(f"→ Skewness (inexpliqué)     : {skew(unexpl):.4f}")
    print(f"→ Kurtosis (inexpliqué)     : {kurtosis(unexpl):.4f}")
    print("----------------------------------------------------------\n")

# 🔹 Stats pour Fixed Greeks
print_stats("Fixed Greeks", flat_fixed_tot, flat_fixed_expl, flat_fixed_unexpl)

# 🔹 Stats pour Dynamic Greeks
print_stats("Dynamic Greeks", flat_dyn_tot, flat_dyn_expl, flat_dyn_unexpl)
