import numpy as np
import pandas as pd
from ForwardStart import ForwardStart
import torch
from config import CONFIG
from MonteCarlo import simulate_heston_qe, plot_terminal_distributions, extract_snapshots, \
    simulate_heston_qe_with_stochastic_params
from PnL_Calculator import compute_pathwise_pnl, analyze_pnl_numpy, compute_pathwise_pnl_choc

calibrated_params = {'kappa': 2.41300630569458, 'v0': 0.029727613553404808, 'theta': 0.04138144478201866,
                     'sigma': 0.3084869682788849, 'rho': -0.8905978202819824}

# S, v = simulate_heston_qe(5667.65,
#                    calibrated_params["v0"],
#                    0.03927,
#                    calibrated_params["kappa"],
#                    calibrated_params["theta"],
#                    calibrated_params["sigma"],
#                    calibrated_params["rho"], n_paths=300_000, seed=42)

S, v, new_params = simulate_heston_qe_with_stochastic_params(5667.65,
                   v0=calibrated_params["v0"],
                   r=0.03927,
                   kappa=calibrated_params["kappa"],
                   theta=calibrated_params["theta"],
                   xi=calibrated_params["sigma"],
                   rho=calibrated_params["rho"], n_paths=30000, seed=42, nb_of_plots=1, t_time=60)

print("simul done")

plot_terminal_distributions(S, v)

St, St1, vt, vt1 = extract_snapshots(S, v, t=60)

print("snapshot done")

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



print("forward model done")

# pnl_tot = compute_pathwise_pnl(St, St1, vt, vt1, forward_model)

pnl_tot = compute_pathwise_pnl_choc(St, St1, vt, vt1, forward_model, new_params)

pnl_tot = pnl_tot.detach().cpu().numpy()

delta = 0.1153
vega = 67.8623
theta = -12.476408
vanna = 0.081020
volga = 881.6949

delta_contribution = delta * (St1-St)
vega_contribution = vega * (np.sqrt(vt1) - np.sqrt(vt))
theta_contribution = theta * (1/252)
vanna_contribution = vanna * (St1-St) * (np.sqrt(vt1) - np.sqrt(vt))
volga_contribution = 0.5 * volga * (np.sqrt(vt1) - np.sqrt(vt))**2


pnl_explained = delta_contribution + vega_contribution + theta_contribution + vanna_contribution + volga_contribution
print("pnl_explained", pnl_explained)
print("pnl_tot", pnl_tot)

analyze_pnl_numpy(pnl_tot, pnl_explained)











