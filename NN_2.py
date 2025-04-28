import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from ForwardStart import ForwardStart
from MonteCarlo import simulate_heston_qe_with_stochastic_params, extract_snapshots
from PnL_Calculator import compute_pathwise_pnl, compute_pathwise_pnl_choc, analyze_pnl_numpy
from sklearn.metrics import mean_squared_error

# Fix all seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GreekCorrectionDataset(Dataset):
    """
    Dataset pour entraîner un MLP à corriger les contributions des grecs.
    """

    def __init__(self, St, St1, vt, vt1, deltas, vegas, thetas, vannas, volgas, pnl_tot):
        self.St = torch.tensor(St, dtype=torch.float32)
        self.St1 = torch.tensor(St1, dtype=torch.float32)
        self.vt = torch.tensor(vt, dtype=torch.float32)
        self.vt1 = torch.tensor(vt1, dtype=torch.float32)
        self.deltas = torch.tensor(deltas, dtype=torch.float32)
        self.vegas = torch.tensor(vegas, dtype=torch.float32)
        self.thetas = torch.tensor(thetas, dtype=torch.float32)
        self.vannas = torch.tensor(vannas, dtype=torch.float32)
        self.volgas = torch.tensor(volgas, dtype=torch.float32)
        self.pnl_tot = torch.tensor(pnl_tot, dtype=torch.float32)

    def __len__(self):
        return self.St.shape[0]

    def __getitem__(self, idx):
        features = torch.stack([
            self.St[idx],
            self.St1[idx],
            self.vt[idx],
            self.vt1[idx],
            self.deltas[idx],
            self.vegas[idx],
            self.thetas[idx],
            self.vannas[idx],
            self.volgas[idx]
        ])
        target = self.pnl_tot[idx]
        return features, target

class GreekCorrectionMLP(nn.Module):
    def __init__(self):
        super(GreekCorrectionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # Output a, b, c, d, e
        )

    def forward(self, x):
        return self.model(x)

def compute_pnl_explained(coeffs, features):
    S_t = features[:, 0]
    S_tp1 = features[:, 1]
    v_t = features[:, 2]
    v_tp1 = features[:, 3]
    delta = features[:, 4]
    vega = features[:, 5]
    theta = features[:, 6]
    vanna = features[:, 7]
    volga = features[:, 8]

    delta_S = S_tp1 - S_t
    delta_sqrt_v = torch.sqrt(torch.clamp(v_tp1, min=0)) - torch.sqrt(torch.clamp(v_t, min=0))

    a, b, c, d, e = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3], coeffs[:, 4]

    pnl_explained = (
        a * delta * delta_S
        + b * vega * delta_sqrt_v
        + c * theta * (1/252)
        + d * vanna * delta_S * delta_sqrt_v
        + e * volga * (delta_sqrt_v ** 2)
    )

    return pnl_explained

def pnl_loss(pred_coeffs, features, targets):
    pnl_explained = compute_pnl_explained(pred_coeffs, features)
    return F.mse_loss(pnl_explained, targets)

def compute_pathwise_greeks(St, vt, params, r=0.03927):
    """
    Calcule les grecs chemin-par-chemin (delta, vega, theta, vanna, volga) en prenant
    S_t et v_t spécifiques pour chaque chemin.
    """
    deltas = []
    vegas = []
    thetas = []
    vannas = []
    volgas = []

    for i in range(len(St)):
        model_i = ForwardStart(
            S0=St[i],
            k=1,
            T0=0.0,
            T1=0.75,
            T2=1.5,
            r=r,
            kappa=params["kappa"],
            v0=vt[i],
            theta=params["theta"],
            sigma=params["sigma"],
            rho=params["rho"]
        )
        deltas.append(model_i.compute_greek("delta"))
        vegas.append(model_i.compute_greek("vega"))
        thetas.append(model_i.compute_greek("theta"))
        vannas.append(model_i.compute_greek("vanna"))
        volgas.append(model_i.compute_greek("volga"))

    return np.array(deltas), np.array(vegas), np.array(thetas), np.array(vannas), np.array(volgas)

if __name__ == "__main__":
    # --- Simulation données d'entraînement ---
    calibrated_params = {'kappa': 2.413, 'v0': 0.0297, 'theta': 0.0413, 'sigma': 0.308, 'rho': -0.891}

    S, v, new_params = simulate_heston_qe_with_stochastic_params(
        5667.65,
        v0=calibrated_params["v0"],
        r=0.03927,
        kappa=calibrated_params["kappa"],
        theta=calibrated_params["theta"],
        xi=calibrated_params["sigma"],
        rho=calibrated_params["rho"],
        n_paths=5000,
        seed=42,
        nb_of_plots=1,
        t_time=50
    )

    St, St1, vt, vt1 = extract_snapshots(S, v, t=50)

    # Compute Grecs chemin par chemin
    deltas, vegas, thetas, vannas, volgas = compute_pathwise_greeks(St, vt, calibrated_params)

    pnl_tot = compute_pathwise_pnl_choc(St, St1, vt, vt1, ForwardStart(S0=5667.65, k=1.0, T0=0.0, T1=0.75, T2=1.5, r=0.03927,
                                                                       kappa=calibrated_params["kappa"],
                                                                       v0=calibrated_params["v0"],
                                                                       theta=calibrated_params["theta"],
                                                                       sigma=calibrated_params["sigma"],
                                                                       rho=calibrated_params["rho"]),
                                        new_params)
    pnl_tot = pnl_tot.detach().cpu().numpy()

    dataset = GreekCorrectionDataset(St, St1, vt, vt1, deltas, vegas, thetas, vannas, volgas, pnl_tot)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = GreekCorrectionMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 1000

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = pnl_loss(outputs, features, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {epoch_loss:.6f}")

    print("✅ Training terminé avec TES DONNÉES réelles chemin-par-chemin !")

    # --- TEST --- sur nouveau snapshot
    S, v, new_params = simulate_heston_qe_with_stochastic_params(
        5667.65,
        v0=calibrated_params["v0"],
        r=0.03927,
        kappa=calibrated_params["kappa"],
        theta=calibrated_params["theta"],
        xi=calibrated_params["sigma"],
        rho=calibrated_params["rho"],
        n_paths=5000,
        seed=42,
        nb_of_plots=1,
        t_time=55
    )

    St_test, St1_test, vt_test, vt1_test = extract_snapshots(S, v, t=55)
    deltas_test, vegas_test, thetas_test, vannas_test, volgas_test = compute_pathwise_greeks(St_test, vt_test, calibrated_params)

    pnl_tot_test = compute_pathwise_pnl_choc(St_test, St1_test, vt_test, vt1_test,
                                             ForwardStart(S0=5667.65, k=1.0, T0=0.0, T1=0.75, T2=1.5, r=0.03927,
                                                          kappa=calibrated_params["kappa"],
                                                          v0=calibrated_params["v0"],
                                                          theta=calibrated_params["theta"],
                                                          sigma=calibrated_params["sigma"],
                                                          rho=calibrated_params["rho"]),
                                             new_params)
    pnl_tot_test = pnl_tot_test.detach().cpu().numpy()

    test_dataset = GreekCorrectionDataset(St_test, St1_test, vt_test, vt1_test, deltas_test, vegas_test,
                                          thetas_test, vannas_test, volgas_test, pnl_tot_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model.eval()
    all_pred_pnl = []
    all_real_pnl = []

    with torch.no_grad():
        for features, targets in test_loader:
            coeffs = model(features)
            pnl_pred = compute_pnl_explained(coeffs, features)
            all_pred_pnl.append(pnl_pred)
            all_real_pnl.append(targets)

    all_pred_pnl = torch.cat(all_pred_pnl).cpu().numpy()
    all_real_pnl = torch.cat(all_real_pnl).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(all_real_pnl, all_pred_pnl))
    print(f"✅ Test terminé : RMSE sur t=55 ➔ t=56 = {rmse:.6f}")


    analyze_pnl_numpy(all_real_pnl, all_pred_pnl)


