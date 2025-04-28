from random import random

import torch
from torch.utils.data import Dataset
import numpy as np
from ForwardStart import ForwardStart
from MonteCarlo import simulate_heston_qe_with_stochastic_params, extract_snapshots
from PnL_Calculator import compute_pathwise_pnl, compute_pathwise_pnl_choc, analyze_pnl_numpy
import random

# Fix all seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class GreekCorrectionDatasetV2(Dataset):
    def __init__(self, St, St1, vt, vt1, delta, vega, theta, vanna, volga, pnl_tot):
        self.St = torch.tensor(St, dtype=torch.float32) if not torch.is_tensor(St) else St
        self.St1 = torch.tensor(St1, dtype=torch.float32) if not torch.is_tensor(St1) else St1
        self.vt = torch.tensor(vt, dtype=torch.float32) if not torch.is_tensor(vt) else vt
        self.vt1 = torch.tensor(vt1, dtype=torch.float32) if not torch.is_tensor(vt1) else vt1
        self.pnl_tot = torch.tensor(pnl_tot, dtype=torch.float32) if not torch.is_tensor(pnl_tot) else pnl_tot

        # Grecs fixes
        self.delta = torch.tensor(delta, dtype=torch.float32)
        self.vega = torch.tensor(vega, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.vanna = torch.tensor(vanna, dtype=torch.float32)
        self.volga = torch.tensor(volga, dtype=torch.float32)

    def __len__(self):
        return self.St.shape[0]

    def __getitem__(self, idx):
        log_return = torch.log(self.St1[idx] / self.St[idx] + 1e-8)  # éviter division par zéro
        delta_v = self.vt1[idx] - self.vt[idx]
        interaction = self.St[idx] * self.vt[idx]

        features = torch.stack([
            self.St[idx],
            self.St1[idx],
            self.vt[idx],
            self.vt1[idx],
            log_return,
            delta_v,
            interaction,
            self.delta,
            self.vega,
            self.theta,
            self.vanna,
            self.volga
        ])
        target = self.pnl_tot[idx]
        return features, target


import torch
import torch.nn as nn

class GreekCorrectionMLPV2(nn.Module):
    def __init__(self):
        super(GreekCorrectionMLPV2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.model(x)


def compute_pnl_explained(coeffs, features):
    """
    Recalcule pnl_explained corrigé avec les coefficients prédits par le réseau.

    Args:
        coeffs : (batch_size, 5) - a, b, c, d, e prédits
        features : (batch_size, 9) - S_t, S_t+1, v_t, v_t+1, delta, vega, theta, vanna, volga

    Returns:
        pnl_explained_corrected : (batch_size,)
    """
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

import torch.nn.functional as F

def pnl_loss(pred_coeffs, features, targets):
    """
    Loss entre pnl_explained corrigé et pnl_tot.

    Args:
        pred_coeffs : (batch_size, 5) sortie du modèle
        features : (batch_size, 9) features d'entrée
        targets : (batch_size,) pnl_tot

    Returns:
        loss : torch scalar
    """
    pnl_explained = compute_pnl_explained(pred_coeffs, features)
    return F.mse_loss(pnl_explained, targets)


if __name__ == "__main__":
    import numpy as np
    from torch.utils.data import DataLoader

    calibrated_params = {'kappa': 2.41300630569458, 'v0': 0.029727613553404808, 'theta': 0.04138144478201866,
                         'sigma': 0.3084869682788849, 'rho': -0.8905978202819824}

    S, v, new_params = simulate_heston_qe_with_stochastic_params(5667.65,
                                                                 v0=calibrated_params["v0"],
                                                                 r=0.03927,
                                                                 kappa=calibrated_params["kappa"],
                                                                 theta=calibrated_params["theta"],
                                                                 xi=calibrated_params["sigma"],
                                                                 rho=calibrated_params["rho"], n_paths=30_000, seed=42,
                                                                 nb_of_plots=1, t_time=50)

    St, St1, vt, vt1 = extract_snapshots(S, v, t=50)

    # Ton forward model
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

    # Tes Grecs constants
    delta = forward_model.compute_greek("delta")
    vega = forward_model.compute_greek("vega")
    theta = forward_model.compute_greek("theta")
    vanna = forward_model.compute_greek("vanna")
    volga = forward_model.compute_greek("volga")

    # Ton pnl_tot simulé
    pnl_tot = compute_pathwise_pnl_choc(St, St1, vt, vt1, forward_model, new_params)

    # Convertis ton pnl_tot en numpy si besoin
    pnl_tot = pnl_tot.detach().cpu().numpy()

    # --- Ensuite tu construis comme avant ---
    dataset = GreekCorrectionDatasetV2(St, St1, vt, vt1, delta, vega, theta, vanna, volga, pnl_tot)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = GreekCorrectionMLPV2()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 94

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

    print("✅ Training terminé avec TES DONNÉES RÉELLES !")

    torch.save(model.state_dict(), "greek_correction_model.pth")
    print("✅ Modèle sauvegardé sous 'greek_correction_model.pth'")




    S, v, new_params = simulate_heston_qe_with_stochastic_params(5667.65,
                                                                 v0=calibrated_params["v0"],
                                                                 r=0.03927,
                                                                 kappa=calibrated_params["kappa"],
                                                                 theta=calibrated_params["theta"],
                                                                 xi=calibrated_params["sigma"],
                                                                 rho=calibrated_params["rho"], n_paths=30_000, seed=42,
                                                                 nb_of_plots=1, t_time=60)



    St_test, St1_test, vt_test, vt1_test = extract_snapshots(S, v, t=60)

    # (2) Tes grecs sont les mêmes
    # (on suppose que ton forward_model est déjà défini)

    delta = forward_model.compute_greek("delta")
    vega = forward_model.compute_greek("vega")
    theta = forward_model.compute_greek("theta")
    vanna = forward_model.compute_greek("vanna")
    volga = forward_model.compute_greek("volga")

    # (3) Ton pnl_tot_test (différence de prix simulé entre t=55 et t=56)
    # Attention : utilise bien compute_pathwise_pnl ou compute_pathwise_pnl_choc
    pnl_tot_test = compute_pathwise_pnl_choc(St_test, St1_test, vt_test, vt1_test, forward_model, new_params)
    pnl_tot_test = pnl_tot_test.detach().cpu().numpy()

    # (4) Créer le dataset de test
    test_dataset = GreekCorrectionDatasetV2(St_test, St1_test, vt_test, vt1_test, delta, vega, theta, vanna, volga,
                                          pnl_tot_test)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # (5) Charger ton modèle entraîné si besoin
    # model = GreekCorrectionMLP()
    # model.load_state_dict(torch.load("greek_correction_mlp.pth"))
    model.eval()

    # (6) Faire les prédictions
    all_pred_pnl = []
    all_real_pnl = []

    with torch.no_grad():
        for features, targets in test_loader:
            coeffs = model(features)
            pnl_pred = compute_pnl_explained(coeffs, features)
            all_pred_pnl.append(pnl_pred)
            all_real_pnl.append(targets)

    # Concaténer tous les résultats
    all_pred_pnl = torch.cat(all_pred_pnl).cpu().numpy()
    all_real_pnl = torch.cat(all_real_pnl).cpu().numpy()

    # (1) Faire repasser les features pour récupérer les coefficients (a,b,c,d,e)
    all_coeffs = []

    with torch.no_grad():
        for features, _ in test_loader:
            coeffs = model(features)
            all_coeffs.append(coeffs)

    all_coeffs = torch.cat(all_coeffs).cpu().numpy()  # shape (n_paths, 5)

    # (2) Extraire les coefficients a, b, c, d, e
    a_pred = all_coeffs[:, 0]
    b_pred = all_coeffs[:, 1]
    c_pred = all_coeffs[:, 2]
    d_pred = all_coeffs[:, 3]
    e_pred = all_coeffs[:, 4]

    print("Sample of a coefficients:", a_pred[:10])
    print("Sample of b coefficients:", b_pred[:10])
    print("Sample of c coefficients:", c_pred[:10])
    print("Sample of d coefficients:", d_pred[:10])
    print("Sample of e coefficients:", e_pred[:10])

    # (3) Calculer les nouveaux Grecs corrigés chemin par chemin
    # (Tes grecs de base sont constants pour tous les chemins)
    delta_corr = a_pred * delta
    vega_corr = b_pred * vega
    theta_corr = c_pred * theta
    vanna_corr = d_pred * vanna
    volga_corr = e_pred * volga

    # Affiche quelques exemples
    print("\nExemples de grecs corrigés (chemin 0 à 9):")
    for i in range(10):
        print(
            f"Path {i}: Δ={delta_corr[i]:.4f}, ν={vega_corr[i]:.4f}, θ={theta_corr[i]:.6f}, Vanna={vanna_corr[i]:.6f}, Volga={volga_corr[i]:.4f}")

    # (7) Calculer l'erreur
    from sklearn.metrics import mean_squared_error

    rmse = np.sqrt(mean_squared_error(all_real_pnl, all_pred_pnl))
    print(f"✅ Test terminé : RMSE sur t=55 ➔ t=56 = {rmse:.6f}")

    analyze_pnl_numpy(all_real_pnl, all_pred_pnl)

