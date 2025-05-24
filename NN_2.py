import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DirectGreekLearningDataset(Dataset):
    def __init__(self, St, St1, vt, vt1, pnl_tot, T_to_maturity=0.5):
        self.St = torch.tensor(St, dtype=torch.float32)
        self.St1 = torch.tensor(St1, dtype=torch.float32)
        self.vt = torch.tensor(vt, dtype=torch.float32)
        self.vt1 = torch.tensor(vt1, dtype=torch.float32)
        self.pnl_tot = torch.tensor(pnl_tot, dtype=torch.float32)
        self.T_to_maturity = T_to_maturity

        # Normalisation des données (très important pour la convergence)
        self.St_mean = self.St.mean()
        self.St_std = self.St.std()
        self.vt_mean = self.vt.mean()
        self.vt_std = self.vt.std()

        self.St_norm = (self.St - self.St_mean) / self.St_std
        self.St1_norm = (self.St1 - self.St_mean) / self.St_std
        self.vt_norm = (self.vt - self.vt_mean) / self.vt_std
        self.vt1_norm = (self.vt1 - self.vt_mean) / self.vt_std

        # Statistiques pour dénormalisation
        self.pnl_mean = self.pnl_tot.mean()
        self.pnl_std = self.pnl_tot.std()
        self.pnl_norm = (self.pnl_tot - self.pnl_mean) / self.pnl_std

    def __len__(self):
        return self.St.shape[0]

    def __getitem__(self, idx):
        S_t = self.St_norm[idx]
        S_tp1 = self.St1_norm[idx]
        v_t = self.vt_norm[idx]
        v_tp1 = self.vt1_norm[idx]

        sqrt_vt = torch.sqrt(torch.clamp(v_t, min=0))
        sqrt_vtp1 = torch.sqrt(torch.clamp(v_tp1, min=0))

        delta_S = S_tp1 - S_t
        delta_sqrt_v = sqrt_vtp1 - sqrt_vt
        log_return = torch.log(self.St1[idx] / self.St[idx])

        features = torch.stack([
            S_t,  # Prix normalisé à t
            S_tp1,  # Prix normalisé à t+1
            v_t,  # Volatilité normalisée à t
            v_tp1,  # Volatilité normalisée à t+1
            sqrt_vt,  # Racine carrée de la volatilité à t
            sqrt_vtp1,  # Racine carrée de la volatilité à t+1
            delta_S,  # Variation du prix
            log_return,  # Log-rendement
            delta_sqrt_v,  # Variation de la racine carrée de la volatilité
            S_t * sqrt_vt,  # Produit prix * sqrt(vol) à t
            S_tp1 * sqrt_vtp1,  # Produit prix * sqrt(vol) à t+1
            torch.tensor(self.T_to_maturity)  # Temps jusqu'à maturité
        ])

        # Target est normalisé pour meilleure convergence
        target = self.pnl_norm[idx]
        return features, target

    def denormalize_pnl(self, normalized_pnl):
        """Dénormalise le PnL prédit"""
        return normalized_pnl * self.pnl_std + self.pnl_mean

    def get_normalization_params(self):
        """Retourne les paramètres de normalisation pour utilisation lors de l'inférence"""
        return {
            'St_mean': self.St_mean.item(),
            'St_std': self.St_std.item(),
            'vt_mean': self.vt_mean.item(),
            'vt_std': self.vt_std.item(),
            'pnl_mean': self.pnl_mean.item(),
            'pnl_std': self.pnl_std.item()
        }


class ResidualBlock(nn.Module):
    """Bloc résiduel pour améliorer la propagation des gradients"""

    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Connexion résiduelle
        return self.relu(out)


class DirectGreekPredictorMLP(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DirectGreekPredictorMLP, self).__init__()

        # Entrée: 12 features
        self.input_layer = nn.Sequential(
            nn.Linear(12, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Couches intermédiaires avec blocs résiduels
        self.res_block1 = ResidualBlock(128)
        self.res_block2 = ResidualBlock(128)

        # Couche intermédiaire avec dropout
        self.intermediate = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Couche de sortie pour les 5 grecs (delta, vega, theta, vanna, volga)
        self.output_layer = nn.Linear(64, 5)

        # Initialisation des poids pour meilleure convergence
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.intermediate(x)
        x = self.output_layer(x)
        return x


def compute_pnl_explained(predicted_greeks, features, dt=1/252):
    """
    Recalcule le PnL expliqué avec les grecs prédits directement par le modèle.

    Args:
        predicted_greeks : (batch_size, 5) grecs prédits par le MLP
        features : (batch_size, 12) caractéristiques normalisées
        dt : pas de temps (1/252 pour un jour de trading)

    Returns:
        pnl_explained : (batch_size,) PnL expliqué normalisé
    """
    # Extrait les variations pertinentes
    delta_S = features[:, 6]  # Variation du prix normalisé
    delta_sqrt_v = features[:, 8]  # Variation de la racine carrée de la volatilité normalisée

    # Extrait les grecs
    delta = predicted_greeks[:, 0]
    vega = predicted_greeks[:, 1]
    theta = predicted_greeks[:, 2]
    vanna = predicted_greeks[:, 3]
    volga = predicted_greeks[:, 4]

    # Calcule le PnL expliqué
    pnl_explained = (
        delta * delta_S +
        vega * delta_sqrt_v +
        theta * dt +
        vanna * delta_S * delta_sqrt_v +
        0.5 * volga * (delta_sqrt_v ** 2)
    )

    return pnl_explained


def pnl_loss(predicted_greeks, features, pnl_tot, dt=1/252, l1_lambda=1e-5):
    """
    Fonction de perte améliorée avec régularisation L1

    Args:
        predicted_greeks : (batch_size, 5) grecs prédits directement
        features : (batch_size, 12) caractéristiques
        pnl_tot : (batch_size,) PnL réel
        dt : pas de temps
        l1_lambda : coefficient de régularisation L1

    Returns:
        loss : scalaire, combinaison de MSE et régularisation L1
    """
    # Calcul du PnL expliqué
    pnl_explained = compute_pnl_explained(predicted_greeks, features, dt)

    # MSE entre PnL expliqué et PnL réel
    mse_loss = F.mse_loss(pnl_explained, pnl_tot)

    # Régularisation L1 (pour encourager la parcimonie des grecs)
    l1_reg = torch.sum(torch.abs(predicted_greeks))

    # Perte totale
    total_loss = mse_loss + l1_lambda * l1_reg

    return total_loss


def train_model(model, train_loader, val_loader=None, epochs=100, lr=5e-4, weight_decay=1e-4,
                patience=10, l1_lambda=1e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Fonction d'entraînement avec validation précoce et learning rate scheduling

    Args:
        model : modèle à entraîner
        train_loader : DataLoader pour les données d'entraînement
        val_loader : DataLoader pour les données de validation (option)
        epochs : nombre maximal d'époques
        lr : taux d'apprentissage initial
        weight_decay : coefficient de régularisation L2
        patience : nombre d'époques sans amélioration avant d'arrêter
        l1_lambda : coefficient de régularisation L1
        device : appareil pour l'entraînement ('cuda' ou 'cpu')

    Returns:
        model : modèle entraîné
        history : historique des pertes
    """
    print(f"Training on device: {device}")
    model = model.to(device)

    # Optimiseur avec learning rate scheduling et weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Pour l'arrêt précoce
    best_val_loss = float('inf')
    no_improve_epochs = 0

    # Historique d'entraînement
    history = {
        'train_loss': [],
        'val_loss': [] if val_loader else None
    }

    for epoch in range(epochs):
        # --- ENTRAÎNEMENT ---
        model.train()
        train_loss = 0.0

        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = pnl_loss(outputs, features, targets, l1_lambda=l1_lambda)

            # Backward pass
            loss.backward()

            # Clip gradient pour stabilité
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * features.size(0)

        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        # --- VALIDATION ---
        if val_loader:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for features, targets in val_loader:
                    features, targets = features.to(device), targets.to(device)
                    outputs = model(features)
                    loss = pnl_loss(outputs, features, targets, l1_lambda=0)  # Pas de régularisation en validation
                    val_loss += loss.item() * features.size(0)

            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Arrêt précoce
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), "best_direct_greek_model.pth")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    # Charger le meilleur modèle
                    model.load_state_dict(torch.load("best_direct_greek_model.pth"))
                    break

            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f}")

    return model, history


def evaluate_model(model, test_loader, normalization_params=None,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Évalue le modèle et retourne les grecs prédits directement

    Args:
        model : modèle entraîné
        test_loader : DataLoader pour les données de test
        normalization_params : paramètres de normalisation pour dénormaliser les résultats
        device : appareil pour l'inférence

    Returns:
        results : dictionnaire avec les résultats d'évaluation
    """
    model = model.to(device)
    model.eval()

    # Stocker les résultats
    all_greeks = []
    all_pred_pnl = []
    all_real_pnl = []
    all_features = []

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)

            # Prédiction des grecs directement
            greeks = model(features)

            # Calcul du PnL prédit
            pnl_pred = compute_pnl_explained(greeks, features)

            # Stocker les résultats
            all_greeks.append(greeks.cpu())
            all_pred_pnl.append(pnl_pred.cpu())
            all_real_pnl.append(targets.cpu())
            all_features.append(features.cpu())

    # Concaténer tous les résultats
    all_greeks = torch.cat(all_greeks).numpy()
    all_pred_pnl = torch.cat(all_pred_pnl).numpy()
    all_real_pnl = torch.cat(all_real_pnl).numpy()
    all_features = torch.cat(all_features).numpy()

    # Dénormaliser le PnL si nécessaire
    if normalization_params:
        all_pred_pnl = all_pred_pnl * normalization_params['pnl_std'] + normalization_params['pnl_mean']
        all_real_pnl = all_real_pnl * normalization_params['pnl_std'] + normalization_params['pnl_mean']

    # Extraire les grecs
    delta = all_greeks[:, 0]
    vega = all_greeks[:, 1]
    theta = all_greeks[:, 2]
    vanna = all_greeks[:, 3]
    volga = all_greeks[:, 4]

    # Calculer les métriques d'erreur
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(all_real_pnl, all_pred_pnl))
    r2 = r2_score(all_real_pnl, all_pred_pnl)

    results = {
        'greeks': {
            'delta': delta,
            'vega': vega,
            'theta': theta,
            'vanna': vanna,
            'volga': volga
        },
        'pnl': {
            'real': all_real_pnl,
            'predicted': all_pred_pnl
        },
        'features': all_features,
        'metrics': {
            'rmse': rmse,
            'r2': r2
        }
    }

    return results


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration pour reproductibilité
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Paramètres du modèle de Heston (ici données d'exemple)
    calibrated_params = {
        'kappa': 2.41300630569458,
        'v0': 0.029727613553404808,
        'theta': 0.04138144478201866,
        'sigma': 0.3084869682788849,
        'rho': -0.8905978202819824
    }

    # Création de données simulées
    from ForwardStart import ForwardStart
    from MonteCarlo import simulate_heston_qe_with_stochastic_params, extract_snapshots
    from PnL_Calculator import compute_pathwise_pnl_choc, analyze_pnl_numpy

    # Simulation des trajectoires
    S, v, new_params = simulate_heston_qe_with_stochastic_params(
        5667.65,
        v0=calibrated_params["v0"],
        r=0.03927,
        kappa=calibrated_params["kappa"],
        theta=calibrated_params["theta"],
        xi=calibrated_params["sigma"],
        rho=calibrated_params["rho"],
        n_paths=30_000,
        seed=42,
        nb_of_plots=1,
        t_time=48
    )

    # Extraction des snapshots
    St, St1, vt, vt1 = extract_snapshots(S, v, t=47)

    forward_model = ForwardStart(
        S0=5667.65,
        k=1,
        T0=0.0,
        T1=0.75,
        T2=1.5,
        r=0.03927,
        kappa=calibrated_params["kappa"],
        v0=calibrated_params["v0"],
        theta=calibrated_params["theta"],
        sigma=calibrated_params["sigma"],
        rho=calibrated_params["rho"]
    )


    # Calcul du PnL total
    pnl_tot = compute_pathwise_pnl_choc(St, St1, vt, vt1, forward_model, new_params)  # Pas besoin du forward_model ici
    pnl_tot = pnl_tot.detach().cpu().numpy()

    # Séparation des données en entraînement et validation
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(St))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=SEED)

    # Création des datasets
    train_dataset = DirectGreekLearningDataset(
        St[train_indices], St1[train_indices],
        vt[train_indices], vt1[train_indices],
        pnl_tot[train_indices],
        T_to_maturity=0.5  # Supposons un temps à maturité de 0.5 ans
    )

    val_dataset = DirectGreekLearningDataset(
        St[val_indices], St1[val_indices],
        vt[val_indices], vt1[val_indices],
        pnl_tot[val_indices],
        T_to_maturity=0.5
    )

    # Création des dataloaders
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Création et entraînement du modèle
    model = DirectGreekPredictorMLP(dropout_rate=0.2)

    # Entraînement avec arrêt précoce et validation
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader=val_loader,
        epochs=150,  # Maximum d'époques
        lr=5e-4,  # Learning rate initial
        weight_decay=1e-4,  # Régularisation L2
        patience=15,  # Patience pour l'arrêt précoce
        l1_lambda=1e-5  # Régularisation L1
    )

    # Sauvegarde des paramètres de normalisation pour une utilisation future
    norm_params = train_dataset.get_normalization_params()

    # Sauvegarde du modèle final et des paramètres de normalisation
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'norm_params': norm_params
    }, "direct_greek_model_complete.pth")

    print("✅ Entraînement terminé et modèle sauvegardé !")

    # === ÉVALUATION SUR UN NOUVEAU JEU DE DONNÉES ===

    # Simulation de nouvelles trajectoires
    S_test, v_test, new_params_test = simulate_heston_qe_with_stochastic_params(
        5667.65,
        v0=calibrated_params["v0"],
        r=0.03927,
        kappa=calibrated_params["kappa"],
        theta=calibrated_params["theta"],
        xi=calibrated_params["sigma"],
        rho=calibrated_params["rho"],
        n_paths=30_000,
        seed=42,  # Graine différente pour les données de test
        nb_of_plots=1,
        t_time=50
    )

    # Extraction des snapshots
    St_test, St1_test, vt_test, vt1_test = extract_snapshots(S_test, v_test, t=49)

    # Calcul du PnL total sur les données de test
    pnl_tot_test = compute_pathwise_pnl_choc(St_test, St1_test, vt_test, vt1_test, forward_model, new_params_test)
    pnl_tot_test = pnl_tot_test.detach().cpu().numpy()

    # Création du dataset de test avec les mêmes paramètres de normalisation
    test_dataset = DirectGreekLearningDataset(
        St_test, St1_test, vt_test, vt1_test, pnl_tot_test, T_to_maturity=0.4  # Temps à maturité différent
    )

    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Évaluation du modèle
    results = evaluate_model(
        trained_model,
        test_loader,
        normalization_params=norm_params
    )

    # Affichage des résultats
    print(f"\n=== RÉSULTATS D'ÉVALUATION ===")
    print(f"RMSE: {results['metrics']['rmse']:.6f}")
    print(f"R²: {results['metrics']['r2']:.6f}")

    # Afficher quelques exemples de grecs prédits
    print("\nExemples de grecs prédits (chemin 0 à 9):")
    for i in range(10):
        print(
            f"Chemin {i}: "
            f"Δ={results['greeks']['delta'][i]:.4f}, "
            f"ν={results['greeks']['vega'][i]:.4f}, "
            f"θ={results['greeks']['theta'][i]:.6f}, "
            f"Vanna={results['greeks']['vanna'][i]:.6f}, "
            f"Volga={results['greeks']['volga'][i]:.4f}"
        )

    # Distribution des grecs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    greek_names = ['delta', 'vega', 'theta', 'vanna', 'volga']

    for i, name in enumerate(greek_names):
        axes[i].hist(results['greeks'][name], bins=50, alpha=0.7)
        axes[i].set_title(f'Distribution de {name}')
        axes[i].set_xlabel('Valeur du greek')
        axes[i].set_ylabel('Fréquence')

    # Plot de PnL réel vs prédit
    axes[5].scatter(results['pnl']['real'], results['pnl']['predicted'], alpha=0.1, s=1)
    axes[5].plot([-0.1, 0.1], [-0.1, 0.1], 'r--')  # Ligne y=x
    axes[5].set_title('PnL réel vs prédit')
    axes[5].set_xlabel('PnL réel')
    axes[5].set_ylabel('PnL prédit')

    plt.tight_layout()
    plt.savefig('direct_greek_prediction_results.png')
    print("\n✅ Évaluation terminée et résultats sauvegardés !")
    analyze_pnl_numpy(results['pnl']['real'], results['pnl']['predicted'])