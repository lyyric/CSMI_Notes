import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import numpy as np
import time

# Configuration
plt.ion()
plt.style.use("default")

key = random.PRNGKey(42)


# Datasets corrigés
def make_circles(
    n_samples: int = 1000, noise: float = 0.05, key: jax.random.PRNGKey = None
):
    if key is None:
        key = random.PRNGKey(0)

    key1, key2 = random.split(key, 2)
    labels = random.bernoulli(key1, 0.5, (n_samples,))
    angles = random.uniform(key2, (n_samples,), minval=0, maxval=2 * jnp.pi)
    radii = jnp.where(labels, 1.0, 0.3)

    x = radii * jnp.cos(angles)
    y = radii * jnp.sin(angles)

    noise_key1, noise_key2 = random.split(key, 2)
    x += random.normal(noise_key1, (n_samples,)) * noise
    y += random.normal(noise_key2, (n_samples,)) * noise

    return jnp.stack([x, y], axis=1)


def make_moons(
    n_samples: int = 1000, noise: float = 0.1, key: jax.random.PRNGKey = None
):
    if key is None:
        key = random.PRNGKey(0)

    key1, key2 = random.split(key, 2)

    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Lune extérieure
    outer_angles = random.uniform(key1, (n_outer,), minval=0, maxval=jnp.pi)
    outer_x = jnp.cos(outer_angles)
    outer_y = jnp.sin(outer_angles)

    # Lune intérieure
    inner_angles = random.uniform(key2, (n_inner,), minval=0, maxval=jnp.pi)
    inner_x = 1 - jnp.cos(inner_angles)
    inner_y = 1 - jnp.sin(inner_angles) - 0.5

    x = jnp.concatenate([outer_x, inner_x])
    y = jnp.concatenate([outer_y, inner_y])

    noise_key1, noise_key2 = random.split(key, 2)
    x += random.normal(noise_key1, (n_samples,)) * noise
    y += random.normal(noise_key2, (n_samples,)) * noise

    return jnp.stack([x, y], axis=1)


def make_gaussian_mixture(n_samples: int = 1000, key: jax.random.PRNGKey = None):
    if key is None:
        key = random.PRNGKey(0)

    key1, key2 = random.split(key, 2)

    # Deux gaussiennes
    labels = random.bernoulli(key1, 0.5, (n_samples,))

    # Centre 1: (-1, -1), Centre 2: (1, 1)
    centers = jnp.array([[-1.5, -1.5], [1.5, 1.5]])

    samples = random.normal(key2, (n_samples, 2)) * 0.3
    samples = jnp.where(labels[:, None], samples + centers[1], samples + centers[0])

    return samples


# MLP corrigé avec choix d'activation
class MLP(eqx.Module):
    layers: list
    use_elu: bool

    def __init__(self, in_size, hidden_sizes, out_size, key, activation="tanh"):
        keys = random.split(key, len(hidden_sizes) + 1)
        sizes = [in_size] + hidden_sizes + [out_size]

        # Stocker un booléen simple
        self.use_elu = activation == "elu"

        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            if self.use_elu:
                x = jax.nn.elu(layer(x))
            else:
                x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)


# Couche RealNVP optimisée pour dim=2 uniquement
class RealNVPLayer(eqx.Module):
    net_s: MLP
    net_t: MLP

    def __init__(self, hidden_sizes, key):
        key1, key2 = random.split(key, 2)

        # Pour dim=2: x[0] conditionne la transformation de x[1]
        # Utilise tanh pour les flots normalisés
        self.net_s = MLP(1, hidden_sizes, 1, key1, activation="tanh")
        self.net_t = MLP(1, hidden_sizes, 1, key2, activation="tanh")

    def forward(self, x):
        x0, x1 = x[0], x[1]

        # x[0] reste inchangé, x[1] est transformé
        s = self.net_s(jnp.array([x0]))[0]
        t = self.net_t(jnp.array([x0]))[0]

        s = jnp.clip(s, -2.0, 2.0)

        z0 = x0
        z1 = x1 * jnp.exp(s) + t

        z = jnp.array([z0, z1])
        log_det = s

        return z, log_det

    def inverse(self, z):
        z0, z1 = z[0], z[1]

        s = self.net_s(jnp.array([z0]))[0]
        t = self.net_t(jnp.array([z0]))[0]

        s = jnp.clip(s, -2.0, 2.0)

        x0 = z0
        x1 = (z1 - t) * jnp.exp(-s)

        x = jnp.array([x0, x1])
        log_det = -s

        return x, log_det


# Couche RealNVP alternée
class RealNVPLayerFlipped(eqx.Module):
    net_s: MLP
    net_t: MLP

    def __init__(self, hidden_sizes, key):
        key1, key2 = random.split(key, 2)
        # Utilise tanh pour les flots normalisés
        self.net_s = MLP(1, hidden_sizes, 1, key1, activation="tanh")
        self.net_t = MLP(1, hidden_sizes, 1, key2, activation="tanh")

    def forward(self, x):
        x0, x1 = x[0], x[1]

        s = self.net_s(jnp.array([x1]))[0]
        t = self.net_t(jnp.array([x1]))[0]

        s = jnp.clip(s, -2.0, 2.0)

        z0 = x0 * jnp.exp(s) + t
        z1 = x1

        z = jnp.array([z0, z1])
        log_det = s

        return z, log_det

    def inverse(self, z):
        z0, z1 = z[0], z[1]

        s = self.net_s(jnp.array([z1]))[0]
        t = self.net_t(jnp.array([z1]))[0]

        s = jnp.clip(s, -2.0, 2.0)

        x0 = (z0 - t) * jnp.exp(-s)
        x1 = z1

        x = jnp.array([x0, x1])
        log_det = -s

        return x, log_det


# Flow simplifié pour dim=2
class NormalizingFlow(eqx.Module):
    layers: list

    def __init__(self, num_layers, hidden_sizes, key):
        keys = random.split(key, num_layers)
        self.layers = []

        for i, k in enumerate(keys):
            if i % 2 == 0:
                layer = RealNVPLayer(hidden_sizes, k)
            else:
                layer = RealNVPLayerFlipped(hidden_sizes, k)
            self.layers.append(layer)

    def forward(self, x):
        log_det_total = 0.0
        for layer in self.layers:
            x, log_det = layer.forward(x)
            log_det_total += log_det
        return x, log_det_total

    def log_prob(self, x):
        def single_log_prob(x_single):
            z, log_det = self.forward(x_single)
            log_prob_base = -0.5 * jnp.sum(z**2) - jnp.log(2 * jnp.pi)
            return log_prob_base + log_det

        if x.ndim == 2:
            return jax.vmap(single_log_prob)(x)
        else:
            return single_log_prob(x)

    def sample(self, num_samples, key):
        z = random.normal(key, (num_samples, 2))

        def single_sample(z_single):
            x = z_single
            for layer in reversed(self.layers):
                x, _ = layer.inverse(x)
            return x

        return jax.vmap(single_sample)(z)


# Flow Matching corrigé
class FlowMatching(eqx.Module):
    velocity_net: MLP

    def __init__(self, dim, hidden_sizes, key):
        # Réseau prenant (x, t) en entrée
        # Utilise ELU pour le flow matching
        self.velocity_net = MLP(dim + 1, hidden_sizes, dim, key, activation="elu")

    def velocity_field(self, x, t):
        # Assurer que t est un scalaire
        if jnp.ndim(t) == 0:
            t_expanded = jnp.array([t])
        else:
            t_expanded = jnp.array([t]) if jnp.ndim(t) == 0 else t

        xt = jnp.concatenate([x, t_expanded], axis=-1)
        return self.velocity_net(xt)


# Entraînement amélioré
def train_normalizing_flow(model, data, epochs=800, lr=2e-3, batch_size=64):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model)

    def loss_fn(model, batch):
        log_probs = model.log_prob(batch)
        return -jnp.mean(log_probs)

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses = []
    n_samples = data.shape[0]

    for epoch in range(epochs):
        key, subkey = random.split(random.PRNGKey(epoch))
        idx = random.randint(subkey, (batch_size,), 0, n_samples)
        batch = data[idx]

        model, opt_state, loss = step(model, opt_state, batch)
        losses.append(loss)

        if epoch % 200 == 0:
            print(f"Époque {epoch}, Perte NF: {loss:.4f}")

    return model, losses


# Entraînement Flow Matching corrigé
def train_flow_matching(model, data, epochs=1000, lr=1e-2, batch_size=256):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(model)

    key = random.PRNGKey(123)  # Clé locale pour l'entraînement

    def loss_fn(model, x1_batch, key):
        batch_size = x1_batch.shape[0]
        dim = x1_batch.shape[1]

        # Échantillonner source et temps
        x0_batch = random.normal(key, (batch_size, dim))
        key, subkey = random.split(key)
        t_batch = random.uniform(subkey, (batch_size,))

        def single_loss(x0, x1, t):
            # Chemin linéaire simple
            x_t = (1 - t) * x0 + t * x1
            v_true = x1 - x0
            v_pred = model.velocity_field(x_t, t)
            return jnp.sum((v_pred - v_true) ** 2)

        losses = jax.vmap(single_loss)(x0_batch, x1_batch, t_batch)
        return jnp.mean(losses)

    @eqx.filter_jit
    def step(model, opt_state, batch, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses = []
    n_samples = data.shape[0]

    for epoch in range(epochs):
        key, subkey = random.split(key)
        idx = random.randint(subkey, (batch_size,), 0, n_samples)
        batch = data[idx]

        key, subkey = random.split(key)
        model, opt_state, loss = step(model, opt_state, batch, subkey)
        losses.append(loss)

        if epoch % 250 == 0:
            print(f"  Époque {epoch}, Perte FM: {loss:.4f}")

    return model, losses


# Génération Flow Matching
def sample_flow_matching(model, num_samples, steps=100, key=None):
    if key is None:
        key = random.PRNGKey(0)

    # Point de départ
    x = random.normal(key, (num_samples, 2))
    dt = 1.0 / steps

    for step in range(steps):
        t = step * dt

        # Calcul vectorisé du champ de vitesse
        def get_velocity(x_single):
            return model.velocity_field(x_single, t)

        v = jax.vmap(get_velocity)(x)
        x = x + v * dt

    return x


# Fonction de visualisation séparée
def plot_results(results):
    """Afficher tous les résultats à la fin"""
    for (
        dataset_name,
        data,
        nf_model,
        nf_losses,
        nf_samples,
        fm_model,
        fm_losses,
        fm_samples,
    ) in results:
        # Visualisation améliorée - 2x3 grille
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(
            f"Comparaison Flots Normalisants vs Flow Matching - {dataset_name}",
            fontsize=18,
            fontweight="bold",
        )

        # Ligne 1: Données et échantillons
        # Données originales
        axes[0, 0].scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.7,
            s=30,
            c="blue",
            edgecolors="navy",
            linewidth=0.5,
        )
        axes[0, 0].set_title(
            f"Données Originales\n{dataset_name}", fontsize=14, fontweight="bold"
        )
        axes[0, 0].set_aspect("equal")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlabel("X", fontsize=12)
        axes[0, 0].set_ylabel("Y", fontsize=12)

        # Échantillons Flots Normalisants
        axes[0, 1].scatter(
            nf_samples[:, 0],
            nf_samples[:, 1],
            alpha=0.7,
            s=30,
            c="red",
            edgecolors="darkred",
            linewidth=0.5,
        )
        axes[0, 1].set_title(
            f"Flots Normalisants\nPerte finale: {nf_losses[-1]:.3f}",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 1].set_aspect("equal")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlabel("X", fontsize=12)
        axes[0, 1].set_ylabel("Y", fontsize=12)

        # Échantillons Flow Matching
        axes[0, 2].scatter(
            fm_samples[:, 0],
            fm_samples[:, 1],
            alpha=0.7,
            s=30,
            c="green",
            edgecolors="darkgreen",
            linewidth=0.5,
        )
        axes[0, 2].set_title(
            f"Flow Matching\nPerte finale: {fm_losses[-1]:.3f}",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 2].set_aspect("equal")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_xlabel("X", fontsize=12)
        axes[0, 2].set_ylabel("Y", fontsize=12)

        # Ligne 2: Courbes de perte et comparaison
        # Courbe de perte Flots Normalisants
        axes[1, 0].plot(
            nf_losses, "r-", linewidth=3, alpha=0.8, label="Flots Normalisants"
        )
        axes[1, 0].set_title(
            "Évolution Perte - Flots Normalisants", fontsize=14, fontweight="bold"
        )
        axes[1, 0].set_xlabel("Époque", fontsize=12)
        axes[1, 0].set_ylabel("Perte", fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_facecolor("#fdf2f2")

        # Courbe de perte Flow Matching
        axes[1, 1].plot(fm_losses, "g-", linewidth=3, alpha=0.8, label="Flow Matching")
        axes[1, 1].set_title(
            "Évolution Perte - Flow Matching", fontsize=14, fontweight="bold"
        )
        axes[1, 1].set_xlabel("Époque", fontsize=12)
        axes[1, 1].set_ylabel("Perte", fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_facecolor("#f2fdf2")

        # Comparaison superposée
        axes[1, 2].scatter(
            data[:, 0],
            data[:, 1],
            alpha=0.4,
            s=15,
            c="blue",
            label="Original",
            edgecolors="navy",
            linewidth=0.3,
        )
        axes[1, 2].scatter(
            nf_samples[:, 0],
            nf_samples[:, 1],
            alpha=0.6,
            s=15,
            c="red",
            label="Flots Norm.",
            edgecolors="darkred",
            linewidth=0.3,
        )
        axes[1, 2].scatter(
            fm_samples[:, 0],
            fm_samples[:, 1],
            alpha=0.6,
            s=15,
            c="green",
            label="Flow Match.",
            edgecolors="darkgreen",
            linewidth=0.3,
        )
        axes[1, 2].legend(fontsize=10)
        axes[1, 2].set_title("Comparaison Superposée", fontsize=14, fontweight="bold")
        axes[1, 2].set_aspect("equal")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_xlabel("X", fontsize=12)
        axes[1, 2].set_ylabel("Y", fontsize=12)

        plt.tight_layout()

        # Sauvegarde et affichage
        filename = (
            f"/Users/franck/Desktop/TP/comparison_complete_{dataset_name.lower()}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()
        plt.pause(5)  # Pause plus longue pour admirer !

        print(f"Graphique complet sauvé: {filename}")
        print(f"Résumé {dataset_name}: NF={nf_losses[-1]:.3f}, FM={fm_losses[-1]:.3f}")
        print("-" * 80)


# Test principal avec de beaux graphiques
def main():
    print("=== Test Flots et Flow Matching avec Visualisations Améliorées ===")

    key = random.PRNGKey(42)
    key1, key2, key3 = random.split(key, 3)

    datasets = {
        "Cercles": make_circles(1000, 0.05, key1),
        "Lunes": make_moons(1000, 0.1, key2),
        "Gaussiennes": make_gaussian_mixture(1000, key3),
    }

    # Stocker tous les résultats
    results = []

    for dataset_name, data in datasets.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        print("\n--- Flot Normalisant ---")
        key, subkey = random.split(key)
        nf_model = NormalizingFlow(8, [32, 32], subkey)

        nf_model, nf_losses = train_normalizing_flow(nf_model, data, epochs=4000)

        key, subkey = random.split(key)
        nf_samples = nf_model.sample(1000, subkey)

        print(f"Perte finale NF: {nf_losses[-1]:.4f}")

        print("\n--- Flow Matching ---")
        key, subkey = random.split(key)
        fm_model = FlowMatching(2, [40] * 4, subkey)

        fm_model, fm_losses = train_flow_matching(fm_model, data, epochs=10000)

        key, subkey = random.split(key)
        fm_samples = sample_flow_matching(fm_model, 1000, 100, subkey)

        print(f"Perte finale FM: {fm_losses[-1]:.4f}")

        # Stocker les résultats au lieu d'afficher immédiatement
        results.append(
            (
                dataset_name,
                data,
                nf_model,
                nf_losses,
                nf_samples,
                fm_model,
                fm_losses,
                fm_samples,
            )
        )

    # Afficher tous les plots à la fin
    print("\n=== Affichage de tous les graphiques ===")
    plot_results(results)


if __name__ == "__main__":
    main()
