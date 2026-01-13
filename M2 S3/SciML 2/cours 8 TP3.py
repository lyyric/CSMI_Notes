import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph


# ---------- 1. Échantillonnage sur la sphère ----------
def sample_sphere_density_with_pdf(
    n_points=2000,
    alpha=0.7,
    theta0=0.0,
    phi0=0.0,
    sigma_theta=0.3,
    sigma_phi=0.3,
    seed=None,
):
    rng = np.random.default_rng(seed)
    n_uniform = int(alpha * n_points)
    n_conc = n_points - n_uniform

    # Points uniformes
    u, v = rng.uniform(0, 1, n_uniform), rng.uniform(0, 1, n_uniform)
    theta_u = 2 * np.pi * u
    phi_u = np.arccos(2 * v - 1)
    x_u = np.sin(phi_u) * np.cos(theta_u)
    y_u = np.sin(phi_u) * np.sin(theta_u)
    z_u = np.cos(phi_u)
    pts_u = np.stack([x_u, y_u, z_u], axis=1)

    # Points concentrés
    theta_c = rng.normal(loc=theta0, scale=sigma_theta, size=n_conc)
    phi_c = rng.normal(loc=phi0, scale=sigma_phi, size=n_conc)
    phi_c = np.clip(phi_c, 0, np.pi)
    theta_c = np.mod(theta_c, 2 * np.pi)
    x_c = np.sin(phi_c) * np.cos(theta_c)
    y_c = np.sin(phi_c) * np.sin(theta_c)
    z_c = np.cos(phi_c)
    pts_c = np.stack([x_c, y_c, z_c], axis=1)

    # Mélange
    points = np.vstack([pts_u, pts_c])
    normals = points.copy()

    # Densité PDF
    p_uniform = 1.0 / (4 * np.pi)
    norm_theta = 1.0 / (np.sqrt(2 * np.pi) * sigma_theta)
    norm_phi = 1.0 / (np.sqrt(2 * np.pi) * sigma_phi)
    phi_angles = np.arccos(np.clip(points[:, 2], -1, 1))
    theta_angles = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * np.pi)
    p_gauss_theta = norm_theta * np.exp(
        -0.5 * ((theta_angles - theta0) / sigma_theta) ** 2
    )
    p_gauss_phi = norm_phi * np.exp(-0.5 * ((phi_angles - phi0) / sigma_phi) ** 2)
    p_gauss = p_gauss_theta * p_gauss_phi
    p = alpha * p_uniform + (1 - alpha) * p_gauss

    return points, normals, p


# ---------- 2. Graphe k-NN et Laplacien ----------
def compute_graph_laplacian(points, k=8):
    N = points.shape[0]
    A = kneighbors_graph(points, k, mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)  # symétriser
    L = csgraph.laplacian(A, normed=False).toarray()  # shape (N,N)
    return L, np.array(A.toarray())


# ---------- 3. Génération des solutions complexes ----------
def generate_complex_sphere_solutions(
    points, L, theta, phi, Lmax=3, n_gauss=5, sigma=0.5, seed=0
):
    rng = np.random.default_rng(seed)
    N = points.shape[0]

    theta0s = rng.uniform(0, 2 * np.pi, n_gauss)
    phi0s = rng.uniform(0, np.pi, n_gauss)
    x0 = np.sin(phi0s) * np.cos(theta0s)
    y0 = np.sin(phi0s) * np.sin(theta0s)
    z0 = np.cos(phi0s)
    centers = np.stack([x0, y0, z0], axis=1)
    U_list = []

    for l in range(1, Lmax + 1):
        for m in range(-l, l + 1):
            Ylm = sph_harm(0, l, theta, phi).real
            dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
            gauss = np.exp(-0.5 * (dist / sigma) ** 2)
            for i in range(n_gauss):
                u = (Ylm + 0.3 * gauss[:, i]) * 0.5
                U_list.append(u)

    U_complex = jnp.stack(U_list, axis=1)  # (N, n_modes)
    print("U_complex.shape =", U_complex.shape)
    print("Calcul de F_complex...", L.shape)
    F_complex = -(L @ U_complex)  # (N, n_modes)
    return F_complex, U_complex


# ---------- 4. GraphResNet pour test ----------
class GraphSAGELayer(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray
    A: jnp.ndarray = eqx.static_field()
    deg: jnp.ndarray = eqx.static_field()

    def __init__(self, in_dim, out_dim, A, key):
        k1, _ = random.split(key)
        self.W = random.normal(k1, (2 * in_dim, out_dim)) / jnp.sqrt(in_dim)
        self.b = jnp.zeros(out_dim)
        self.A = A
        self.deg = jnp.maximum(self.A.sum(1, keepdims=True), 1e-8)

    def __call__(self, X):
        neigh = self.A @ X / self.deg
        H = jnp.concatenate([X, neigh], axis=1)
        return jax.nn.gelu(H @ self.W + self.b)


class ChebConvLayer(eqx.Module):
    W: jnp.ndarray
    b: jnp.ndarray
    K: int  # ordre du Chebyshev
    L_scale: jnp.ndarray = eqx.static_field()
    λ_max: float = eqx.static_field()

    def __init__(self, in_dim, out_dim, K, L, key):
        k1, _ = random.split(key)
        self.W = random.normal(k1, (K * in_dim, out_dim)) / jnp.sqrt(in_dim)
        self.b = jnp.zeros(out_dim)
        self.K = K
        self.λ_max = jnp.max(jnp.linalg.eigvalsh(L))
        self.L_scale = (2.0 * L / self.λ_max) - jnp.eye(L.shape[0])

    def __call__(self, X):
        """L est la matrice Laplacienne normalisée"""
        Tx = [X]
        if self.K > 1:
            Tx.append(self.L_scale @ X)
        for k in range(2, self.K):
            Tx_k = 2 * (self.L_scale @ Tx[-1]) - Tx[-2]
            Tx.append(Tx_k)
        H = jnp.concatenate(Tx, axis=1)
        return jax.nn.tanh(H @ self.W + self.b)


# ------------------------------
# Graph Residual Block
# ------------------------------
class GraphResBlock(eqx.Module):
    layer: eqx.Module

    def __init__(self, in_dim, out_dim, key, matrice, layer_type="sage", K=4):
        if layer_type == "sage":
            self.layer = GraphSAGELayer(in_dim, out_dim, matrice, key)
        elif layer_type == "cheb":
            self.layer = ChebConvLayer(in_dim, out_dim, K, matrice, key)
        else:
            raise ValueError(f"Unknown layer_type {layer_type}")

    def __call__(self, X):
        Y = self.layer(X)
        if Y.shape == X.shape:
            return X + Y  # résidu si mêmes dimensions
        return Y


# ------------------------------
# Graph Residual Network
# ------------------------------
class GraphResNet(eqx.Module):
    blocks: list
    readout: jnp.ndarray

    def __init__(
        self,
        key,
        in_dim,
        hidden_dim,
        out_dim,
        n_blocks,
        matrice,
        layer_type="sage",
        K=3,
    ):
        keys = random.split(key, n_blocks + 1)
        self.blocks = [
            GraphResBlock(
                in_dim=hidden_dim if i > 0 else in_dim,
                out_dim=hidden_dim,
                key=keys[i],
                matrice=matrice,
                layer_type=layer_type,
                K=K,
            )
            for i in range(n_blocks)
        ]
        self.readout = random.normal(keys[-1], (hidden_dim, out_dim)) / jnp.sqrt(
            hidden_dim
        )

    def __call__(self, X):
        H = X
        for block in self.blocks:
            H = block(H)
        return H @ self.readout


def plot_prediction(
    points, u_true, u_pred, source, title="Comparaison u vs prédiction"
):
    fig = plt.figure(figsize=(12, 5))

    # Vérité
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    sc1 = ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=u_true[:, 0],
        cmap="coolwarm",
        s=10,
    )
    fig.colorbar(sc1, ax=ax1)
    ax1.set_title("Solution réelle u")

    # Prédiction
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    sc2 = ax2.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=u_pred[:, 0],
        cmap="coolwarm",
        s=10,
    )
    fig.colorbar(sc2, ax=ax2)
    ax2.set_title("Prédiction du réseau")

    # Prédiction
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    sc3 = ax3.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=source[:, 0],
        cmap="coolwarm",
        s=10,
    )
    fig.colorbar(sc3, ax=ax3)
    ax3.set_title("Source")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# ---------- 5. Main ----------
if __name__ == "__main__":
    layer_type = "sage"
    # points
    points, normals, pdf = sample_sphere_density_with_pdf(
        n_points=3000, alpha=1.0, theta0=np.pi / 4, phi0=np.pi / 4, seed=42
    )
    phi = np.arctan2(points[:, 1], points[:, 0])
    theta = np.arccos(points[:, 2])

    # Laplacien
    L, A = compute_graph_laplacian(points, k=8)
    print("L.shape =", L.shape)

    # solutions complexes
    F, U = generate_complex_sphere_solutions(points, L, theta, phi, Lmax=3, n_gauss=10)

    U_batch = U.T[..., None]  # shape (75, 2000, 1)
    F_batch = F.T[..., None]
    print("F.shape =", F_batch.shape, "U.shape =", U_batch.shape)
    # plot exemple
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=F_batch[10, :, 0],
        cmap="coolwarm",
        s=10,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Valeurs de U")
    ax.set_title("F: 1er harmonique sphérique")
    plt.show()

    # modèle GraphResNet pour test
    key = random.PRNGKey(0)

    if layer_type == "sage":
        W = A
    else:
        W = L

    model = GraphResNet(
        in_dim=1,
        hidden_dim=32,
        out_dim=1,
        n_blocks=6,
        matrice=W,
        layer_type=layer_type,
        key=key,
    )

    Upred = jax.vmap(model)(F_batch)
    print("Upred.shape =", Upred.shape)

    # optimiser avec Optax
    opt = optax.adam(7e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, f, u):
        pred = jax.vmap(model)(f)
        return jnp.mean((pred - u) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, f, u):
        # tirer des indices aléatoires pour le mini-batch
        N = F_batch.shape[0]
        idx = jax.random.choice(key, N, shape=(50,), replace=False)

        F_mini = f[idx]
        u_mini = u[idx]

        # calcul de la loss et des gradients sur le mini-batch
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, F_mini, u_mini)

        # mise à jour du modèle
        updates, opt_state = opt.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    for epoch in range(120):  # exemple rapide
        model, opt_state, l = step(model, opt_state, F_batch, U_batch)
        print(f"Epoch {epoch}: loss = {l:.3e}")

    Upred_final = model(F_batch[0])
    plot_prediction(points, U_batch[0], Upred_final, F_batch[0])
