import jax
import jax.numpy as jnp
import timeit as it
from jax.scipy.sparse.linalg import cg
import equinox as eqx
from typing import Sequence
import optax
from functools import partial
from equinox import static_field
from typing import Callable, Sequence


#################### functions to create data ####################
# ---- 1D Laplacien FD ----
def build_matrix(N, L=1.0):
    """Matrice Laplacien 1D FD Dirichlet"""
    dx = L / (N + 1)
    diag = -2.0 * jnp.ones(N)
    off = jnp.ones(N - 1)
    A = jnp.diag(diag) + jnp.diag(off, 1) + jnp.diag(off, -1)
    return -A / dx**2, dx


# ---- Solveur ----
def solve(mu, sigma, w):
    """Résout -u'' = exp(-(x-mu)^2/(2 sigma^2)), avec BC Dirichlet"""
    f = jnp.exp(-0.5 * ((x - mu) / sigma) ** 2) * jnp.sqrt(2 * w * jnp.pi)

    # CG retourne (solution, info)
    u, info = cg(A, f, tol=1e-7, maxiter=500)
    return u, f


solve_jit = jax.jit(jax.vmap(jax.jit(solve)))


##################################### Neural models ##############################


class ConvBlock1D(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, in_channels, out_channels, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, key=k1
        )
        self.conv2 = eqx.nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, key=k2
        )

    def __call__(self, x):
        x = jax.nn.softplus(self.conv1(x))
        x = jax.nn.softplus(self.conv2(x))
        return x


class ConvBlock1D(eqx.Module):
    conv1: eqx.nn.Conv1d
    conv2: eqx.nn.Conv1d

    def __init__(self, in_channels, out_channels, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, key=k1
        )
        self.conv2 = eqx.nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, key=k2
        )

    def __call__(self, x):
        x = jax.nn.gelu(self.conv1(x))
        x = jax.nn.gelu(self.conv2(x))
        return x


class UNet1D(eqx.Module):
    down_blocks: list
    pools: list = static_field()
    up_convs: list
    up_blocks: list
    bottleneck: ConvBlock1D
    final_conv: eqx.nn.Conv1d

    def __init__(self, in_channels, out_channels, features, key):
        n_keys = 2 * len(features) + 3
        keys = jax.random.split(key, n_keys)

        # encodeur
        self.down_blocks = [ConvBlock1D(in_channels, features[0], keys[0])]
        self.pools = [eqx.nn.MaxPool1d(2, 2)]
        for i in range(1, len(features)):
            self.down_blocks.append(ConvBlock1D(features[i - 1], features[i], keys[i]))
            self.pools.append(eqx.nn.MaxPool1d(2, 2))

        # bottleneck
        self.bottleneck = ConvBlock1D(
            features[-1], features[-1] * 2, keys[len(features)]
        )

        # décodeur
        self.up_convs = []
        self.up_blocks = []
        in_ch = features[-1] * 2
        for i, feat in enumerate(reversed(features)):
            k1, k2 = jax.random.split(keys[len(features) + 1 + i], 2)
            self.up_convs.append(
                eqx.nn.ConvTranspose1d(in_ch, feat, kernel_size=2, stride=2, key=k1)
            )
            self.up_blocks.append(ConvBlock1D(feat * 2, feat, k2))
            in_ch = feat

        # final conv
        print(out_channels, features[0])
        self.final_conv = eqx.nn.Conv1d(
            features[0], out_channels, kernel_size=3, stride=1, padding=1, key=keys[-1]
        )

    def __call__(self, x):
        x = jnp.swapaxes(x, -2, -1)
        skips = []
        for down, pool in zip(self.down_blocks, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)  #

        x = self.bottleneck(x)

        for upconv, upblock, skip in zip(
            self.up_convs, self.up_blocks, reversed(skips)
        ):
            x = upconv(x)
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                x = jnp.pad(x, ((0, 0), (0, diff)))
            x = jnp.concatenate([skip, x], axis=0)
            x = upblock(x)

        x = self.final_conv(x)
        x = jnp.swapaxes(x, -2, -1)
        return x


class Kernel(eqx.Module):
    layers: tuple  # tuple de Linear, non liste Python
    activation: Callable = eqx.static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int] = [32],
        activation: Callable = jax.nn.gelu,
        key: jax.Array = jax.random.PRNGKey(0),
    ):
        sizes = [2 * in_size, *hidden_sizes, out_size]
        keys = jax.random.split(key, num=len(sizes) - 1)

        # Créer un tuple de layers dès l'init
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        )
        print(self.layers)

        self.activation = activation

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.stack([x, y], axis=0).squeeze()  # forme (n, 2)
        for layer in self.layers[:-1]:
            inputs = self.activation(layer(inputs))
        inputs = self.layers[-1](inputs)
        return inputs


class Homogeneous(eqx.Module):
    layers: tuple  # tuple de Linear, non liste Python
    activation: Callable = eqx.static_field()

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_sizes: Sequence[int] = [32],
        activation: Callable = jax.nn.gelu,
        key: jax.Array = jax.random.PRNGKey(0),
    ):
        sizes = [2 * in_size, *hidden_sizes, out_size]
        keys = jax.random.split(key, num=len(sizes) - 1)

        # Créer un tuple de layers dès l'init
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        )

        self.activation = activation

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class IntegralOperator(eqx.Module):
    """Opérateur intégral à noyau neuronal k_θ(x, y)."""

    kernel: eqx.Module
    ys: jax.Array
    dy: float
    hom: eqx.Module

    def __init__(self, kernel, ys, dy, hom):
        self.kernel = kernel
        self.ys = ys
        self.dy = dy
        self.hom = hom

    def compute_local_x(self, x, f):
        """Évalue u(x) = ∑_j k_θ(x, y_j) f(y_j) dy pour un seul x."""
        # f : (Ny,)
        f = f.squeeze()

        # k_vals : (Ny, 1)
        k_vals = jax.vmap(self.kernel, in_axes=(None, 0))(x, self.ys)

        # produit scalaire sur Ny
        u_x = jnp.sum(k_vals.squeeze(-1) * f * self.dy) + self.hom(x)

        # renvoie un tableau (1,) pour cohérence
        return u_x[None]

    @eqx.filter_jit
    def __call__(self, f):

        return jax.vmap(self.compute_local_x, in_axes=(0, None))(self.ys, f)


#####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 100 couples (mu, sigma)
key = jax.random.PRNGKey(0)
mus = jax.random.uniform(key, (200,), minval=0.2, maxval=0.8)
sigmas = 0.05 + 0.15 * jax.random.uniform(key, (200,))
ws = jax.random.uniform(key, (200,), minval=0.5, maxval=2.2)

# Fixer le maillage
N = 256
A, dx = build_matrix(N)
x = jnp.linspace(dx, 1.0 - dx, N)


## data
u, f = solve_jit(mus, sigmas, ws)
## model
model = UNet1D(1, 1, (4, 8, 16), key)
model2 = IntegralOperator(
    kernel=Kernel(in_size=1, out_size=1, hidden_sizes=[20, 20, 20], key=key),
    ys=x,
    dy=jnp.ones(N) / N,
    hom=Homogeneous(in_size=1, out_size=1, hidden_sizes=[20, 20, 20], key=key),
)

u_data = u[:, :, None]
f_data = f[:, :, None]


def mse_loss(model, f, u):
    preds = jax.vmap(model)(f)  # vectorisation sur le batch
    return jnp.mean((preds - u) ** 2)


# ----------------------------
# Training step Unet
# ----------------------------
@jax.jit
def train_step(model, opt_state, f_train, u_train):
    indices = jax.random.choice(key, N, (50,), replace=False)
    f_batch = f_train[indices]
    u_batch = u_train[indices]
    loss, grads = jax.value_and_grad(mse_loss)(model, f_batch, u_batch)
    updates, opt_state = optimizer.update(grads, opt_state, params=model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


################## optimiser ##################


decay_rate = 0.99  # 1% de décroissance à chaque step
scheduler = optax.exponential_decay(
    init_value=3e-3, transition_steps=200, decay_rate=decay_rate
)
scheduler2 = optax.exponential_decay(
    init_value=7e-3, transition_steps=1000, decay_rate=decay_rate
)

optimizer = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0),  # gradient descent
)
optimizer2 = optax.chain(
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler2),
    optax.scale(-1.0),  # gradient descent
)
opt_state1 = optimizer.init(model)
opt_state2 = optimizer2.init(model2)

################## Training ##################

n_epochs = 3000
for epoch in range(n_epochs):
    model, opt_state1, loss = train_step(model, opt_state1, f_data, u_data)
    model2, opt_state2, loss2 = train_step(model2, opt_state2, f_data, u_data)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss = {loss:.6f}")
        print(f"Epoch {epoch}, Loss2 = {loss2:.6f}")


################# Plot results ##################

import matplotlib.pyplot as plt

# Clé aléatoire pour choisir des exemples
key_plot = jax.random.PRNGKey(123)

# On choisit 2 indices aléatoires
indices = jax.random.choice(key_plot, u_data.shape[0], (2,), replace=False)
u_batch = u_data[indices]  # solution exacte
f_batch = f_data[indices]  # source

# Passer par le réseau
u_pred = jax.vmap(model)(f_batch)  # (2, 1, L)
u_pred2 = jax.vmap(model2)(f_batch)  # (2, 1, L)

for i in range(2):
    plt.figure(figsize=(10, 4))
    plt.plot(x, u_batch[i, :, 0], label="u exact (FD)", linewidth=2)
    plt.plot(
        x,
        u_pred[i, :, 0],
        "--",
        label="u prédiction (UNet)",
        linewidth=2,
    )
    plt.plot(
        x,
        u_pred2[i, :, 0],
        "--",
        label="u prédiction (Integral)",
        linewidth=2,
    )
    plt.plot(
        x,
        u_batch[i, :, 0] - u_pred[i, :, 0],
        "r",
        label="différence Unet ",
        alpha=0.6,
    )
    plt.plot(
        x,
        u_batch[i, :, 0] - u_pred2[i, :, 0],
        "r",
        label="différence Integral ",
        alpha=0.6,
    )
    plt.title(f"Exemple {i}")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend()
    plt.grid(True)
    plt.show()
