import jax
import jax.numpy as jnp

# from flax import linen as nn
import equinox as eqx
import functools

jax.config.update("jax_enable_x64", True)


class MLP1D(eqx.Module):
    layers: list

    def __init__(self, key, in_dim=1, hidden_dims=(16, 32, 16), out_dim=1):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = (in_dim,) + hidden_dims + (out_dim,)
        self.layers = []
        for i in range(len(dims) - 2):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))
            self.layers.append(jax.nn.tanh)
        # Dernière couche linéaire sans activation
        self.layers.append(eqx.nn.Linear(dims[-2], dims[-1], key=keys[-1]))
        self.layers = tuple(self.layers)  # pour que ce soit immuable

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Sampler1D:
    def __init__(self, x_min=0.0, x_max=1.0, n_collocation=1000, n_boundary=2000):
        self.x_min = x_min
        self.x_max = x_max
        self.n_collocation = n_collocation
        self.n_boundary = n_boundary

    def collocation_points(self, key):
        return jax.random.uniform(
            key, (self.n_collocation, 1), minval=self.x_min, maxval=self.x_max
        )

    def boundary_points(self):
        # Points sur le bord gauche
        left = jnp.linspace(self.x_min, self.x_min, self.n_boundary).reshape(-1, 1)
        # Points sur le bord droit
        right = jnp.linspace(self.x_max, self.x_max, self.n_boundary).reshape(-1, 1)
        # Concaténation
        return jnp.vstack([left, right])


def f(x):
    return 4 * jnp.pi**2.0 * jnp.sin(2 * jnp.pi * x)


# residual:
def residual_point(model, x):
    u = lambda xi: model(xi)[0]  # scalar->scalar
    u_xx = jax.hessian(u)(x)[:, 0]
    return u_xx + f(x)  # -u_xx = f(x)


# Loss pour un point collocation
def pinn_loss_point(model, x):
    return residual_point(model, x) ** 2


def bc_residual_point(model, x):
    u = lambda xi: model(xi)[0]
    return u(x)  # dirichlet condition u(x) = 0


# Loss pour un point boundary
def boundary_loss_point(model, x):
    return bc_residual_point(model, x) ** 2


BC_WEIGHT = 50.0


# Loss totale vectorisée
def pinn_loss(model, collocation, boundary):
    # vmap sur collocation
    loss_pde = jnp.mean(jax.vmap(lambda x: pinn_loss_point(model, x))(collocation))
    # vmap sur boundary
    loss_bc = jnp.mean(jax.vmap(lambda x: boundary_loss_point(model, x))(boundary))
    return loss_pde + BC_WEIGHT * loss_bc


# JIT

model_key = jax.random.PRNGKey(0)


# Modèle MLP
model = MLP1D(model_key)

# Sampler
sampler = Sampler1D()


# -------------------------------
# Optimisation simple
# -------------------------------
import matplotlib.pyplot as plt
import timeit as ti


def train_step(model, sampler, lr, key):
    x_collocation = sampler.collocation_points(key)
    x_boundary = sampler.boundary_points()
    loss, grads = eqx.filter_value_and_grad(pinn_loss)(model, x_collocation, x_boundary)
    updates = jax.tree.map(lambda g: -lr * g, grads)
    model = eqx.apply_updates(model, updates)
    return model, loss


def train(model, sampler, lr, steps):
    sample_key = jax.random.PRNGKey(1)
    train_step_jit = eqx.filter_jit(
        lambda model: train_step(model, sampler, lr, sample_key)
    )
    for step in range(steps):
        model, loss = train_step_jit(model)
        if step % 500 == 0:
            print(f"Step {step:5d} | loss={float(loss):.6e}")
    return model


print(" //// begining training //// ")
start = ti.default_timer()
lr = 2.8e-4
model = train(model, sampler, lr, 10_000)
# model = train(model, sampler, lr, 2)
end = ti.default_timer()
print(f"Training time: {end - start:.2f} seconds")
# -------------------------------
# Évaluation
# -------------------------------
test_key = jax.random.PRNGKey(2)
x_test = sampler.collocation_points(test_key)
u_pred = jax.vmap(lambda x: model(x))(x_test)  # (1000,1,1)
u_pred = u_pred[:, 0]  # aplatisse en (1000,)
plt.scatter(x_test[:, 0], u_pred, label="PINN")
plt.scatter(x_test[:, 0], jnp.sin(2.0 * jnp.pi * x_test[:, 0]), label="Exact")
plt.legend()
plt.show()
