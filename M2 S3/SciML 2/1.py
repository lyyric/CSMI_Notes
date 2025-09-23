import jax
import jax.numpy as jnp
import equinox as eqx
from jax import tree_util
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
import timeit as ti

jax.config.update("jax_enable_x64", True)


# -------------------------------
# MLP
# -------------------------------
class MLP1D(eqx.Module):
    layers: tuple

    def __init__(self, key, in_dim=1, hidden_dims=(16, 32, 16), out_dim=1):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = (in_dim,) + hidden_dims + (out_dim,)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))
            layers.append(jax.nn.tanh)
        layers.append(eqx.nn.Linear(dims[-2], dims[-1], key=keys[-1]))
        self.layers = tuple(layers)

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y


# -------------------------------
# Sampler
# -------------------------------
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
        left = jnp.full((self.n_boundary, 1), self.x_min)
        right = jnp.full((self.n_boundary, 1), self.x_max)
        return jnp.vstack([left, right])


# -------------------------------
# PDE: -ε u'' + u' = f
# -------------------------------
EPS = 0.3
BC_WEIGHT = 50.0


def f(x):
    return (4.0 * jnp.pi**2 * EPS) * jnp.sin(2.0 * jnp.pi * x) + 2.0 * jnp.pi * jnp.cos(
        2.0 * jnp.pi * x
    )


def residual_point(model, x):
    u = lambda xi: model(xi)[0]
    u_x = jax.grad(u)(x)[0]
    u_xx = jax.hessian(u)(x)[0, 0]
    return -EPS * u_xx + u_x - f(x)[0]


def pinn_loss_point(model, x):
    return residual_point(model, x) ** 2


def bc_residual_point(model, x):
    u = lambda xi: model(xi)[0]
    return u(x)


def boundary_loss_point(model, x):
    return bc_residual_point(model, x) ** 2


def pinn_loss(model, collocation, boundary):
    loss_pde = jnp.mean(jax.vmap(lambda x: pinn_loss_point(model, x))(collocation))
    loss_bc = jnp.mean(jax.vmap(lambda x: boundary_loss_point(model, x))(boundary))
    return loss_pde + BC_WEIGHT * loss_bc


# -------------------------------
# 残差向量 + 计算 M
# -------------------------------
def residuals_vector(model, collocation, boundary):
    r_pde = jax.vmap(lambda x: residual_point(model, x))(collocation)
    r_bc = jax.vmap(lambda x: bc_residual_point(model, x))(boundary)
    return jnp.concatenate([r_pde, jnp.sqrt(BC_WEIGHT) * r_bc], axis=0)


def compute_M_matrix(model, sampler, key, n_c_small=64, n_b_small=32):
    # 小批采样，避免矩阵过大
    key_c, _ = jax.random.split(key)
    x_c = jax.random.uniform(key_c, (n_c_small, 1),
                             minval=sampler.x_min, maxval=sampler.x_max)
    nb2 = n_b_small // 2
    x_b = jnp.vstack([
        jnp.full((nb2, 1), sampler.x_min),
        jnp.full((n_b_small - nb2, 1), sampler.x_max)
    ])

    # 扁平化参数
    theta0, unravel = ravel_pytree(model)

    def r_flat(theta_vec):
        m = unravel(theta_vec)
        return residuals_vector(m, x_c, x_b)  # (Nres,)

    # Jacobian Jr: (Nres, Nparams)
    Jr = jax.jacobian(r_flat)(theta0)
    M = Jr.T @ Jr
    return M, Jr


# -------------------------------
# Training (SGD)
# -------------------------------
def train_step(model, sampler, lr, key):
    x_collocation = sampler.collocation_points(key)
    x_boundary = sampler.boundary_points()
    loss, grads = eqx.filter_value_and_grad(pinn_loss)(model, x_collocation, x_boundary)
    updates = tree_util.tree_map(lambda g: -lr * g, grads)
    model = eqx.apply_updates(model, updates)
    return model, loss


def train(model, sampler, lr, steps):
    key = jax.random.PRNGKey(1)
    train_step_jit = eqx.filter_jit(train_step)
    for step in range(steps):
        key, subkey = jax.random.split(key)
        model, loss = train_step_jit(model, sampler, lr, subkey)
        if step % 500 == 0:
            print(f"Step {step:5d} | loss={float(loss):.6e}")
    return model


# -------------------------------
# Main
# -------------------------------
print(" //// begining training //// ")
start = ti.default_timer()
lr = 2.8e-4
model_key = jax.random.PRNGKey(0)
model = MLP1D(model_key)
sampler = Sampler1D()
model = train(model, sampler, lr, 3000)
end = ti.default_timer()
print(f"Training time: {end - start:.2f} seconds")

# -------------------------------
# Compute M after training
# -------------------------------
M_key = jax.random.PRNGKey(123)
M, Jr = compute_M_matrix(model, sampler, M_key, n_c_small=32, n_b_small=16)
print("M shape:", M.shape)
print("rank(Jr):", jnp.linalg.matrix_rank(Jr))
print("eigvals(M) min/max:", float(jnp.min(jnp.linalg.eigvalsh(M))),
      "/", float(jnp.max(jnp.linalg.eigvalsh(M))))

# -------------------------------
# Evaluation
# -------------------------------
test_key = jax.random.PRNGKey(2)
x_test = sampler.collocation_points(test_key)
u_pred = jax.vmap(lambda x: model(x))(x_test)[:, 0]
u_true = jnp.sin(2.0 * jnp.pi * x_test[:, 0])

plt.scatter(x_test[:, 0], u_pred, s=6, label="PINN")
plt.scatter(x_test[:, 0], u_true, s=6, label="Exact")
plt.legend()
plt.show()
