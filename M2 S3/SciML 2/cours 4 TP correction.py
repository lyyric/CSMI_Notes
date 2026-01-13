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
model = train(model, sampler, lr, 1000)
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

# ------------------------------------------------
# Optimisation avec gradient naturel et linesearch
# ------------------------------------------------
from jax.flatten_util import ravel_pytree


# get the gradient of the loss with respect to parameters of the network
def grad_theta_loss(model, collocation, boundary):
    grad_pytree = eqx.filter_grad(
        lambda m: pinn_loss(m, collocation, boundary).squeeze()
    )(model)
    grad_vec, _ = ravel_pytree(grad_pytree)
    return grad_vec


# get the gradient of residual with respect to parameters of the network at a point x
def grad_theta_residual_point(model, x):
    grad_pytree = eqx.filter_grad(lambda m: residual_point(m, x).squeeze())(model)
    grad_vec, _ = ravel_pytree(grad_pytree)
    return grad_vec


# vmap for batched evaluations
jac_residual = jax.vmap(grad_theta_residual_point, in_axes=(None, 0))


# same for bc residual
def grad_theta_bc_residual_point(model, x):
    grad_pytree = eqx.filter_grad(lambda m: bc_residual_point(m, x).squeeze())(model)
    grad_vec, _ = ravel_pytree(grad_pytree)
    return grad_vec


# vmap for batched evaluations
jac_bc_residual = jax.vmap(grad_theta_bc_residual_point, in_axes=(None, 0))


# get the Gram matrices at collocation points
def get_Gram_matrix(model, collocation, boundary):

    jac = jac_residual(model, collocation)
    print("mmm ", jac.shape)
    ndof = jac.shape[1]
    n_colloc = jac.shape[0]
    regularization_matrix = 1.0e-6 * jnp.eye(ndof)
    jac = (1.0 / n_colloc) * jnp.einsum("bi,bk->ik", jac, jac) + regularization_matrix

    jac_bc = jac_bc_residual(model, boundary)
    n_colloc = jac_bc.shape[0]
    jac_bc = (1.0 / n_colloc) * jnp.einsum(
        "bi,bk->ik", jac_bc, jac_bc
    ) + regularization_matrix

    return jac + BC_WEIGHT * jac_bc


get_Gram_matrix_jit = eqx.filter_jit(get_Gram_matrix)


# try to minimize the loss along line cur_grad -eta*search_direction
def armijo_line_search(
    model,
    cur_loss: jnp.ndarray,
    cur_grad: jnp.ndarray,
    search_direction: jnp.ndarray,
    collocation,
    boundary,
):

    nbMaxSteps = 10
    alpha = 0.01
    beta = 0.5
    default_learning_rate = 0.02

    eta = jnp.array(1.0)

    # get current parameters theta of the approx space as jnp.ndarray
    # and the function to reconstruct a similar network with new parameters
    params, static = eqx.partition(model, eqx.is_array)
    theta, unravel = ravel_pytree(params)

    # create a new approx space nspace with params theta - eta*search_direction
    # and evaluate the loss of this approx space
    new_theta = unravel(theta - eta * search_direction)
    new_model = eqx.combine(new_theta, static)
    new_loss = pinn_loss(new_model, collocation, boundary)

    dL = jnp.dot(cur_grad, search_direction)
    nbsteps = 0

    while (new_loss > cur_loss - alpha * eta * dL) and (nbsteps < nbMaxSteps):
        # actualize eta and nbsteps
        eta *= beta
        nbsteps += 1
        # create a new approx space nspace with params theta - eta*search_direction
        # and evaluate the loss of this approx space
        new_theta = unravel(theta - eta * search_direction)
        new_model = eqx.combine(new_theta, static)
        new_loss = pinn_loss(new_model, collocation, boundary)

    if (new_loss > cur_loss) or (nbsteps >= nbMaxSteps):
        eta = jnp.array(default_learning_rate)
        new_theta = unravel(theta - eta * search_direction)
        new_model = eqx.combine(new_theta, static)
        new_loss = pinn_loss(new_model, collocation, boundary)

    return eta, new_loss, new_model


# jit versions
pinn_loss_jit = eqx.filter_jit(pinn_loss)
grad_theta_loss_jit = eqx.filter_jit(grad_theta_loss)
jac_residual_jit = eqx.filter_jit(jac_residual)
get_Gram_matrix_jit = eqx.filter_jit(get_Gram_matrix)


def one_step(model, key):
    subkey, key = jax.random.split(key, 2)
    collocation = sampler.collocation_points(subkey)
    boundary = sampler.boundary_points()
    cur_loss = pinn_loss_jit(model, collocation, boundary)
    cur_grad = grad_theta_loss_jit(model, collocation, boundary)
    cur_gram = get_Gram_matrix_jit(model, collocation, boundary)
    new_grad = jnp.linalg.lstsq(cur_gram, cur_grad)[0]
    _, new_loss, new_model = armijo_line_search(
        model, cur_loss, cur_grad, new_grad, collocation, boundary
    )

    return new_loss, new_model, key


key = jax.random.PRNGKey(0)
model = MLP1D(key)
x_collocation = sampler.collocation_points(key)
x_boundary = sampler.boundary_points()

grad = grad_theta_loss(model, x_collocation, x_boundary)
print("grad.shape: ", grad.shape)
start = ti.default_timer()
jac = jac_residual_jit(model, x_collocation)
end = ti.default_timer()
print("jac.shape: ", jac.shape)
print("time in first call :", end - start)

nmodel = jax.tree_util.tree_map(
    lambda x: x * 0.9 if isinstance(x, jnp.ndarray) else x, model
)

start = ti.default_timer()
njac = jac_residual_jit(nmodel, x_collocation)
end = ti.default_timer()
print("jac.shape: ", jac.shape)
print("time in second call :", end - start)
assert not jnp.all(njac == jac)

jac_bc = jac_bc_residual(model, x_boundary)
print("jac_bc.shape: ", jac_bc.shape)

start = ti.default_timer()
gram_matrix = get_Gram_matrix_jit(model, x_collocation, x_boundary)
end = ti.default_timer()
print("gram_matrix.shape: ", gram_matrix.shape)
print("time in first call :", end - start)

start = ti.default_timer()
ngram_matrix = get_Gram_matrix_jit(nmodel, x_collocation, x_boundary)
end = ti.default_timer()
print("gram_matrix.shape: ", gram_matrix.shape)
print("time in second call :", end - start)
assert not jnp.all(ngram_matrix == gram_matrix)

print("initial loss: ", pinn_loss(model, x_collocation, x_boundary))

new_model = model
start = ti.default_timer()
for i in range(30):
    new_loss, new_model, key = one_step(new_model, key)
    print("step %d, loss: " % i, new_loss)

end = ti.default_timer()
print(f"Training time: {end - start:.2f} seconds")
# -------------------------------
# Évaluation
# -------------------------------
test_key = jax.random.PRNGKey(2)
x_test = sampler.collocation_points(test_key)
u_pred = jax.vmap(lambda x: new_model(x))(x_test)  # (1000,1,1)
u_pred = u_pred[:, 0]  # aplatisse en (1000,)
plt.scatter(x_test[:, 0], u_pred, label="PINN with ENG")
plt.scatter(x_test[:, 0], jnp.sin(2.0 * jnp.pi * x_test[:, 0]), label="Exact")
plt.legend()
plt.show()
