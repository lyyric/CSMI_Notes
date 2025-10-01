import jax
import jax.numpy as jnp
import equinox as eqx
import optax


# -------------------------------
# 1. Réseau Equinox
# -------------------------------
class MLP(eqx.Module):
    layers: list

    def __init__(self, in_size, hidden_sizes, out_size, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        self.layers = []
        sizes = [in_size] + hidden_sizes + [out_size]
        for i in range(len(sizes) - 1):
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i]))

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# -------------------------------
# 2. Projection
# -------------------------------
def project(network, x_quad, u_target, n_iters=400, lr=3e-3):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(network)

    def loss_fn(net):
        u_pred = jax.vmap(net)(x_quad)
        return jnp.mean((u_pred - u_target) ** 2)

    @jax.jit
    def step(net, opt_state):
        grads = jax.grad(loss_fn)(net)
        updates, opt_state = optimizer.update(grads, opt_state)
        net = eqx.apply_updates(net, updates)
        return net, opt_state

    for _ in range(n_iters):
        network, opt_state = step(network, opt_state)
    return network


# -------------------------------
# 3. Euler explicite pour estimation temporaire
# -------------------------------
def euler_predict(network, x_quad, dt):
    def u_fn(x):
        return network(x)

    grad2_fn = lambda x: jax.hessian(u_fn)(x)[0, 0]
    d2u = grad2_fn(x_quad)
    u_vals = network(x_quad)
    u_next_est = u_vals + dt * d2u
    return u_next_est


# -------------------------------
# 4. Boucle temporelle avec stockage des réseaux
# -------------------------------
key = jax.random.PRNGKey(0)
key = jax.random.PRNGKey(0)
N_quad = 2000
x_quad = jax.random.uniform(key, (N_quad, 1))

# Condition initiale
u0_vals = jnp.exp(-200 * (x_quad - 0.5) ** 2)
mlp = MLP(1, [32, 32], 1, key)
mlp = project(mlp, x_quad, u0_vals)


dt = 0.0005
T = 0.01
n_steps = int(T / dt)

networks_over_time = [mlp]  # stockage du réseau initial

for n in range(n_steps):
    key = jax.random.PRNGKey(0)
    N_quad = 2000
    x_quad = jax.random.uniform(key, (N_quad, 1))
    # 1. Estimation Euler au début du pas de temps
    u_next_est = jax.vmap(euler_predict, in_axes=(None, 0, None))(mlp, x_quad, dt)
    # 2. Projection sur le réseau pour obtenir les nouveaux paramètres
    mlp = project(mlp, x_quad, u_next_est)
    # 3. Stockage du réseau à ce pas de temps
    networks_over_time.append(mlp)
    print(f"Step {n+1}/{n_steps} completed.")

# networks_over_time contient le réseau à chaque pas de temps

import matplotlib.pyplot as plt

# Choix des pas de temps à afficher
time_indices = [
    0,
    int(0.003 / dt),
    int(0.006 / dt),
    int(0.01 / dt),
]

x_quad = jnp.linspace(0, 1, 100).reshape(-1, 1)
plt.figure(figsize=(8, 5))
for n in time_indices:
    u_vals = jax.vmap(networks_over_time[n])(x_quad)
    plt.plot(x_quad.flatten(), u_vals.flatten(), label=f"t={n*dt:.2f}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Évolution de la solution de la chaleur 1D")
plt.legend()
plt.grid(True)
plt.show()
