class SpectralConv1d(eqx.Module):
    weights: Any
    in_c: int
    out_c: int
    modes: int

    def __init__(self, key, in_c, out_c, modes):
        self.in_c = in_c
        self.out_c = out_c
        self.modes = modes
        scale = 1 / (in_c * out_c)
        self.weights = scale * jr.normal(key, (in_c, out_c, modes), dtype=jnp.complex64)

    def __call__(self, x):
        x_ft = jnp.fft.rfft(x, axis=-1)
        n_modes = min(self.modes, x_ft.shape[-1])
        x_ft_cut = x_ft[:, :n_modes]
        out_ft = jnp.einsum("ix,iox->ox", x_ft_cut, self.weights[:, :, :n_modes])
        x_new = jnp.fft.irfft(out_ft, n=x.shape[-1], axis=-1)
        return x_new

class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    layers: list
    projection: eqx.nn.Conv1d

    def __init__(self, key, in_channels, out_channels, modes, width, n_layers):
        keys = jr.split(key, n_layers + 2)
        self.lifting = eqx.nn.Conv1d(in_channels, width, kernel_size=1, key=keys[0])
        self.layers = []
        
        for i in range(n_layers):
            k1, k2 = jr.split(keys[i+1])
            layer_blocks = [
                SpectralConv1d(k1, width, width, modes),
                eqx.nn.Conv1d(width, width, kernel_size=1, key=k2)
            ]
            self.layers.append(layer_blocks)

        self.projection = eqx.nn.Conv1d(width, out_channels, kernel_size=1, key=keys[-1])

    def __call__(self, x):
        x = self.lifting(x)
        for i in range(len(self.layers)):
            spectral_op = self.layers[i][0]
            linear_op = self.layers[i][1]
            x1 = spectral_op(x)
            x2 = linear_op(x)
            x = jax.nn.gelu(x1 + x2)
            
        x = self.projection(x)
        return x
