# ============================================================
# exam_fno_pod_1d.py  (POD per variable + 1D FNO + training loop)
# ============================================================
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# 0) Utils: shapes + seeding
# ----------------------------
def set_seed(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_torch(x: np.ndarray, device: torch.device):
    return torch.tensor(x, dtype=torch.float32, device=device)

def assert_shape(x, name="X"):
    assert x.ndim == 3, f"{name} must be (Batch, Channels, Nx), got {x.shape}"
    B, C, Nx = x.shape
    assert B > 0 and C > 0 and Nx > 1, f"bad {name} shape: {x.shape}"

class ChannelNormalizer:
    """
    Normalize per-channel using mean/std computed over (batch, x).
    x shape: (B, C, Nx)
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        # mean/std over batch and space for each channel
        self.mean = x.mean(dim=(0, 2), keepdim=True)           # (1,C,1)
        self.std = x.std(dim=(0, 2), keepdim=True) + self.eps  # (1,C,1)
        return self

    def transform(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor):
        return x * self.std + self.mean


# ----------------------------
# 1) POD per variable (one POD per channel)
# ----------------------------
def pod_per_variable(X: np.ndarray, r: int, center: bool = True):
    """
    X: (Batch, n_vars, Nx) numpy
    r: number of POD modes kept per variable
    Returns:
      mean:  (n_vars, Nx)          (0 if center=False)
      basis: (n_vars, r, Nx)       POD spatial modes
      coeff: (Batch, n_vars, r)    POD coefficients
      energy: (n_vars, r)          normalized captured energy per variable
    """
    assert_shape(X, "X")
    B, n_vars, Nx = X.shape
    r = min(r, B, Nx)

    mean = np.zeros((n_vars, Nx), dtype=X.dtype)
    basis = np.zeros((n_vars, r, Nx), dtype=X.dtype)
    coeff = np.zeros((B, n_vars, r), dtype=X.dtype)
    energy = np.zeros((n_vars, r), dtype=X.dtype)

    for j in range(n_vars):
        Xj = X[:, j, :]  # (B, Nx)
        if center:
            mj = Xj.mean(axis=0, keepdims=True)  # (1, Nx)
            Xc = Xj - mj
            mean[j, :] = mj[0]
        else:
            Xc = Xj

        # SVD of snapshot matrix: Xc = U S Vt
        # Vt rows are spatial modes
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Keep r modes
        Vtr = Vt[:r, :]                 # (r, Nx)
        Sr = S[:r]                      # (r,)
        Ur = U[:, :r]                   # (B, r)

        basis[j, :, :] = Vtr
        coeff[:, j, :] = Ur * Sr[None, :]  # (B,r) times singular values

        # captured energy ratio
        total = (S**2).sum() + 1e-12
        energy[j, :] = (Sr**2) / total

    return mean, basis, coeff, energy

def pod_reconstruct(mean, basis, coeff):
    """
    mean:  (n_vars, Nx)
    basis: (n_vars, r, Nx)
    coeff: (B, n_vars, r)
    return Xhat: (B, n_vars, Nx)
    """
    B, n_vars, r = coeff.shape
    Nx = basis.shape[-1]
    Xhat = np.zeros((B, n_vars, Nx), dtype=coeff.dtype)
    for j in range(n_vars):
        # coeff_j (B,r) @ basis_j (r,Nx) -> (B,Nx)
        Xhat[:, j, :] = coeff[:, j, :] @ basis[j, :, :] + mean[j, :][None, :]
    return Xhat


# ----------------------------
# 2) 1D FNO
# ----------------------------
class SpectralConv1d(nn.Module):
    """
    1D spectral convolution using rFFT.
    Input:  (B, Cin, Nx)
    Output: (B, Cout, Nx)
    """
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-frequency modes to keep

        # complex weights: (Cin, Cout, modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def compl_mul1d(self, x_ft, w_real, w_imag):
        # x_ft: (B, Cin, K) complex
        # w:    (Cin, Cout, K)
        # -> (B, Cout, K)
        w = torch.complex(w_real, w_imag)
        return torch.einsum("bik,iok->bok", x_ft, w)

    def forward(self, x: torch.Tensor):
        B, Cin, Nx = x.shape
        K = Nx // 2 + 1
        x_ft = torch.fft.rfft(x, dim=-1)  # (B,Cin,K) complex

        out_ft = torch.zeros(B, self.out_channels, K, device=x.device, dtype=torch.cfloat)

        m = min(self.modes, K)
        out_ft[:, :, :m] = self.compl_mul1d(x_ft[:, :, :m], self.weight_real[:, :, :m], self.weight_imag[:, :, :m])

        x_out = torch.fft.irfft(out_ft, n=Nx, dim=-1)  # (B,Cout,Nx)
        return x_out


class FNO1d(nn.Module):
    """
    1D FNO with:
      - lift: 1x1 conv to width
      - L blocks: SpectralConv1d + 1x1 conv (residual)
      - projection: 1x1 conv to Cout
    """
    def __init__(self, in_channels: int, out_channels: int, width: int = 64, modes: int = 16, depth: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.depth = depth

        self.lift = nn.Conv1d(in_channels, width, kernel_size=1)

        self.spec_convs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.w_convs = nn.ModuleList([nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)])

        self.proj1 = nn.Conv1d(width, width, kernel_size=1)
        self.proj2 = nn.Conv1d(width, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # x: (B, Cin, Nx)
        x = self.lift(x)  # (B,width,Nx)
        for k in range(self.depth):
            x1 = self.spec_convs[k](x)
            x2 = self.w_convs[k](x)
            x = F.gelu(x1 + x2)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)
        return x


# ----------------------------
# 3) Dataset
# ----------------------------
class TensorPairDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        assert_shape(X, "X")
        assert_shape(Y, "Y")
        assert X.shape[0] == Y.shape[0] and X.shape[-1] == Y.shape[-1], "Batch/Nx mismatch"
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ----------------------------
# 4) Training loop
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0.0
    n = 0
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        bs = xb.shape[0]
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)

def train_fno(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    width: int = 64,
    modes: int = 16,
    depth: int = 4,
    batch_size: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
    save_path: str = "best_fno1d.pt",
):
    Cin = X_train.shape[1]
    Cout = Y_train.shape[1]

    train_loader = DataLoader(TensorPairDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorPairDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    model = FNO1d(Cin, Cout, width=width, modes=modes, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            total += loss.item() * bs
            n += bs

        train_loss = total / max(n, 1)
        val_loss = evaluate(model, val_loader, loss_fn)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, save_path)

        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[epoch {ep:03d}] train={train_loss:.6e}  val={val_loss:.6e}  best={best_val:.6e}")

    # load best
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return model


# ----------------------------
# 5) Replace this with exam data loading
# ----------------------------
def load_your_data():
    """
    ✅ 考试时你只需要把这里替换成“读老师数据”的部分。
    约定：
      X: (Batch, Cin, Nx)
      Y: (Batch, Cout, Nx)
    """
    # ---- DEMO synthetic data ----
    B = 512
    Nx = 256
    Cin = 2
    Cout = 2

    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    X = np.zeros((B, Cin, Nx), dtype=np.float32)
    Y = np.zeros((B, Cout, Nx), dtype=np.float32)

    for i in range(B):
        a = np.random.uniform(0.5, 2.0)
        b = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)

        X[i, 0, :] = np.sin(a*x + phase)
        X[i, 1, :] = np.cos(b*x + phase)

        # target: some nonlinear mapping
        Y[i, 0, :] = 0.7 * X[i, 0, :] + 0.2 * (X[i, 1, :]**2)
        Y[i, 1, :] = 0.5 * X[i, 1, :] + 0.1 * np.sin(X[i, 0, :])

    return X, Y


# ----------------------------
# 6) Main: POD + FNO training
# ----------------------------
def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # === Load data ===
    X_np, Y_np = load_your_data()
    assert_shape(X_np, "X")
    assert_shape(Y_np, "Y")

    B, Cin, Nx = X_np.shape
    Cout = Y_np.shape[1]
    print(f"Data: B={B}, Cin={Cin}, Cout={Cout}, Nx={Nx}")

    # === (A) POD per variable (optional in pipeline, but required by exam) ===
    # You can run POD on X or Y depending on the question.
    r = min(16, B, Nx)
    meanX, basisX, coeffX, energyX = pod_per_variable(X_np, r=r, center=True)
    print("POD done. Example energy captured for var0 first 5 modes:", energyX[0, :5])

    # reconstruction check (quick sanity)
    X_rec = pod_reconstruct(meanX, basisX, coeffX)
    rec_err = np.linalg.norm(X_np - X_rec) / (np.linalg.norm(X_np) + 1e-12)
    print(f"POD relative reconstruction error (X): {rec_err:.3e}")

    # === (B) Train FNO ===
    # split
    n_train = int(0.8 * B)
    Xtr_np, Xva_np = X_np[:n_train], X_np[n_train:]
    Ytr_np, Yva_np = Y_np[:n_train], Y_np[n_train:]

    Xtr = to_torch(Xtr_np, device)
    Ytr = to_torch(Ytr_np, device)
    Xva = to_torch(Xva_np, device)
    Yva = to_torch(Yva_np, device)

    # normalize (strongly recommended)
    xnorm = ChannelNormalizer().fit(Xtr)
    ynorm = ChannelNormalizer().fit(Ytr)
    Xtr_n = xnorm.transform(Xtr)
    Ytr_n = ynorm.transform(Ytr)
    Xva_n = xnorm.transform(Xva)
    Yva_n = ynorm.transform(Yva)

    model = train_fno(
        Xtr_n, Ytr_n, Xva_n, Yva_n,
        width=64, modes=16, depth=4,
        batch_size=32, epochs=50, lr=1e-3,
        device=device, save_path="best_fno1d.pt"
    )

    # inference example
    model.eval()
    with torch.no_grad():
        pred_n = model(Xva_n[:8])
        pred = ynorm.inverse(pred_n)
        mse = F.mse_loss(pred, Yva[:8]).item()
        print("Example MSE on 8 val samples:", mse)

    print("OK. You have: POD-per-variable + 1D FNO + training loop.")


if __name__ == "__main__":
    main()
