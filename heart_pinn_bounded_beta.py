# ============================================
# Heart PINN with BOUNDED beta
# ============================================

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# --------------------------------------------
# True parameters
# --------------------------------------------
alpha_true = 0.65
beta_true = 16.0
A_true = 1.6
f_true = 1.35

x0_true = 0.15
v0_true = 0.0

# --------------------------------------------
# Beta bounds (IMPORTANT PART)
# --------------------------------------------
beta_min = 8.0
beta_max = 20.0

# --------------------------------------------
# Data
# --------------------------------------------
t_min_raw, t_max_raw = 0.0, 8.0
n_true = 1200
n_obs = 300
noise_std_x = 0.03
noise_std_v = 0.02

def rk4(alpha, beta, A, f, x0, v0, t):
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    x[0], v[0] = x0, v0

    def rhs(tt, xx, vv):
        return vv, A*np.sin(2*np.pi*f*tt) - alpha*vv - beta*xx

    for i in range(len(t)-1):
        h = t[i+1] - t[i]

        k1x, k1v = rhs(t[i], x[i], v[i])
        k2x, k2v = rhs(t[i]+h/2, x[i]+h*k1x/2, v[i]+h*k1v/2)
        k3x, k3v = rhs(t[i]+h/2, x[i]+h*k2x/2, v[i]+h*k2v/2)
        k4x, k4v = rhs(t[i]+h, x[i]+h*k3x, v[i]+h*k3v)

        x[i+1] = x[i] + h*(k1x+2*k2x+2*k3x+k4x)/6
        v[i+1] = v[i] + h*(k1v+2*k2v+2*k3v+k4v)/6

    return x, v

t_raw = np.linspace(t_min_raw, t_max_raw, n_true)
x_true, v_true = rk4(alpha_true, beta_true, A_true, f_true, x0_true, v0_true, t_raw)

T_scale = t_max_raw - t_min_raw
tau = (t_raw - t_min_raw) / T_scale

idx = np.sort(np.random.choice(n_true, n_obs, replace=False))

tau_obs = tau[idx]
t_obs_raw = t_raw[idx]

x_obs = x_true[idx] + np.random.normal(0, noise_std_x, n_obs)
v_obs = v_true[idx] + np.random.normal(0, noise_std_v, n_obs)

# --------------------------------------------
# Helpers
# --------------------------------------------
def to_tensor(x, grad=False):
    return torch.tensor(x, dtype=torch.float32, device=device).view(-1,1).requires_grad_(grad)

def grad(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]

# --------------------------------------------
# PINN with bounded beta
# --------------------------------------------
class PINN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1,32), nn.Tanh(),
            nn.Linear(32,32), nn.Tanh(),
            nn.Linear(32,32), nn.Tanh(),
            nn.Linear(32,1)
        )

        self.raw_beta = nn.Parameter(torch.tensor(0.0))

    def beta(self):
        s = torch.sigmoid(self.raw_beta)
        return beta_min + (beta_max - beta_min)*s

    def forward(self,tau):
        return self.net(tau)

# --------------------------------------------
# Model
# --------------------------------------------
model = PINN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

tau_obs_t = to_tensor(tau_obs, True)
x_obs_t = to_tensor(x_obs)
v_obs_t = to_tensor(v_obs)

tau_col = to_tensor(np.linspace(0,1,1200), True)

tau0 = to_tensor([0.0], True)
x0_t = to_tensor([x0_true])
v0_t = to_tensor([v0_true])

# --------------------------------------------
# Training
# --------------------------------------------
for epoch in range(5000):
    opt.zero_grad()

    x_pred = model(tau_obs_t)
    dx = grad(x_pred, tau_obs_t) / T_scale

    loss_x = ((x_pred - x_obs_t)**2).mean()
    loss_v = ((dx - v_obs_t)**2).mean()

    x_c = model(tau_col)
    dx_c = grad(x_c, tau_col)
    ddx_c = grad(dx_c, tau_col)

    t_raw_c = t_min_raw + T_scale * tau_col
    forcing = A_true * torch.sin(2*math.pi*f_true*t_raw_c)

    res = ddx_c/(T_scale**2) + alpha_true*dx_c/T_scale + model.beta()*x_c - forcing
    loss_phys = (res**2).mean()

    x_init = model(tau0)
    dx_init = grad(x_init, tau0)/T_scale

    loss_ic = ((x_init-x0_t)**2 + (dx_init-v0_t)**2).mean()

    loss = 10*loss_x + 10*loss_v + 10*loss_phys + 20*loss_ic

    loss.backward()
    opt.step()

    if epoch % 500 == 0:
        print(epoch, "beta =", model.beta().item())

# --------------------------------------------
# Results
# --------------------------------------------
beta_learned = model.beta().item()
error = abs(beta_learned - beta_true)/beta_true*100

print("\nTrue beta:", beta_true)
print("Learned beta:", beta_learned)
print("Error %:", error)

# Plot
tau_t = to_tensor(tau, True)
x_pred = model(tau_t).detach().cpu().numpy().flatten()

plt.figure(figsize=(10,5))
plt.plot(t_raw, x_true, label="True")
plt.scatter(t_obs_raw, x_obs, s=10, label="Noisy")
plt.plot(t_raw, x_pred, label="PINN")
plt.legend()
plt.title("Bounded Beta PINN")
plt.show()
