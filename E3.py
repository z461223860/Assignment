import torch
import matplotlib.pyplot as plt
import numpy as np
import E1 as ee
import E2 as eee

dim = 2
H = np.identity(dim)
M = np.identity(dim)
C = 0.1 * np.identity(dim)
D = 0.1 * np.identity(dim)
R = np.identity(dim)
sigma = np.array([0.05,0.05])
t_0 = 0
T = 1
time_tensor_dim = 1
space_tensor_dim = 1 * 2
lqr = ee.LQR(H, M, C, D, R, t_0, T)


class PDE_DGM(torch.nn.Module):
    def __init__(self, H, M, C, D, R, sigma, alpha=np.array([1.0, 1.0]).reshape(2, 1)):
        super().__init__()
        self.H = torch.tensor(H)
        self.M = torch.tensor(M)
        self.C = torch.tensor(C)
        self.D = torch.tensor(D)
        self.R = torch.tensor(R)
        self.sigma = torch.tensor(sigma).reshape(2, 1)
        self.sigma_sigma_t = torch.matmul(self.sigma, self.sigma.T)
        self.alpha = torch.tensor(alpha).reshape(2, 1)
        self.M_alpha = torch.matmul(self.M, self.alpha)
        self.a_D_a = torch.matmul(self.alpha.T, torch.matmul(self.D, self.alpha))
        self.net_dgm = eee.Net_DGM(time_tensor_dim + space_tensor_dim - 1, 100, activation='Tanh')