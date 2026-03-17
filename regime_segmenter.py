import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Matern32Kernel(nn.Module):
    def __init__(self, lengthscale=1.0, variance=1.0):
        super().__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale, dtype=torch.float32))
        self.variance = nn.Parameter(torch.tensor(variance, dtype=torch.float32))

    def forward(self, x1, x2):
        # x1: (N, 1), x2: (M, 1)
        dist = torch.cdist(x1, x2)
        sqrt3_d = np.sqrt(3) * dist / self.lengthscale
        return self.variance * (1 + sqrt3_d) * torch.exp(-sqrt3_d)

class ChangePointKernel(nn.Module):
    def __init__(self, c, s=10.0):
        super().__init__()
        self.c = c # Change point location
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.k1 = Matern32Kernel()
        self.k2 = Matern32Kernel()

    def forward(self, x1, x2):
        sig1 = torch.sigmoid(self.s * (x1 - self.c))
        sig2 = torch.sigmoid(self.s * (x2 - self.c))
        
        k1_val = self.k1(x1, x2)
        k2_val = self.k2(x1, x2)
        
        return sig1 * k1_val * sig2.T + (1 - sig1) * k2_val * (1 - sig2).T

def gp_marginal_likelihood(kernel, x, y, noise_var=1e-4):
    K = kernel(x, x) + noise_var * torch.eye(x.size(0), device=x.device)
    try:
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(1), L)
        mll = -0.5 * y.unsqueeze(0) @ alpha - torch.sum(torch.log(torch.diag(L))) - 0.5 * x.size(0) * np.log(2 * np.pi)
        return mll.squeeze()
    except:
        return torch.tensor(-1e9, device=x.device)

def optimize_gp(kernel, x, y, steps=50):
    optimizer = optim.Adam(kernel.parameters(), lr=0.1)
    best_mll = -1e9
    for _ in range(steps):
        optimizer.zero_grad()
        mll = gp_marginal_likelihood(kernel, x, y)
        loss = -mll
        loss.backward()
        optimizer.step()
        if mll.item() > best_mll:
            best_mll = mll.item()
    return best_mll

def detect_change_points(prices, window_size=60):
    """
    Algorithm 1: Offline Change-Point Detection
    """
    n = len(prices)
    change_points = []
    x_all = torch.arange(n, dtype=torch.float32).unsqueeze(1)
    y_all = torch.tensor(prices, dtype=torch.float32)
    
    step = 20
    for i in range(0, n - window_size, step):
        x = x_all[i:i+window_size]
        y = y_all[i:i+window_size]
        y = (y - y.mean()) / (y.std() + 1e-5)
        
        # Single kernel model GP_M
        km = Matern32Kernel()
        lm = optimize_gp(km, x, y)
        
        # Dual kernel model GP_C
        best_lc = -1e9
        best_c = -1
        for c_idx in range(10, window_size - 10, 10):
            c_val = x[c_idx].item()
            kc = ChangePointKernel(c=c_val)
            lc = optimize_gp(kc, x, y)
            if lc > best_lc:
                best_lc = lc
                best_c = c_val
                
        # Severity check: nu = exp(L_C) / (exp(L_M) + exp(L_C))
        # Using log-sum-exp trick for numerical stability
        if best_lc > -1e8 and lm > -1e8:
            max_l = max(lm, best_lc)
            nu = np.exp(best_lc - max_l) / (np.exp(lm - max_l) + np.exp(best_lc - max_l))
            if nu >= 0.9:
                change_points.append(int(best_c))
                
    return sorted(list(set(change_points)))

def build_regime_library(prices, features, returns, min_len=5, max_len=21):
    """
    Segment data into regimes and store in Regime_Library
    """
    cps = detect_change_points(prices)
    cps = [0] + cps + [len(prices)]
    
    library = []
    for i in range(len(cps) - 1):
        start = cps[i]
        end = cps[i+1]
        length = end - start
        
        if length < min_len:
            continue
            
        # Split into chunks of max_len (5 to 21 days)
        for j in range(start, end, max_len):
            chunk_end = min(j + max_len, end)
            if chunk_end - j >= min_len:
                regime_x = features[j:chunk_end]
                regime_y = returns[j:chunk_end]
                library.append({
                    'x': regime_x,
                    'r': regime_y,
                    'start': j,
                    'end': chunk_end
                })
    return library
