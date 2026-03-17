import torch
import torch.optim as optim
import numpy as np
from model import XTrendModel

def mle_loss(mu, sigma, target_returns):
    """
    Negative log-likelihood of next-day returns based on Gaussian distribution.
    mu: (B, L_t)
    sigma: (B, L_t)
    target_returns: (B, L_t)
    """
    nll = 0.5 * torch.log(2 * np.pi * sigma**2) + ((target_returns - mu)**2) / (2 * sigma**2)
    return nll.mean()

def sharpe_loss(z_t, target_returns):
    """
    Directly optimize the Sharpe ratio of predicted positions.
    z_t: (B, L_t)
    target_returns: (B, L_t)
    """
    R = z_t * target_returns # Portfolio returns: (B, L_t)
    
    mean_R = R.mean()
    std_R = R.std() + 1e-6
    
    sharpe = mean_R / std_R
    return -sharpe # Minimize negative Sharpe

def joint_loss(z_t, mu, sigma, target_returns, alpha=0.1):
    """
    L_Joint = alpha * L_MLE + L_Sharpe
    """
    l_mle = mle_loss(mu, sigma, target_returns)
    l_sharpe = sharpe_loss(z_t, target_returns)
    return alpha * l_mle + l_sharpe

def train_expanding_window(model, data_x, data_y, regime_library, num_epochs=50, lr=1e-3, alpha=0.1):
    """
    Expanding Window backtesting & training scheme.
    data_x: (Total_Time, input_dim)
    data_y: (Total_Time,) - Next day returns
    regime_library: list of dicts with 'x' (L_c, input_dim) and 'start', 'end'
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    total_time = len(data_x)
    initial_window = 252 # 1 year initial training
    step_size = 21 # 1 month step
    
    predictions = []
    
    for t in range(initial_window, total_time, step_size):
        print(f"Training expanding window: 0 to {t}")
        train_x = data_x[:t]
        train_y = data_y[:t]
        
        # Strictly enforce causality: Context set data must be earlier than target sequence
        valid_regimes = [r['x'] for r in regime_library if r['end'] <= t]
        if len(valid_regimes) == 0:
            valid_regimes = [torch.zeros(10, data_x.shape[-1])]
            
        # Pad regimes to fixed length L_c=21
        L_c = 21
        padded_regimes = []
        for r in valid_regimes:
            if len(r) >= L_c:
                padded_regimes.append(r[-L_c:])
            else:
                pad = torch.zeros(L_c - len(r), r.shape[-1])
                padded_regimes.append(torch.cat([pad, r], dim=0))
        
        context_x = torch.stack(padded_regimes).unsqueeze(0) # (1, C, L_c, input_dim)
        
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Chunk train_x into sequences of length L_t = 63
            L_t = 63
            batch_x = []
            batch_y = []
            for i in range(0, len(train_x) - L_t, 10):
                batch_x.append(train_x[i:i+L_t])
                batch_y.append(train_y[i:i+L_t])
                
            if len(batch_x) == 0:
                continue
                
            batch_x = torch.stack(batch_x) # (B, L_t, input_dim)
            batch_y = torch.stack(batch_y) # (B, L_t)
            
            B = batch_x.shape[0]
            batch_context = context_x.expand(B, -1, -1, -1) # (B, C, L_c, input_dim)
            
            z_t, mu, sigma = model(batch_x, batch_context)
            
            loss = joint_loss(z_t, mu, sigma, batch_y, alpha)
            loss.backward()
            optimizer.step()
            
        # Testing on the next step_size window
        model.eval()
        test_end = min(t + step_size, total_time)
        test_x = data_x[t:test_end].unsqueeze(0) # (1, L_test, input_dim)
        
        with torch.no_grad():
            test_context = context_x # (1, C, L_c, input_dim)
            z_test, _, _ = model(test_x, test_context)
            predictions.append(z_test.squeeze(0))
            
    if len(predictions) > 0:
        return torch.cat(predictions, dim=0)
    return None

if __name__ == "__main__":
    # Dummy data test
    input_dim = 10
    total_time = 500
    data_x = torch.randn(total_time, input_dim)
    data_y = torch.randn(total_time)
    
    regime_library = [
        {'x': torch.randn(15, input_dim), 'start': 0, 'end': 15},
        {'x': torch.randn(20, input_dim), 'start': 15, 'end': 35},
        {'x': torch.randn(10, input_dim), 'start': 35, 'end': 45},
    ]
    
    model = XTrendModel(input_dim=input_dim, hidden_dim=32)
    preds = train_expanding_window(model, data_x, data_y, regime_library, num_epochs=2)
    print("Predictions shape:", preds.shape if preds is not None else "None")
