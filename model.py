import torch
import torch.nn as nn

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Dynamic weighting of inputs (e.g., 1, 21, 63, 126, 252 days returns + MACD)
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )
        self.feature_network = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: (B, L, input_dim)
        weights = self.weight_network(x) # (B, L, input_dim)
        features = self.feature_network(x) # (B, L, input_dim)
        return weights * features # (B, L, input_dim)

class XTrendModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Preprocessing
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        
        # 2. Encoder
        # LSTM with state preservation
        self.encoder_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Core Attention Module (Multi-head Cross-Attention)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 4. Decoder
        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        
        self.ffn_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Output z_t in (-1, 1)
        )
        
        self.ffn_mle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2) # Output mu and log_sigma for MLE
        )

    def forward(self, target_x, context_x):
        """
        target_x: (B, L_t, input_dim) - Target sequence features
        context_x: (B, C, L_c, input_dim) - Context regimes from library
        """
        B, L_t, _ = target_x.shape
        _, C, L_c, _ = context_x.shape
        
        # --- Preprocessing ---
        target_feat = self.vsn(target_x) # Shape: (B, L_t, input_dim)
        target_feat = self.feature_proj(target_feat) # Shape: (B, L_t, hidden_dim)
        
        context_x_flat = context_x.view(B * C, L_c, -1) # Shape: (B*C, L_c, input_dim)
        context_feat_flat = self.vsn(context_x_flat) # Shape: (B*C, L_c, input_dim)
        context_feat_flat = self.feature_proj(context_feat_flat) # Shape: (B*C, L_c, hidden_dim)
        
        # --- Encoder ---
        # Target encoding
        q_t, _ = self.encoder_lstm(target_feat) # Shape: (B, L_t, hidden_dim)
        
        # Context encoding
        _, (h_n, _) = self.encoder_lstm(context_feat_flat) # h_n Shape: (1, B*C, hidden_dim)
        context_repr = h_n.squeeze(0).view(B, C, self.hidden_dim) # Shape: (B, C, hidden_dim)
        
        K = context_repr # Shape: (B, C, hidden_dim)
        V = context_repr # Shape: (B, C, hidden_dim)
        
        # --- Multi-head Cross-Attention ---
        # Step 1: Context self-attention
        V_prime, _ = self.self_attn(V, V, V) # Shape: (B, C, hidden_dim)
        
        # Step 2: Cross-attention
        y_t, _ = self.cross_attn(q_t, K, V_prime) # Shape: (B, L_t, hidden_dim)
        
        # --- Decoder ---
        # Concatenate q_t and y_t
        decoder_in = torch.cat([q_t, y_t], dim=-1) # Shape: (B, L_t, hidden_dim * 2)
        
        dec_out, _ = self.decoder_lstm(decoder_in) # Shape: (B, L_t, hidden_dim)
        
        # Final outputs
        z_t = self.ffn_z(dec_out).squeeze(-1) # Shape: (B, L_t)
        
        mle_params = self.ffn_mle(dec_out) # Shape: (B, L_t, 2)
        mu = mle_params[:, :, 0] # Shape: (B, L_t)
        log_sigma = mle_params[:, :, 1] # Shape: (B, L_t)
        sigma = torch.exp(log_sigma) + 1e-6 # Shape: (B, L_t)
        
        return z_t, mu, sigma
