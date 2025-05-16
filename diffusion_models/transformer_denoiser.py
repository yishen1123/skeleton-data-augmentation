# diffusion_models/transformer_denoiser.py
# Contains the definition for the Transformer-based denoising model.

import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Tensor of timesteps, shape (B,)
        Returns:
            torch.Tensor: Time embedding tensor, shape (B, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        
        scale = math.log(10000) / (half_dim - 1)
        freqs  = torch.exp(torch.arange(half_dim, device=device) * -scale)
        
        x = x.float()  # Ensure x is float for multiplication
        emb = x[:, None] * freqs[None, :] # Broadcast x to (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Concatenate sin and cos
        
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1)) # Pad last dimension
        return emb


class SimplePoseTransformerDenoiser(nn.Module):
    def __init__(self, input_features=34, sequence_length=500, model_dim=256,
                 num_layers=4, num_heads=4, dim_feedforward=1024, dropout=0.1):
        """
        Args:
            input_features (int): Dimension F of the input features per timestep (C*V*M). Default 34.
            sequence_length (int): Length T of the input sequence (segment_len). Default 500.
            model_dim (int): Hidden dimension of the Transformer. Default 256.
            num_layers (int): Number of TransformerEncoderLayer blocks. Default 4.
            num_heads (int): Number of attention heads. Default 4.
            dim_feedforward (int): Dimension of the feed-forward network. Default 1024.
            dropout (float): Dropout rate. Default 0.1.
        """
        super().__init__()
        self.input_features = input_features
        self.sequence_length = sequence_length # T
        self.model_dim = model_dim

        self.input_proj = nn.Linear(input_features, model_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),                # Generate sinusoidal embedding
            nn.Linear(model_dim, model_dim * 4),        # Expand dimension
            nn.Mish(),                                  # Activation function
            nn.Linear(model_dim * 4, model_dim)         # Project back to model dimension
        )

        #　Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, sequence_length, model_dim) * 0.02) 

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,                  # Model dimension
            nhead=num_heads,                    # Number of attention heads
            dim_feedforward=dim_feedforward,    # Feed-forward dimension
            dropout=dropout,                    # Dropout rate
            activation='gelu',                  # Activation function
            batch_first=True,                   # Expect input shape (B, T, F)
            norm_first=True                     # Apply LayerNorm before attention/FFN (more stable)
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers       
        )

        self.output_proj = nn.Linear(model_dim, input_features)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
             torch.nn.init.constant_(module.bias, 0)
             torch.nn.init.constant_(module.weight, 1.0)


    def forward(self, x, t):
        """
        Forward pass of the denoiser model.

        Args:
            x (torch.Tensor): Noisy pose sequence tensor, shape (B, T, F).
            t (torch.Tensor): Timesteps tensor, shape (B,).

        Returns:
            torch.Tensor: Predicted noise tensor, shape (B, T, F).
        """
        B, T, F = x.shape
        if T != self.sequence_length:
             raise ValueError(f"Input sequence length {T} != model expected {self.sequence_length}")
        if F != self.input_features:
             raise ValueError(f"Input features {F} != model expected {self.input_features}")

        x = self.input_proj(x)                                  # (B, T, F) -> (B, T, model_dim)

        t_emb = self.time_mlp(t)                                # (B,) -> (B, model_dim)

        x = x + t_emb.unsqueeze(1)                              # (B, T, model_dim) + (B, 1, model_dim) -> (B, T, model_dim)

        x = x + self.pos_encoder                                # (B, T, model_dim) + (1, T, model_dim) -> (B, T, model_dim)

        transformer_output = self.transformer_encoder(x)

        noise_pred = self.output_proj(transformer_output)       # (B, T, model_dim) -> (B, T, F)

        return noise_pred 