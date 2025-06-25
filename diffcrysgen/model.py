

# Importing necessary libraries

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal Positional Embedding

class PositionalEmbedding(torch.nn.Module):
    """
    Given a noise label, it provides the corresponding positional embedding vector.
    """
    def __init__(self, emb_dim, max_positions = 10000):
        """
        Parameter : emb_dim (Type : Int) : Embedding Dimension
        """
        super(PositionalEmbedding, self).__init__()
        self.emb_dim = emb_dim
        self.max_positions = max_positions

    def forward(self, x):
        """
        Input : x [Tensor object], representing noise levels for a batch of input images
        """

        # Precompute frequencies for efficiency
        freqs = torch.arange(0, self.emb_dim // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.emb_dim // 2)
        freqs = (1 / self.max_positions) ** freqs

        # Outer product between Input tensor and freqs tensor
        x = torch.tensordot(x, freqs, dims=0)
        # Sinusoidal embedding
        # sin block
        sin_block = torch.sin(x)
        # cos block
        cos_block = torch.cos(x)
        # Concatenate sin and cos block along column dimension
        embedding = torch.cat([sin_block, cos_block], dim=1)
        # Finally, concatenate sin and cos block in such a way so that even columns
        # come from sin block and odd column comes from cos block.
        sin_unsqueezed = sin_block.unsqueeze(2)
        cos_unsqueezed = cos_block.unsqueeze(2)
        final_embedding = torch.reshape(torch.cat([sin_unsqueezed, cos_unsqueezed], dim=2), (embedding.shape[0], -1))
        return final_embedding


# Module for Preconditioning
# Adapted from "Elucidating the Design Space of Diffusion-Based
# Generative Models" by Karras et al.

class Precond(nn.Module):
  def __init__(self,
               denoise_fn,
               sigma_min=0,               # minimum supported noise level
               sigma_max=float("inf"),    # maximum supported noise level
               sigma_data=0.5,            # expected standard deviation of training data
               ):
    super().__init__()

    self.denoise_fn_F = denoise_fn
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.sigma_data = sigma_data

  def forward(self, x, sigma):
    x = x.to(torch.float32)
    sigma = sigma.to(torch.float32).reshape(-1,1,1)
    dtype = torch.float32

    c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
    c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
    c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
    c_noise = sigma.log() / 4

    x_in = c_in * x
    F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

    assert F_x.dtype == dtype
    D_x = c_skip * x + c_out * F_x.to(torch.float32)
    return D_x

  def round_sigma(self, sigma):
    return torch.as_tensor(sigma)


# EDM Loss

def EDMLoss(denoise_fn, data):
  P_mean = -1.2
  P_std = 1.2
  sigma_data = 0.5
  rnd_normal = torch.randn(data.shape[0], device=data.device)
  sigma = (rnd_normal * P_std + P_mean).exp()
  weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
  y = data
  n = torch.randn_like(y) * sigma.unsqueeze(1).unsqueeze(1)
  D_yn = denoise_fn(y + n, sigma)
  target = y
  loss = weight.unsqueeze(1).unsqueeze(1) * ((D_yn - target) ** 2)
  return loss


# Final Model

class Model(nn.Module):
  def __init__(self, denoise_fn, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    super().__init__()
    self.P_mean = P_mean
    self.P_std = P_std
    self.sigma_data = sigma_data
    self.denoise_fn_D = Precond(denoise_fn)

  def forward(self, x):
    loss = EDMLoss(self.denoise_fn_D, x)
    return loss.mean(-1).mean()


# time-encoding layer

def make_te(dim_in, dim_out):
    return nn.Sequential(nn.Linear(dim_in, dim_out),nn.SiLU(),nn.Linear(dim_out, dim_out))



#=================================== UNet================================================= 

def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, kernel_size = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    # Standardize weights
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, kernel_size)


class WSConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = 'zeros', eps: float = 1e-5):
        super(WSConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # Apply weight standardization before convolution
        standardized_weight = weight_standardization(self.weight, self.eps)
        return F.conv1d(x, standardized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, eps=1e-5):
        super(ResNetBlock, self).__init__()

        # First WSConv1d layer with GroupNorm
        self.conv1 = WSConv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=groups, eps=eps)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        # Second WSConv1d layer with GroupNorm
        self.conv2 = WSConv1d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, groups=groups, eps=eps)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        # Shortcut connection to match the shape when in_channels != out_channels
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) if in_channels != out_channels else nn.Identity()

        # activation
        self.activation = nn.SiLU()

    def forward(self,x):
        # Save the input for the residual connection
        identity = x

        # First conv-norm-activation
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)

        # Second conv-norm-activation
        out = self.conv2(out)
        out = self.norm2(out)

        # Add the residual (shortcut connection)
        identity = self.shortcut(identity)
        out += identity

        # Apply ReLU after adding the residual
        out = self.activation(out)
        
        return out


# attention block
# we apply attention block at a resolution level of 17

class AttentionBlock(nn.Module):
    """
    MHA expects input of shape (batch_size, seq_length, embedding_dim).
    Our data shape (batch_size, num_channels, resolution), e.g. (64,128,17).
    embedding_dim == resolution
    seq_length = num_channels
    """
    def __init__(self, emb_dim, num_heads=1, dropout=0.1):
        super(AttentionBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout)
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(emb_dim)
        self.layernorm2 = nn.LayerNorm(emb_dim)
        
        # Feed-forward network (after attention)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
       
    def forward(self, x):
        
        # Apply self-attention
        attn_output, attn_weights = self.attention(x, x, x)  # Query, Key, Value all are x (self-attention)
        
        # Residual connection followed by layer normalization : add & norm
        x = self.layernorm1(x + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Residual connection followed by layer normalization : add & norm
        x = self.layernorm2(x + ffn_output)
        
        return x


class UNet(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super(UNet, self).__init__()

        # map noise labels
        self.map_noise = PositionalEmbedding(emb_dim=time_emb_dim)

        # encoder path
        self.te1 = make_te(time_emb_dim, 3)
        self.b1 = nn.Sequential(ResNetBlock(3, 32),ResNetBlock(32, 32),ResNetBlock(32,32)) # shape = [batch,32,136]
        self.down1 = nn.Conv1d(32, 32, 4, 2, 1) # shape = [batch,32,68]

        self.te2 = make_te(time_emb_dim, 32)
        self.b2 = nn.Sequential(ResNetBlock(32, 64),ResNetBlock(64, 64),ResNetBlock(64,64)) # shape = [batch,64,68]
        self.down2 = nn.Conv1d(64, 64, 4, 2, 1) # shape = [batch,64,34]

        self.te3 = make_te(time_emb_dim, 64)
        self.b3 = nn.Sequential(ResNetBlock(64, 128),ResNetBlock(128, 128),ResNetBlock(128,128)) # shape = [batch,128,34]
        self.down3 = nn.Conv1d(128, 128, 4, 2, 1) # shape = [batch,128,17]

        self.attn1 = AttentionBlock(emb_dim=17)

        # Bottleneck
        self.te_mid = make_te(time_emb_dim, 128)
        self.b_mid = nn.Sequential(ResNetBlock(128, 64),ResNetBlock(64, 64),ResNetBlock(64,128)) # shape = [batch,128,17]

        self.attn2 = AttentionBlock(emb_dim=17)


        # decoder path
        self.up1 = nn.ConvTranspose1d(128, 128, 4, 2, 1) # shape = [batch,128,34]
        self.te4 = make_te(time_emb_dim, 256)
        self.b4 = nn.Sequential(ResNetBlock(256,128),ResNetBlock(128,64),ResNetBlock(64,64)) # shape = [batch,64,34]

        self.up2 = nn.ConvTranspose1d(64, 64, 4, 2, 1) # shape = [batch,64,68]
        self.te5 = make_te(time_emb_dim, 128)
        self.b5 = nn.Sequential(ResNetBlock(128, 64),ResNetBlock(64, 32),ResNetBlock(32, 32)) # shape = [batch,32,68]

        self.up3 = nn.ConvTranspose1d(32, 32, 4, 2, 1) # shape = [batch,32,136]
        self.te6 = make_te(time_emb_dim, 64)
        self.b6 = nn.Sequential(ResNetBlock(64, 32),ResNetBlock(32, 32),ResNetBlock(32, 32)) # shape = [batch,32,136]

        # output
        self.conv_out = nn.Conv1d(32, 3, 3, 1, 1) # shape = [batch,3,136]

    def forward(self, x, noise_labels):
        t = self.map_noise(noise_labels)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1))  
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1))  
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1))  

        out_mid = self.b_mid(self.attn1(self.down3(out3)) + self.te_mid(t).reshape(n, -1, 1)) 

        # self-attention
        out_mid = self.attn2(out_mid)

        out4 = torch.cat((out3,self.up1(out_mid)), dim=1) 
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1))  
        out5 = torch.cat((out2, self.up2(out4)), dim=1) 
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1)) 

        out6 = torch.cat((out1, self.up3(out5)), dim=1)  
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1))  

        out = self.conv_out(out6) 
        return out









