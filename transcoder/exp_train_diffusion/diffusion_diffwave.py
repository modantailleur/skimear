import tqdm
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torchaudio.transforms as TT
import  matplotlib.pyplot as plt


from glob import glob
from math import sqrt
from torch.nn import Linear, Conv1d, ConvTranspose2d, SiLU
from torch.utils.data.distributed import DistributedSampler

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    layer = nn.utils.weight_norm(layer)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    '''
    Positional Encoding
    detail could refer to:
    https://arxiv.org/abs/1706.03762 and https://arxiv.org/abs/2009.09761
    '''
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
            
        x = self.projection1(x)
        x = SiLU()(x)
        x = self.projection2(x)
        x = SiLU()(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x

class ResBlock(nn.Module):
    def __init__(self, res_channel, dilation, n_mels, cond=True):
        super().__init__()
        self.dilated_conv = Conv1d(res_channel, 2 * res_channel, 3,\
                                    padding=dilation, dilation=dilation)
        self.diffstep_proj = Linear(512, res_channel)
        self.cond_proj = Conv1d(n_mels, 2 * res_channel, 1)
        self.output_proj =  Conv1d(res_channel, 2 * res_channel, 1)
        self.cond = cond

    def forward(self, inp, diff_step, conditioner):
        
        diff_step = self.diffstep_proj(diff_step).unsqueeze(-1)
        x = inp + diff_step

        if self.cond:
            conditioner = self.cond_proj(conditioner)
            #MT: added next line: the linear interpolation makes it a bit weird, because it means that the audio 
            #will ultimately be time stretched and frequency stretched, but it's only of 0.3%
            # conditioner = nn.functional.interpolate(conditioner, size=x.size(-1), mode='linear', align_corners=True)
            x = self.dilated_conv(x) + conditioner
        gate, val = torch.chunk(x, 2, dim=1) # gate function
        x = torch.sigmoid(gate) * torch.tanh(val)
        
        x = self.output_proj(x)
        residual, skip = torch.chunk(x, 2, dim=1)
        return (inp + residual) / np.sqrt(2.0), skip


class DiffWave(nn.Module):
    def __init__(self, res_channels, n_layers, n_mels, cond=True, diff_steps=1000):
        super().__init__()
        self.cond = cond
        self.res_channels = res_channels
        self.n_layers = n_layers
        self.n_mels = n_mels
        self.inp_proj = Conv1d(1, res_channels, 1)
        self.noise_schedule = np.linspace(1e-4, 0.05, diff_steps)
        self.embedding = DiffusionEmbedding(len(self.noise_schedule))
        self.spectrogram_upsampler = SpectrogramUpsampler(n_mels)
        
        dilate_cycle = n_layers // 3
        self.layers = nn.ModuleList([
            ResBlock(res_channels, 2**(i % dilate_cycle), n_mels, self.cond)
            for i in range(n_layers)
        ])
        self.skip_proj = Conv1d(res_channels, res_channels, 1)
        self.output = Conv1d(res_channels, 1, 1)
        nn.init.zeros_(self.output.weight)  
        
    def forward(self, audio, diffusion_step, spectrogram):
        x = audio.unsqueeze(1) # (batch_size, 1, audio_sample)
        x = self.inp_proj(x)
        x = F.relu(x)
        diffusion_step = self.embedding(diffusion_step)
        
        spectrogram = self.spectrogram_upsampler(spectrogram)
        if not self.cond:
            spectrogram = None
            
        skip = 0
        for layer in self.layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip += skip_connection
            
        x = skip / np.sqrt(len(self.layers))
        x = self.skip_proj(x)
        x = F.relu(x)
        x = self.output(x)
        return x
    
class DDPM(nn.Module):
    def __init__(self, model, device, diff_steps=1000):
        super().__init__()
        self.model = model
        self.device = device
        self.noise_schedule = np.linspace(1e-4, 0.05, diff_steps)
        self.beta = self.noise_schedule
        self.alpha = 1 - self.beta
        self.alpha_bar = np.cumprod(self.alpha, 0)

    def forward(self, audio, t, noise):
        # xt = x0 * alpha_bar_sqrt + one_minus_alpha_bar * noise 

        alpha_bar = torch.tensor(self.alpha_bar[t], device = self.device, \
                                 dtype = torch.float32).unsqueeze(1)
        alpha_bar_sqrt = alpha_bar ** 0.5
        one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
        return alpha_bar_sqrt * audio + one_minus_alpha_bar * noise
    # def forward(self, audio, t, noise):
    #     # xt = x0 * alpha_bar_sqrt + one_minus_alpha_bar * noise 

    #     alpha_bar = torch.tensor(self.alpha_bar[t], device = self.device, \
    #                              dtype = torch.float32).unsqueeze(1)
    #     alpha_bar_sqrt = alpha_bar ** 0.5
    #     one_minus_alpha_bar = (1 - alpha_bar) ** 0.5

    #     alpha_bar_sqrt = alpha_bar_sqrt.unsqueeze(dim=1).unsqueeze(dim=1)
    #     one_minus_alpha_bar = one_minus_alpha_bar.unsqueeze(dim=1).unsqueeze(dim=1)

    #     print(alpha_bar_sqrt.shape, audio.shape, one_minus_alpha_bar.shape, noise.shape)

    #     return alpha_bar_sqrt * audio + one_minus_alpha_bar * noise
    
    def reverse(self, x_t, pred_noise, t):
        alpha_t = np.take(self.alpha, t)
        alpha_t_bar = np.take(self.alpha_bar, t)
        
        mean = (1 / (alpha_t ** 0.5)) * (
          x_t - (1 - alpha_t) / (1 - alpha_t_bar) ** 0.5 * pred_noise
        )
        sigma = np.take(self.beta, t) ** 0.5
        z = torch.randn_like(x_t)
        return mean + sigma * z
    
    def generate(self, spectrogram):
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(self.device)    
        x = torch.randn(spectrogram.shape[0], 256 * spectrogram.shape[-1], device=self.device)

        with torch.no_grad():
            for t in reversed(range(len(self.alpha))):
                t_tensor = torch.tensor(t, device=self.device).unsqueeze(0)
                pred_noise = self.model(x, t_tensor, spectrogram).squeeze(1)
                x = self.reverse(x, pred_noise, t)
        audio = torch.clamp(x, -1.0, 1.0)
        return audio