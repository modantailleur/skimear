
import librosa
import torch
import numpy as np
# from skimage.metrics import structural_similarity as ssim
import wave
import json
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

EPS = 1e-12

class AudioMetrics:
    def __init__(self, rate, device=torch.device("cpu")):
        self.rate = rate
        self.hop_length = int(rate / 100)
        self.n_fft = int(2048 / (44100 / rate))
        self.spectrogram_extractor = Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=1.0).to(device)
        self.device = device

    def lsd(self, est, target):
        est_spec = self.spectrogram_extractor(est)
        target_spec = self.spectrogram_extractor(target)

        lsd = torch.log10(target_spec**2 / ((est_spec + EPS) ** 2) + EPS) ** 2
        lsd = torch.mean(torch.mean(lsd, dim=3) ** 0.5, dim=2).squeeze(dim=1)
        return lsd
    
    def snr(self, est, target):
        """
        Calculate Signal-to-Noise Ratio (SNR) between two signals x and y.
        """
        # Calculate Euclidean norms
        y_norm_squared = torch.norm(target, p=2, dim=1)**2
        diff_norm_squared = torch.norm(est - target, p=2, dim=1)**2
        
        # Calculate SNR in dB
        snr = 10 * torch.log10(y_norm_squared / diff_norm_squared)

        return snr
