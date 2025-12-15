from models import GomiGAN, DiffusionWrapper
from config import GANConfig, DiffusionConfig
import librosa
from scipy.io.wavfile import write
import torch
import numpy as np

model = GomiGAN.from_pretrained(
  pretrained_model_path="gan_state_dict.pt", **GANConfig().__dict__
)

file_name = "street_traffic-milan-1094-42444-a_2s"
#file_name = "voice_countdown_2s"
#file_name = "airport-lisbon-1000-40000-a_2s"
#file_name = "birds_2s"
waveform, sr = librosa.load('./test_files/'+file_name+'.wav', sr=24000)
waveform_white_noise =   2 * torch.rand((1, 48341)) - 1
waveform_tone = torch.from_numpy(librosa.tone(5000, duration=2)).unsqueeze(dim=0).type(torch.FloatTensor)

print('WWWWWWWWWW')
print(torch.max(waveform_tone))
print(torch.min(waveform_tone))

#8512
#8768
#waveform = np.pad(waveform, (0, 8767))

waveform = torch.Tensor(waveform).unsqueeze(0)

print(waveform.shape)

assert waveform.ndim == 2
# n_mels: int = 128,
# sample_rate: int = 24000,
# win_length: int = 1024,
# hop_length: int = 256,
melspec = model.prepare_melspectrogram(waveform_tone)

print('XXXXXXXXXX')
print(torch.max(melspec))
print(torch.min(melspec))
print(torch.mean(melspec))
print('BBBBBBBB')
print(melspec.shape)
# To reconstruct wavefrom from mel-spectrogram, run:
assert melspec.ndim == 3
waveform_gomin = model(melspec)
print(waveform_gomin.shape)
print('AAAAAAAAAAAAA')
melspec_input = melspec*100 -100
melspec_input = 10**(melspec_input.numpy()/10)
waveform_griff = librosa.feature.inverse.mel_to_audio(melspec_input, sr=24000, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=12000)
waveform_griff = waveform_griff * 20
print(waveform_griff.shape)

write(file_name+"_original.wav", 24000, waveform[0].detach().cpu().numpy())
write(file_name+"_generated_gomin.wav", 24000, waveform_gomin[0][0].detach().cpu().numpy())
write(file_name+"_generated_griff.wav", 24000, waveform_griff[0])