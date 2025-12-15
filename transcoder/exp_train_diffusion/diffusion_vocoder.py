#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:45:28 2022

@author: user
"""

import torch
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import utils.pinv_transcoder as pt
import utils.util as ut
from gomin.models import GomiGAN, DiffusionWrapper
from gomin.config import GANConfig, DiffusionConfig
import utils.bands_transform as bt
import librosa
from scipy.io.wavfile import write
from utils.util import create_folder
import torchaudio.functional as F
from utils.util import sort_labels_by_score
from diffusers import DDPMScheduler, ScoreSdeVeScheduler
from diffusers import DDPMPipeline, ScoreSdeVePipeline
from diffusion_inference import DDPMInference, ScoreSdeVeInference
import time
import os
from torchaudio.transforms import GriffinLim, InverseMelScale

class Vocoder:
    def __init__(self, mel_dataset, vocoder, output_audio_path, output_audio_name, setting_data, setting_identifier, output_pann_path = None, dtype=torch.FloatTensor, pann_prediction=True):
        """
        Initializes the TSEvaluater class. This saves inferences on logits 
        for teacher-student models that are not transcoders (effnet_b0,
        effnet_b7, self)

        Args:
        - setting_data: The setting data for the evaluation.
        - model: The pre-trained model.
        - models_path: The path to the model files.
        - model_name: The name of the model file.
        - outputs_path: The path to save the output files.
        - eval_dataset: The evaluation dataset.
        - dtype: The data type for the model (default: torch.FloatTensor).
        """

        self.setting_data = setting_data
        self.output_audio_path = output_audio_path
        self.output_audio_name = output_audio_name
        self.mel_dataset = mel_dataset
        self.vocoder = vocoder
        self.setting_identifier = setting_identifier
        self.dtype = dtype
        self.pann_prediction = pann_prediction 
        self.output_pann_path = output_pann_path

    def vocode(self, batch_size=8, device=torch.device("cpu")):
    
        self.inverse_transform = InverseMelScale(sample_rate=self.setting_data['mel']['sr'], n_stft=2048, n_mels=self.setting_data['mel']['n_freq'])
        self.griff_lim = GriffinLim(n_fft=1024,
                                win_length=1024,
                                hop_length=256)
        self.griff_lim.to(device)
        self.inverse_transform.to(device)
        #self.model.eval()
        self.batch_size = batch_size
        gomin_model = GomiGAN.from_pretrained(
        pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
        )
        gomin_model.eval()
        gomin_model.to(device)

        self.mel_dataloader = torch.utils.data.DataLoader(self.mel_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        #determined empirically for gomin mels, idk how to do else, needs to be patched
        tqdm_it=tqdm(self.mel_dataloader, desc='VOCODE: Chunk {}/{}'.format(0,0))

        for (idx, spec, audio_name) in tqdm_it:
                    spec = spec.type(self.dtype)
                    spec = spec.to(device)  

                    if self.vocoder == 'gomin':
                        create_folder(self.output_audio_path+'/'+self.setting_identifier+'/gomin')
                        create_folder(self.output_pann_path+'/'+self.setting_identifier+'/gomin')
                        with torch.no_grad():
                            wav_gomin = gomin_model(spec).squeeze(1).cpu().numpy()
                            for idx_audio, (wav_file, wav_name) in enumerate(zip(wav_gomin, audio_name)):
                                write(self.output_audio_path+self.setting_identifier + '/' + 'gomin/' + self.setting_identifier + "___gomin___" + wav_name+".wav", self.setting_data['mel']['sr'], wav_file)

                    if self.vocoder == 'grifflim':
                        create_folder(self.output_audio_path+'/'+self.setting_identifier+'/grifflim')
                        create_folder(self.output_pann_path+'/'+self.setting_identifier+'/grifflim')
                        #MT: energy compensation for the gomin normalisation of the spectrogram WARNING: remove if we use gomin
                        #needs to be change for the real compensation calculation (see gomin energy_compensation)
                        spec_grifflim = 10**((spec*100-100)/10) - 1e-10
                        spec_grifflim = spec_grifflim*20*20

                        for idx_audio, (cur_spec, wav_name) in enumerate(zip(spec_grifflim, audio_name)):
                             cur_spec = cur_spec.detach().cpu().numpy()
                             y_out = librosa.feature.inverse.mel_to_audio(cur_spec, sr=24000, n_fft=1024, hop_length=256, win_length=1024, fmin=0, fmax=12000)
                             write(self.output_audio_path+self.setting_identifier + '/' + 'grifflim/' + self.setting_identifier + "___grifflim___" + wav_name+".wav", self.setting_data['mel']['sr'], y_out)
