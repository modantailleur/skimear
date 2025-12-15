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
from diffusion_diffwave import DDPM

class DiffusionEvaluater:
    def __init__(self, setting_data, model, models_path, model_name, method,
                 output_mel_path, output_mel_name, eval_dataset, schedule, diff_steps, device, seed=None, dtype=torch.FloatTensor):
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

        self.output_mel_path = output_mel_path
        self.output_mel_name = output_mel_name
        self.setting_data = setting_data
        self.dtype = dtype
        self.seed = seed

        self.eval_dataset = eval_dataset

        self.models_path = models_path
        self.model_name = model_name
        self.method = method

        self.model = model
        if self.model is not None:
            self.load_model(device=device)

        #self.lr = 1e-3
        #self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        # self.diffusion_inference = diffusion_inference
        if schedule == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=diff_steps)
            self.diffusion_inference = ScoreSdeVeInference(self.model, self.noise_scheduler,diff_steps)

        if schedule == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=diff_steps, beta_schedule='sigmoid')
            self.diffusion_inference = DDPMInference(self.model, self.noise_scheduler,diff_steps)

        if self.seed is not None:
            print('SETTING TORCH MANUAL SEED TO : ', self.seed)
            torch.manual_seed(self.seed)
        
    def evaluate(self, batch_size=8, device=torch.device("cpu")):
    
        #self.model.eval()
        self.batch_size = batch_size
        gomin_model = GomiGAN.from_pretrained(
        pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
        )
        gomin_model.eval()

        # chunker = ut.AudioChunks(n=128, hop=24)
        #determined empirically for gomin mels, idk how to do else, needs to be patched
        # num_chunks, troncated = chunker.calculate_num_chunks(128*7)
        # prev_wav = 'None'
        # count = 0

        # spec_L = np.empty([0, 513, round(128*self.chunk_len)])
        # spec_L = np.empty([0, 128, 128])
        # tho_spec_L = spec_L
        # out_L = spec_L

        #for gomin
        # wav_gomin_original_L = np.empty([0])
        # wav_gomin_out_L = wav_gomin_original_L
        # wav_gomin_pinv_L = wav_gomin_original_L
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION')
        
        # save the output of the model in a .dat file. This avoids havind memory issues
        output_mels = np.memmap(self.output_mel_path + self.output_mel_name + '.dat', dtype=np.float64,
                      mode='w+', shape=(self.eval_dataset.len_dataset, self.setting_data['mel']['n_freq'], self.setting_data['mel']['n_time']))
        output_mels_fname = np.memmap(self.output_mel_path + self.output_mel_name + 'fname.dat', dtype='S100',
                mode='w+', shape=(self.eval_dataset.len_dataset))
        with torch.no_grad():
            for (idx, spec, tho_spec, _, audio_name, _, _) in tqdm_it:

                wav_name = audio_name[0].decode('UTF-8')
                wav_name = wav_name.split('___')[0]
            
                # if prev_wav == 'None':
                #     prev_wav = wav_name
                                
                spec = spec.to(device)
                tho_spec = tho_spec.to(device)

                if self.method == 'diffusion':
                    spec = torch.unsqueeze(spec, dim=1)
                    tho_spec = torch.unsqueeze(tho_spec, dim=1)
                    #MT: old diffusion version
                    # out = self.model.sample(tho_spec, batch_size=self.batch_size)
                    tho_spec_diffusion = tho_spec * 2 - 1
                    start_time = time.time()
                    out = self.diffusion_inference.inference(tho_spec_diffusion, device=device)
                    end_time = time.time()
                    #MT: to remove ? already in self.diffusion_inference.inference
                    # out = (out + 1) / 2
                    out = out.squeeze(dim=1)
                elif self.method == 'oracle':
                    out = spec
                elif self.method == 'pinv':
                    out = tho_spec
                elif self.method == 'transcoder':
                    out = self.model(tho_spec.squeeze(dim=1))
                    out = out.unsqueeze(dim=1)

                output_mels[idx, :, :] = out.detach().cpu().numpy()
                output_mels_fname[idx] = audio_name
                output_mels.flush()
                output_mels_fname.flush()
        
    def load_model(self, device):
        self.model = self.model.to(device)
        load_path = self.models_path + self.model_name +'.pt'
        print(f'LOADING MODEL : {load_path}')
        state_dict = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state_dict)

class DiffwaveEvaluater:
    def __init__(self, setting_data, setting_identifier, model, models_path, model_name, method,
                 output_audio_path, output_audio_name, eval_dataset, schedule, diff_steps, device, seed=None, dtype=torch.FloatTensor):
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

        self.output_audio_path = output_audio_path
        self.output_audio_name = output_audio_name
        self.setting_data = setting_data
        self.setting_identifier = setting_identifier
        self.dtype = dtype
        self.seed = seed

        self.eval_dataset = eval_dataset

        self.models_path = models_path
        self.model_name = model_name
        self.method = method

        self.model = model
        if self.model is not None:
            print('LOADING MODEL')
            self.load_model(device=device)
        else:
            print('MODEL NOT LOADED')
        #self.lr = 1e-3
        #self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        
        # self.diffusion_inference = diffusion_inference
        if schedule == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=diff_steps)
            self.diffusion_inference = ScoreSdeVeInference(self.model, self.noise_scheduler,diff_steps)

        if schedule == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=diff_steps, beta_schedule='sigmoid')
            self.diffusion_inference = DDPMInference(self.model, self.noise_scheduler,diff_steps)
        self.diff_steps = diff_steps

        if self.seed is not None:
            print('SETTING TORCH MANUAL SEED TO : ', self.seed)
            torch.manual_seed(self.seed)
            
    def evaluate(self, batch_size=8, device=torch.device("cpu")):
        self.model.to(device)
        self.diff_method = DDPM(self.model, device=device, diff_steps=self.diff_steps)

        #self.model.eval()
        self.batch_size = batch_size
        gomin_model = GomiGAN.from_pretrained(
        pretrained_model_path="gomin/gan_state_dict.pt", **GANConfig().__dict__
        )
        gomin_model.eval()
                
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
        tqdm_it=tqdm(self.eval_dataloader, desc='EVALUATION')
        create_folder(self.output_audio_path+'/'+self.setting_identifier)

        with torch.no_grad():
            for (idx, _, tho_spec, _, audio_name) in tqdm_it:
                     
                tho_spec = tho_spec.to(device)

                if self.method == 'diffwave':
                    tho_spec = tho_spec * 2 - 1
                    tho_spec = torch.unsqueeze(tho_spec, dim=1)
                    tho_spec = tho_spec.squeeze(dim=1).squeeze(dim=1)
                    #generate is expecting a 3D tensor (batch, channels, time)
                    out = self.diff_method.generate(tho_spec)

                    for idx_audio, (wav_file, wav_name) in enumerate(zip(out, audio_name)):
                        wav_name_decoded = wav_name.decode('UTF-8')
                        wav_name_decoded = wav_name_decoded.split('___')[0]
                        # write(self.output_audio_path+self.setting_identifier + '/' + 'gomin/' + self.setting_identifier + "___gomin___" + wav_name+".wav", self.setting_data['mel']['sr'], wav_file)

                        # Construct the output path
                        output_dir = os.path.join(self.output_audio_path, self.setting_identifier.replace("step=eval", "step=vocode"))
                        output_path = os.path.join(output_dir, f"{self.setting_identifier.replace('step=eval', 'step=vocode')}___{wav_name_decoded}.wav")
                        
                        # Create the directory if it does not exist
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save the audio file
                        write(output_path, self.setting_data['mel']['sr'], wav_file.detach().cpu().numpy())      
                          
    def load_model(self, device):
        self.model = self.model.to(device)
        load_path = self.models_path + self.model_name +'.pt'
        print(f'LOADING MODEL : {load_path}')
        state_dict = torch.load(load_path, map_location=device)
        self.model.load_state_dict(state_dict)
