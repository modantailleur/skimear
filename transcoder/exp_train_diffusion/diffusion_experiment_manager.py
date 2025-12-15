#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:47:28 2022

@author: user
"""

import torch
import os
import torch.nn as nn
from pathlib import Path
import models as md
from data_generator import DcaseTask1Dataset, MelDataset, OutputsDataset, AudioOutputsDataset, WavDataset, DiffwaveAudioOutputsDataset, RandomAudioOutputsDataset
from diffusion_trainer import DiffusionTrainer, DiffusionTrainerDiffWave
from diffusion_evaluater import DiffusionEvaluater, DiffwaveEvaluater
from diffusion_vocoder import Vocoder
import utils.util as ut
import numpy as np
import yaml
from tqdm import tqdm
import torch.nn.functional as F
# import torch_tensorrt
# from denoising_diffusion_pytorch_modan import(Unet, MLP, 
#                                         GaussianDiffusion,
#                                         Trainer, Evaluater)
# from diffusers import UNet2DModel
from diffusion_models import UNet2DModel, ClassicCNN
from diffusers import DDPMScheduler, ScoreSdeVeScheduler
from diffusion_inference import DDPMInference, ScoreSdeVeInference
from metrics import AudioMetrics
from diffusion_diffwave import DiffWave
import pandas as pd
from voice_metrics import WerMetricW2V2, WerMetricWhisper, WerMetricSPT, STOIMetric, WerMetricSB
import soundfile as sf

def train_dl_model(setting_exp):
    """
    Trains a deep learning model based on the provided SettingExp experimental settings. 
    Different trainers are used depending of the deep learning model:
    - MelTrainer for transcoders that are trained directly by comparing the groundtruth and predicted Mel spectrogram
    - HybridTrainer for transcoders that are trained by comparing the groundtruth logits of the pre-trained models with 
        the logits of PANN that takes as input a predicted Mel spectrogram. The trainer is hybrid, which means 
        that it can also be trained partly on groundtruth Mel spectrograms if prop_logit is different than None
    - LogitTrainer for non-transcoding methods, that only involve training on logits

    The models are saved during the training, in the path corresponding to the "model_path" attribute of setting_exp

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.

    Returns:
        losses (dict), duration (float): A dict containing the losses during training, and the duration of training (in seconds).
    """

    if setting_exp.method != "diffwave":
        train_dataset = DcaseTask1Dataset(setting_exp.setting_data, subset='train', data_path=setting_exp.data_path)
    else:
        train_dataset = WavDataset(setting_exp.setting_data, subset='train', data_path=setting_exp.data_path)

        # model = md.ClassicCNN(kernel_size=setting_exp.kernel_size, dilation=setting_exp.dilation, 
        #         nb_layers=setting_exp.nb_layers,nb_channels=setting_exp.nb_channels, device=setting_exp.device)
        # trainer = HybridTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.method, setting_exp.model_name, 
        #                         train_dataset=train_dataset, 
        #                         # valid_dataset=valid_dataset, 
        #                         # eval_dataset=eval_dataset,
        #                         learning_rate=setting_exp.learning_rate)
    if setting_exp.method == 'transcoder':
        model = UNet2DModel(
            #WARNING: This is only for square matrices, need to find a solution in case not square
            sample_size=setting_exp.setting_data['mel']['n_freq'],  # the target image resolution
            in_channels=1,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 64, 128, 128, 256, 256),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ),
        )

        # model = ClassicCNN(sample_size=setting_exp.setting_data['mel']['n_freq'])

        trainer = DiffusionTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.method, setting_exp.model_name, 
                                train_dataset=train_dataset, 
                                # valid_dataset=valid_dataset, 
                                # eval_dataset=eval_dataset,
                                learning_rate=setting_exp.learning_rate,
                                gradient_accumulation_steps=setting_exp.gradient_accumulation_steps,
                                lr_warmup_steps=setting_exp.lr_warmup_steps,
                                diff_steps=setting_exp.diff_steps,
                                schedule=setting_exp.schedule,
                                )

        losses = trainer.train(batch_size=setting_exp.batch_size, epochs=setting_exp.epoch, iteration=setting_exp.iteration, device=setting_exp.device)
        trainer.save_model()

        # losses = trainer.train(batch_size=setting_exp.batch_size, epochs=setting_exp.epoch, iteration=setting_exp.iteration, device=setting_exp.device, gradient_accumulate_every=setting_exp.gradient_accumulate_every)
        # trainer.save_model()

    ####################
    # MT: diffusion with diffusers
    if setting_exp.method == 'diffusion':
        model = UNet2DModel(
            #WARNING: This is only for square matrices, need to find a solution in case not square
            sample_size=setting_exp.setting_data['mel']['n_freq'],  # the target image resolution
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(64, 64, 128, 128, 256, 256),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ),
        )

        # Optimize the UNet portion with Torch-TensorRT
        # model = torch.compile(
        #     model,
        #     backend="torch_tensorrt",
        #     options={
        #         "truncate_long_and_double": True,
        #         "precision": torch.float32,
        #     },
        #     dynamic=False,
        # )

        trainer = DiffusionTrainer(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.method, setting_exp.model_name, 
                                train_dataset=train_dataset, 
                                # valid_dataset=valid_dataset, 
                                # eval_dataset=eval_dataset,
                                learning_rate=setting_exp.learning_rate,
                                gradient_accumulation_steps=setting_exp.gradient_accumulation_steps,
                                lr_warmup_steps=setting_exp.lr_warmup_steps,
                                diff_steps=setting_exp.diff_steps,
                                schedule=setting_exp.schedule,
                                model_chkpt_name=setting_exp.model_chkpt_name, 
                                model_chkpt_setting_name=setting_exp.model_chkpt_setting_name,
                                )

        losses = trainer.train(batch_size=setting_exp.batch_size, epochs=setting_exp.epoch, iteration=setting_exp.iteration, device=setting_exp.device)
        trainer.save_model()
    
    if setting_exp.method == "diffwave":
        
        model = DiffWave(64, 12, setting_exp.setting_data["mel"]["n_freq"], diff_steps=setting_exp.diff_steps)
        trainer = DiffusionTrainerDiffWave(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.method, setting_exp.model_name, 
                                train_dataset=train_dataset, 
                                # valid_dataset=valid_dataset, 
                                # eval_dataset=eval_dataset,
                                learning_rate=setting_exp.learning_rate,
                                gradient_accumulation_steps=setting_exp.gradient_accumulation_steps,
                                lr_warmup_steps=setting_exp.lr_warmup_steps,
                                diff_steps=setting_exp.diff_steps,
                                schedule=setting_exp.schedule,
                                model_chkpt_name=setting_exp.model_chkpt_name, 
                                model_chkpt_setting_name=setting_exp.model_chkpt_setting_name,
                                )
        losses = trainer.train(batch_size=setting_exp.batch_size, epochs=setting_exp.epoch, iteration=setting_exp.iteration, device=setting_exp.device)
        trainer.save_model()

    return(losses, trainer.train_duration)   

def mel_dechunker(output_mel_path, output_mel_name, setting_data, eval_dataset):
    chunker = ut.AudioChunks(n=setting_data['mel']['n_time'], hop=setting_data['mel']['n_time_hop'])
    output_mels = np.memmap(output_mel_path+output_mel_name + '.dat', dtype=np.float64, mode ='r',shape=(eval_dataset.len_dataset, setting_data['mel']['n_freq'], setting_data['mel']['n_time']))
    output_mels_fname = np.memmap(output_mel_path+output_mel_name + 'fname.dat', dtype='S100', mode ='r',shape=(eval_dataset.len_dataset))
    output_mels_fname_idx = [int(s.decode('utf-8').split('___')[1]) for s in output_mels_fname]
    n_chunks_per_file = np.max(output_mels_fname_idx) + 1
    new_shape = (eval_dataset.len_dataset // n_chunks_per_file, setting_data['mel']['n_freq'], setting_data['mel']['n_time'] + setting_data['mel']['n_time_hop'] * (n_chunks_per_file - 1))
    output_concat_mels = np.memmap(output_mel_path+output_mel_name + 'aggregated.dat', dtype=np.float64,
                      mode='w+', shape=new_shape)

    for i in range(0, eval_dataset.len_dataset, n_chunks_per_file):
        batch_output_mels = output_mels[i:i+n_chunks_per_file]
        # Concatenate batch_output_mels along the time axis using concat_spec_with_hop
        concatenated_batch = chunker.concat_spec_with_hop(batch_output_mels)
        # Assign the concatenated batch to output_concate_mels
        output_concat_mels[i//n_chunks_per_file] = concatenated_batch
        output_concat_mels.flush()

def evaluate_dl_model(setting_exp):
    """
    Evaluates a deep learning model based on the provided SettingExp experimental settings. 
    Different evaluaters are used depending of the trained deep learning model:
    - DLTranscoderEvaluater for transcoders 
    - TSEvaluater for non-transcoding models (self and efficient nets)

    As the Mel data stored during this evaluation might be too heavy in memory, the results
    of the evaluation both in term of Mel data and logit data are stored directly inside the evaluaters, 
    and not in the exp_train_model/main_doce_training.py file.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
    """

    if setting_exp.model_name is not None:
        model_file_path = setting_exp.model_path + setting_exp.model_setting_name + '_settings.yaml'
    else:
        model_file_path = ''

    if os.path.exists(model_file_path):
        with open(model_file_path) as file:
            setting_model = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print("Model's training setting file does not exist.")
    
    if setting_exp.method == 'diffwave':
        eval_dataset = WavDataset(setting_exp.setting_data, subset='eval', data_path=setting_exp.data_path)
    else:
        eval_dataset = DcaseTask1Dataset(setting_exp.setting_data, subset='eval', data_path=setting_exp.data_path)    

    if setting_exp.method in ['oracle', 'pinv']:
        evaluater = DiffusionEvaluater(setting_exp.setting_data, None, None, None, setting_exp.method,
                            setting_exp.output_mel_path, setting_exp.output_mel_name,
                        eval_dataset=eval_dataset,
                        schedule = None, diff_steps=None, device=setting_exp.device)
        evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
        mel_dechunker(setting_exp.output_mel_path, setting_exp.output_mel_name, setting_exp.setting_data, eval_dataset)

    if setting_exp.method == 'transcoder':
        cnn_kernel_size = setting_model.get('cnn_kernel_size')
        cnn_dilation = setting_model.get('cnn_dilation')
        cnn_nb_layers = setting_model.get('cnn_nb_layers')
        cnn_nb_channels = setting_model.get('cnn_nb_channels')
        model = md.ClassicCNN(kernel_size=cnn_kernel_size, dilation=cnn_dilation, 
                        nb_layers=cnn_nb_layers, nb_channels=cnn_nb_channels, device=setting_exp.device)
        evaluater = DiffusionEvaluater(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.model_name, setting_exp.method,
                            setting_exp.output_audio_path,
                        eval_dataset=eval_dataset)
        evaluater.load_model(device=setting_exp.device)
        outputs = evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
        mel_dechunker(setting_exp.output_mel_path, setting_exp.output_mel_name, setting_exp.setting_data, eval_dataset)

    if setting_exp.method == 'diffusion':

        model = UNet2DModel(
            #WARNING: This is only for square matrices, need to find a solution in case not square
            sample_size=setting_model.get('sample_size'),  # the target image resolution
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=setting_model.get('block_out_channels'),  # the number of output channes for each UNet block
            down_block_types=setting_model.get('down_block_types'), 
            up_block_types=setting_model.get('up_block_types'),
        )

        evaluater = DiffusionEvaluater(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.model_name, setting_exp.method,
                            setting_exp.output_mel_path, setting_exp.output_mel_name,
                        eval_dataset=eval_dataset,
                        schedule = setting_model['schedule'], diff_steps=setting_model['diff_steps'], device=setting_exp.device, seed=setting_exp.seed)
        
        outputs = evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
        mel_dechunker(setting_exp.output_mel_path, setting_exp.output_mel_name, setting_exp.setting_data, eval_dataset)

    if setting_exp.method == 'diffwave':
        
        model = DiffWave(64, 12, setting_exp.setting_data["mel"]["n_freq"], diff_steps=setting_model['diff_steps'])
        # evaluater = DiffwaveEvaluater(setting_exp.setting_data, model, setting_exp.model_path, setting_exp.model_name, setting_exp.method,
        #                     setting_exp.output_mel_path, setting_exp.output_mel_name,
        #                 eval_dataset=eval_dataset,
        #                 schedule=None, diff_steps=None, device=setting_exp.device)
                        # schedule = setting_model['schedule'], diff_steps=setting_model['diff_steps'], device=setting_exp.device)
        
        evaluater = DiffwaveEvaluater(setting_exp.setting_data, setting_exp.setting_identifier, model, setting_exp.model_path, 
                                      setting_exp.model_name, setting_exp.method, setting_exp.output_audio_path, 
                                      setting_exp.output_audio_name, eval_dataset=eval_dataset,
                                        schedule=None, diff_steps=setting_exp.diff_steps, seed=setting_exp.seed, device=setting_exp.device)
        # outputs = evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)
        outputs = evaluater.evaluate(batch_size=setting_exp.batch_size, device=setting_exp.device)

def vocode_dl_model(setting_exp):
    """
    Vocodes using Mel spectrograms. 

    As the Mel data stored during this evaluation might be too heavy in memory, the results
    of the evaluation both in term of Mel data and logit data are stored directly inside the evaluaters, 
    and not in the exp_train_model/main_doce_training.py file.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
    """

    if setting_exp.model_name is not None:
        model_file_path = setting_exp.model_path + setting_exp.model_name + '_settings.yaml'
    else:
        model_file_path = ''
    
    if os.path.exists(model_file_path):
        with open(model_file_path) as file:
            setting_model = yaml.load(file, Loader=yaml.FullLoader)
    else:
        print("Model's training setting file does not exist.")
    
    eval_dataset = DcaseTask1Dataset(setting_exp.setting_data, subset='eval', data_path=setting_exp.data_path)      
    mel_dataset = MelDataset(eval_dataset, setting_exp.output_mel_path, setting_exp.output_mel_name, setting_exp.setting_data)
    vocoding = Vocoder(mel_dataset, "gomin", setting_exp.output_audio_path, setting_exp.output_audio_name, setting_exp.setting_data, setting_exp.setting_identifier, pann_prediction=True, output_pann_path=setting_exp.output_pann_path)
    vocoding.vocode(batch_size=3, device=setting_exp.device)
    vocoding = Vocoder(mel_dataset, "grifflim", setting_exp.output_audio_path, setting_exp.output_audio_name, setting_exp.setting_data, setting_exp.setting_identifier, pann_prediction=True, output_pann_path=setting_exp.output_pann_path)
    vocoding.vocode(batch_size=3, device=setting_exp.device)

def compute_metrics(setting_exp):
    """
    Calculate evaluation metrics for the current experiment.

    Args:
        setting_exp (object): A SettingExp instance from the exp_train_model/main_doce_training.py file, containing the experimental settings.
 
    Returns:
        metrics: A dictionary containing the calculated main metrics.
        others: A dictionary containing additional information such as predicted and grountruth labels of classifiers.
    """
    if setting_exp.metrics_type == 'voice':
        if 'diffwave' in setting_exp.method:
            wer_metric_w2v2 = WerMetricW2V2(device=setting_exp.device)
            wer_metric_whisper = WerMetricWhisper(device=setting_exp.device)
            wer_metric_spt = WerMetricSPT(device=setting_exp.device)
            wer_metric_ct = WerMetricSB(device=setting_exp.device, asr_short_name="ct")
            wer_metric_cr = WerMetricSB(device=setting_exp.device, asr_short_name="cr")
            stoi_metric = STOIMetric()
            outputs_dataset_gomin = DiffwaveAudioOutputsDataset(setting_data=setting_exp.setting_data, 
                                            audio_path = setting_exp.output_audio_path + setting_exp.output_audio_name,
                                            audio_path_oracle = setting_exp.output_audio_path + setting_exp.output_audio_name_oracle,
                                            audio_path_gt = setting_exp.audio_path_gt, evalset = setting_exp.evalset, sr=24000)

            outputs_dataloader = torch.utils.data.DataLoader(outputs_dataset_gomin, batch_size=1, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
            tqdm_it=tqdm(outputs_dataloader, desc='METRICS: Chunk {}/{}'.format(0,0))

            wers_w2v2_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_whisper_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_spt_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_ct_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_cr_gomin = torch.Tensor([]).to(setting_exp.device)
            stois = torch.Tensor([]).to(setting_exp.device)

            for (idx, audio, audio_gt, name, transcript) in tqdm_it:
                audio = audio.to(setting_exp.device)
                audio_gt = audio_gt.to(setting_exp.device)
                transcript = transcript[0]

                stoi = torch.Tensor([stoi_metric.calculate(audio.squeeze(dim=1).detach().cpu().numpy(), audio_gt.squeeze(dim=1).detach().cpu().numpy(), outputs_dataset_gomin.sr)]).unsqueeze(dim=1).to(setting_exp.device)
                stois = torch.cat((stois, stoi))

                audio_transcript_w2v2 = wer_metric_w2v2.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_w2v2.get_wer(transcript, audio_transcript_w2v2)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_w2v2_gomin = torch.cat((wers_w2v2_gomin, wer))
                
                audio_transcript_whisper = wer_metric_whisper.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_whisper.get_wer(transcript, audio_transcript_whisper)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_whisper_gomin = torch.cat((wers_whisper_gomin, wer))

                audio_transcript_spt = wer_metric_spt.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_spt.get_wer(transcript, audio_transcript_spt)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_spt_gomin = torch.cat((wers_spt_gomin, wer))

                audio_transcript_ct = wer_metric_ct.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_ct.get_wer(transcript, audio_transcript_ct)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_ct_gomin = torch.cat((wers_ct_gomin, wer))

                audio_transcript_cr = wer_metric_cr.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_cr.get_wer(transcript, audio_transcript_cr)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_cr_gomin = torch.cat((wers_cr_gomin, wer))

            metrics = {
                'stoi': stois.detach().cpu().numpy(),
                'wer_w2v2_gomin': wers_w2v2_gomin.detach().cpu().numpy(),
                'wer_whisper_gomin': wers_whisper_gomin.detach().cpu().numpy(),
                'wer_spt_gomin': wers_spt_gomin.detach().cpu().numpy(),
                'wer_ct_gomin': wers_ct_gomin.detach().cpu().numpy(),
                'wer_cr_gomin': wers_cr_gomin.detach().cpu().numpy(),
            }
            return(metrics, None)
        
        elif "random" in setting_exp.method:
            wer_metric_w2v2 = WerMetricW2V2(device=setting_exp.device)
            wer_metric_whisper = WerMetricWhisper(device=setting_exp.device)
            wer_metric_spt = WerMetricSPT(device=setting_exp.device)
            wer_metric_ct = WerMetricSB(device=setting_exp.device, asr_short_name="ct")
            wer_metric_cr = WerMetricSB(device=setting_exp.device, asr_short_name="cr")

            stoi_metric = STOIMetric()
            outputs_dataset_gomin = RandomAudioOutputsDataset(setting_data=setting_exp.setting_data, 
                                            audio_path = setting_exp.output_audio_path + setting_exp.output_audio_name,
                                            audio_path_oracle = setting_exp.output_audio_path + setting_exp.output_audio_name_oracle,
                                            audio_path_gt = setting_exp.audio_path_gt, evalset = setting_exp.evalset, vocoder="gomin", sr=24000)

            outputs_dataloader = torch.utils.data.DataLoader(outputs_dataset_gomin, batch_size=1, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
            tqdm_it=tqdm(outputs_dataloader, desc='METRICS: Chunk {}/{}'.format(0,0))

            wers_w2v2_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_whisper_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_spt_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_ct_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_cr_gomin = torch.Tensor([]).to(setting_exp.device)
            stois = torch.Tensor([]).to(setting_exp.device)

            for (idx, audio, audio_mel, audio_gt, name, transcript) in tqdm_it:
                audio = audio.type(setting_exp.dtype)
                audio_gt = audio_gt.type(setting_exp.dtype)

                audio = audio.to(setting_exp.device)
                audio_gt = audio_gt.to(setting_exp.device)
                transcript = transcript[0]
                
                stoi = torch.Tensor([stoi_metric.calculate(audio.squeeze(dim=1).detach().cpu().numpy(), audio_mel.squeeze(dim=1).detach().cpu().numpy(), outputs_dataset_gomin.sr)]).unsqueeze(dim=1).to(setting_exp.device)
                stois = torch.cat((stois, stoi))

                audio_transcript_w2v2 = wer_metric_w2v2.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_w2v2.get_wer(transcript, audio_transcript_w2v2)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_w2v2_gomin = torch.cat((wers_w2v2_gomin, wer))
                
                audio_transcript_whisper = wer_metric_whisper.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_whisper.get_wer(transcript, audio_transcript_whisper)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_whisper_gomin = torch.cat((wers_whisper_gomin, wer))

                audio_transcript_spt = wer_metric_spt.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_spt.get_wer(transcript, audio_transcript_spt)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_spt_gomin = torch.cat((wers_spt_gomin, wer))

                audio_transcript_ct = wer_metric_ct.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_ct.get_wer(transcript, audio_transcript_ct)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_ct_gomin = torch.cat((wers_ct_gomin, wer))

                audio_transcript_cr = wer_metric_cr.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_cr.get_wer(transcript, audio_transcript_cr)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_cr_gomin = torch.cat((wers_cr_gomin, wer))

            metrics = {
                'stoi': stois.detach().cpu().numpy(),
                'wer_w2v2_gomin': wers_w2v2_gomin.detach().cpu().numpy(),
                'wer_whisper_gomin': wers_whisper_gomin.detach().cpu().numpy(),
                'wer_spt_gomin': wers_spt_gomin.detach().cpu().numpy(),
                'wer_ct_gomin': wers_ct_gomin.detach().cpu().numpy(),
                'wer_cr_gomin': wers_cr_gomin.detach().cpu().numpy(),
            }

            return(metrics, None)
        else:
            wer_metric_w2v2 = WerMetricW2V2(device=setting_exp.device)
            wer_metric_whisper = WerMetricWhisper(device=setting_exp.device)
            wer_metric_spt = WerMetricSPT(device=setting_exp.device)
            wer_metric_ct = WerMetricSB(device=setting_exp.device, asr_short_name="ct")
            wer_metric_cr = WerMetricSB(device=setting_exp.device, asr_short_name="cr")

            stoi_metric = STOIMetric()
            outputs_dataset_gomin = AudioOutputsDataset(setting_data=setting_exp.setting_data, 
                                            audio_path = setting_exp.output_audio_path + setting_exp.output_audio_name,
                                            audio_path_oracle = setting_exp.output_audio_path + setting_exp.output_audio_name_oracle,
                                            audio_path_gt = setting_exp.audio_path_gt, evalset = setting_exp.evalset, vocoder="gomin", sr=24000)

            outputs_dataloader = torch.utils.data.DataLoader(outputs_dataset_gomin, batch_size=1, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
            tqdm_it=tqdm(outputs_dataloader, desc='METRICS: Chunk {}/{}'.format(0,0))

            wers_w2v2_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_whisper_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_spt_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_ct_gomin = torch.Tensor([]).to(setting_exp.device)
            wers_cr_gomin = torch.Tensor([]).to(setting_exp.device)
            stois = torch.Tensor([]).to(setting_exp.device)

            for (idx, audio, audio_mel, audio_gt, name, transcript) in tqdm_it:
                audio = audio.to(setting_exp.device)
                audio_gt = audio_gt.to(setting_exp.device)
                transcript = transcript[0]
                
                stoi = torch.Tensor([stoi_metric.calculate(audio.squeeze(dim=1).detach().cpu().numpy(), audio_mel.squeeze(dim=1).detach().cpu().numpy(), outputs_dataset_gomin.sr)]).unsqueeze(dim=1).to(setting_exp.device)
                stois = torch.cat((stois, stoi))

                audio_transcript_w2v2 = wer_metric_w2v2.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_w2v2.get_wer(transcript, audio_transcript_w2v2)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_w2v2_gomin = torch.cat((wers_w2v2_gomin, wer))
                
                audio_transcript_whisper = wer_metric_whisper.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_whisper.get_wer(transcript, audio_transcript_whisper)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_whisper_gomin = torch.cat((wers_whisper_gomin, wer))

                audio_transcript_spt = wer_metric_spt.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_spt.get_wer(transcript, audio_transcript_spt)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_spt_gomin = torch.cat((wers_spt_gomin, wer))

                audio_transcript_ct = wer_metric_ct.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_ct.get_wer(transcript, audio_transcript_ct)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_ct_gomin = torch.cat((wers_ct_gomin, wer))

                audio_transcript_cr = wer_metric_cr.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_cr.get_wer(transcript, audio_transcript_cr)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_cr_gomin = torch.cat((wers_cr_gomin, wer))

            outputs_dataset_grifflim = AudioOutputsDataset(setting_data=setting_exp.setting_data, 
                                            audio_path = setting_exp.output_audio_path + setting_exp.output_audio_name,
                                            audio_path_oracle = setting_exp.output_audio_path + setting_exp.output_audio_name_oracle,
                                            audio_path_gt = setting_exp.audio_path_gt, evalset = setting_exp.evalset, vocoder="grifflim", sr=24000)
            outputs_dataloader = torch.utils.data.DataLoader(outputs_dataset_grifflim, batch_size=1, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)
            tqdm_it=tqdm(outputs_dataloader, desc='METRICS: Chunk {}/{}'.format(0,0))
            wers_grifflim = torch.Tensor([]).to(setting_exp.device)

            for (idx, audio, audio_mel, audio_gt, name, transcript) in tqdm_it:
                audio = audio.to(setting_exp.device)
                audio_gt = audio_gt.to(setting_exp.device)
                transcript = transcript[0]

                audio_transcript_w2v2 = wer_metric_w2v2.get_transcript(audio, outputs_dataset_gomin.sr)
                wer = torch.Tensor([wer_metric_w2v2.get_wer(transcript, audio_transcript_w2v2)]).unsqueeze(dim=1).to(setting_exp.device)
                wers_grifflim = torch.cat((wers_grifflim, wer))

            metrics = {
                'stoi': stois.detach().cpu().numpy(),
                'wer_w2v2_gomin': wers_w2v2_gomin.detach().cpu().numpy(),
                'wer_whisper_gomin': wers_whisper_gomin.detach().cpu().numpy(),
                'wer_spt_gomin': wers_spt_gomin.detach().cpu().numpy(),
                'wer_ct_gomin': wers_ct_gomin.detach().cpu().numpy(),
                'wer_cr_gomin': wers_cr_gomin.detach().cpu().numpy(),
                'wer_w2v2_grifflim': wers_grifflim.detach().cpu().numpy(),
            }

            return(metrics, None)
    
