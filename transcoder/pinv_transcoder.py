#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 14:07:32 2022

@author: user
"""
import torch
import torch.utils.data
import torch.nn.functional as F
import librosa 

def pinv(x, tho_tr, mels_tr, reshape=None, dtype=torch.FloatTensor, device=torch.device("cpu"), input_is_db=True):
    """Convert a third octave spectrogram to a mel spectrogram using 
    a pseudo-inverse method.

    Parameters
    ----------
    x : torch.Tensor
        input third octave spectrogram of size (batch size, third octave 
                                                transform time bins, 
                                                third octave transform
                                                frequency bins)
        
    tho_tr : ThirdOctaveTransform instance
        third octave transform used as input (see ThirdOctaveTransform class)
    
    mels_tr : mels transform classes instance
        mels bands transform to match (see PANNMelsTransform for example)
    
    reshape : int
        if not set to None, will reshape the input tensor to match the given
        reshape value in terms of time bins. Simple copy of every time bin
        with some left and right extensions if 'reshape' is not a power of 
        two of the original 'time bins' value from the input tensor. 
        
    dtype : 
        data type to apply
        

    Returns
    -------
    x_mels_pinv : torch.Tensor
        mel spectrogram of size (batch size, mel transform time bins, 
                                 mel transform frequency bins)
    """

    #x_phi_inv: (2049,29)
    x_phi_inv = tho_tr.inv_tho_basis_torch

    #remove the log component from the input third octave spectrogram
    x_power = tho_tr.db_to_power_torch(x, device=device)

    #compensate the energy loss due to windowing
    x_power = tho_tr.compensate_energy_loss(x_power, mels_tr)
    
    # # The energy is not the same depending on the size of the temporal 
    # # window that is used. The larger the window, the bigger the energy 
    # # (because it sums on every frequency bin). A scale factor is needed
    # # in order to correct this. For example, with a window size of 1024, 
    # # the number of frequency bins after the fft will be 513 (N/2 +1). 
    # # With a window size of 4096, the number of frequency bins after the
    # # fft will be 2049 (N/2 +1). The scaling factor is then 2049/513.

    # scaling_factor = (1+mels_tr.flen/2) / (1+tho_tr.flen/2)
    # if mels_tr.window == 'hann':
    #     #hann window loses 50% of energy
    #     scaling_factor = scaling_factor * 0.5
    # else:
    #     raise Exception("Window unrecognised.") 

    # # scaling factor is raised to the power of 2 because it is supposed de be used 
    # # on an energy spectrum and x_power is a power spectrum
    # scaling_factor = scaling_factor ** 2
    
    # # scaling of the power spectrum to fit the scale of the stft used in the Mel transform
    # x_power = x_power * scaling_factor

    #put tensor to correct device
    x_phi_inv.to(x.dtype)
    x_power.to(x.dtype)

    x_phi_inv = x_phi_inv.to(device)
    x_power = x_power.to(device)

    #add one dimension to the pseudo-inverted matrix 
    #for the batch size of the input
    x_phi_inv = x_phi_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)

    #permute third octave transform time bins and third octave transform 
    # frequency bins to allow matrix multiplication
    x_power = torch.permute(x_power, (0,2,1))

    if tho_tr.tho_freq:
        x_spec_pinv = torch.matmul(x_phi_inv, x_power)
        
    else:
        x_mel_inv = librosa.filters.mel(sr=32000, n_fft=4096, fmin=50, fmax=14000, n_mels=64)
        x_mel_inv = x_mel_inv.T
        x_mel_inv = torch.from_numpy(x_mel_inv)
        x_mel_inv = x_mel_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)
        x_spec_pinv = torch.matmul(x_mel_inv, x_power)
        #x_spec_pinv = x_power

    #eventually reshape time dimension to match 'reshape' value
    if reshape:
        x_spec_pinv = F.interpolate(x_spec_pinv, size=reshape, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
        #permute again to have the dimensions in correct order 
        x_spec_pinv = torch.permute(x_spec_pinv, (0,2,1))

    else: 
        x_spec_pinv = x_spec_pinv

    # if mels_tr.sr != tho_tr.sr:
    #     x_spec_pinv = F.interpolate(x_spec_pinv, size=int(1+mels_tr.flen/2), scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
    if input_is_db:
        #from power spectrogram to mel spectrogram
        x_mels_pinv = mels_tr.power_to_mels(x_spec_pinv)
    else:
        if mels_tr.name == "yamnet":
            raise Exception("It is not possible to train regular Mel spectrogram for YamNet (as opposed to logMel Spectrogram") 
        if mels_tr.name == "pann":
            x_mels_pinv = mels_tr.power_to_mels_no_db(x_spec_pinv)
    
    x_mels_pinv = x_mels_pinv.squeeze(0)

    return(x_mels_pinv)

def pinv_mel_to_mel(x, mels_tr_input, mels_tr_output, reshape=None, device=torch.device("cpu"), input_is_db=True):
    """Convert a third octave spectrogram to a mel spectrogram using 
    a pseudo-inverse method.

    Parameters
    ----------
    x : torch.Tensor
        input third octave spectrogram of size (batch size, third octave 
                                                transform time bins, 
                                                third octave transform
                                                frequency bins)
        
    tho_tr : ThirdOctaveTransform instance
        third octave transform used as input (see ThirdOctaveTransform class)
    
    mels_tr : mels transform classes instance
        mels bands transform to match (see PANNMelsTransform for example)
    
    reshape : int
        if not set to None, will reshape the input tensor to match the given
        reshape value in terms of time bins, doing a linear interpolation on
        the time axis.
        
    device : str
        set the data to the correct device (gpu or cpu)
    
    input_is_db: bool
        if set to true, will consider that the given input is in dB scale.

    Returns
    -------
    x_mels_pinv : torch.Tensor
        mel spectrogram of size (batch size, mel transform time bins, 
                                 mel transform frequency bins)
    """

    #x_phi_inv: (2049,29)
    x_phi_inv = mels_tr_input.inv_mel_basis

    #remove the log component from the input third octave spectrogram
    x_power = mels_tr_input.db_to_power(x)
    
    #compensate the energy loss due to windowing
    x_power = mels_tr_input.compensate_energy_loss(x_power, mels_tr_output)
    
    #put tensor to correct device
    x_phi_inv.to(x.dtype)
    x_power.to(x.dtype)

    x_phi_inv = x_phi_inv.to(device)
    x_power = x_power.to(device)

    #add one dimension to the pseudo-inverted matrix 
    #for the batch size of the input
    x_phi_inv = x_phi_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)

    #permute third octave transform time bins and third octave transform 
    # frequency bins to allow matrix multiplication
    # x_power = torch.permute(x_power, (0,2,1))
    x_phi_inv = torch.permute(x_phi_inv, (0, 2, 1))

    print('AAAAAA')
    print(x_power.device)
    print(x_phi_inv.device)

    x_spec_pinv = torch.matmul(x_phi_inv, x_power)
    
    print('BBBB')
    print(x_spec_pinv.shape)

    x_spec_pinv = F.interpolate(x_spec_pinv, size=137, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
    #permute again to have the dimensions in correct order 
    x_spec_pinv = torch.permute(x_spec_pinv, (0,2,1))
    x_spec_pinv = F.interpolate(x_spec_pinv, size=2049, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)

    #from power spectrogram to mel spectrogram

    print('OOOOOOOOOOOo')
    print(x_spec_pinv.device)

    x_mels_pinv = mels_tr_output.power_to_mels(x_spec_pinv)
    
    x_mels_pinv = x_mels_pinv.squeeze(0)

    return(x_mels_pinv)


def pinv_gomin(x, tho_tr, mels_tr, reshape=None, device=torch.device("cpu"), input_is_db=True):
    """Convert a third octave spectrogram to a mel spectrogram using 
    a pseudo-inverse method.

    Parameters
    ----------
    x : torch.Tensor
        input third octave spectrogram of size (batch size, third octave 
                                                transform time bins, 
                                                third octave transform
                                                frequency bins)
        
    tho_tr : ThirdOctaveTransform instance
        third octave transform used as input (see ThirdOctaveTransform class)
    
    mels_tr : mels transform classes instance
        mels bands transform to match (see PANNMelsTransform for example)
    
    reshape : int
        if not set to None, will reshape the input tensor to match the given
        reshape value in terms of time bins, doing a linear interpolation on
        the time axis.
        
    device : str
        set the data to the correct device (gpu or cpu)
    
    input_is_db: bool
        if set to true, will consider that the given input is in dB scale.

    Returns
    -------
    x_mels_pinv : torch.Tensor
        mel spectrogram of size (batch size, mel transform time bins, 
                                 mel transform frequency bins)
    """

    #x_phi_inv: (2049,29)
    x_phi_inv = tho_tr.inv_tho_basis_torch

    #remove the log component from the input third octave spectrogram
    x_power = tho_tr.db_to_power_torch(x, device=device)
    
    #compensate the energy loss due to windowing
    x_power = tho_tr.compensate_energy_loss(x_power, mels_tr)
    # #compensate loss due to the frequency rate (24000/32000)
    # x_power = (24000/32000)*x_power
    # #compensate loss due to the diminution of frequency bins (because of the frequency rate, also)
    # x_power = (513/2049)*x_power
    #x_power = 0.01*x_power

    #put tensor to correct device
    x_phi_inv.to(x.dtype)
    x_power.to(x.dtype)

    x_phi_inv = x_phi_inv.to(device)
    x_power = x_power.to(device)

    #add one dimension to the pseudo-inverted matrix 
    #for the batch size of the input
    x_phi_inv = x_phi_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)

    #permute third octave transform time bins and third octave transform 
    # frequency bins to allow matrix multiplication
    x_power = torch.permute(x_power, (0,2,1))

    #get the time size tho to compensate the energy loss due to time
    time_size_tho = x_power.shape[2]

    #MT - test
    test_scale_factor = 1

    print('MMMMMMMMm')
    print(x_power.shape)
    print(torch.mean(x_power))

    if tho_tr.tho_freq:
        x_spec_pinv = torch.matmul(x_phi_inv, x_power)
        
    else:
        x_mel_inv = librosa.filters.mel(sr=32000, n_fft=4096, fmin=50, fmax=14000, n_mels=64)
        x_mel_inv = x_mel_inv.T
        x_mel_inv = torch.from_numpy(x_mel_inv)
        x_mel_inv = x_mel_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)
        x_spec_pinv = torch.matmul(x_mel_inv, x_power)

    print('LLLLLLLLL')
    print(x_spec_pinv.shape)
    print(torch.mean(x_spec_pinv))

    #get the freq size of the fb spectro to compensate the energy loss due to the sr changing
    freq_size_fb = x_spec_pinv.shape[1]

    #eventually reshape time dimension to match 'reshape' value
    if reshape:
        x_spec_pinv = F.interpolate(x_spec_pinv, size=reshape, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
        #permute again to have the dimensions in correct order 
        x_spec_pinv = torch.permute(x_spec_pinv, (0,2,1))

        #compensate energy loss due to time interpolation (time is at index 1 in x_spec_pinv)
        x_spec_pinv = x_spec_pinv * time_size_tho / x_spec_pinv.shape[1]
        test_scale_factor =  test_scale_factor * time_size_tho / x_spec_pinv.shape[1]
        print('ZZZZZZZZ')
        print(x_spec_pinv.shape)
        print(torch.mean(x_spec_pinv))

        x_spec_pinv = F.interpolate(x_spec_pinv, size=513, scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)

        #compensate energy loss due to sr changing (from 2049 in freq to 513) (freq is at index 2 in x_spec_pinv)
        x_spec_pinv = x_spec_pinv * freq_size_fb / x_spec_pinv.shape[2]
        test_scale_factor = test_scale_factor * freq_size_fb / x_spec_pinv.shape[2]
        print('WWWWWWWWWWWWw')
        print(x_spec_pinv.shape)
        print(torch.mean(x_spec_pinv))

    else: 
        x_spec_pinv = x_spec_pinv

    print('NNNNNNNNn')
    print(torch.mean(x_spec_pinv))

    print('PPPPPPPPPPPPPPPPPPPPPPPPPP')
    print(test_scale_factor)

    #MT: just for test, no real signification
    x_spec_pinv = x_spec_pinv 

    if input_is_db:
        #from power spectrogram to mel spectrogram
        x_mels_pinv = mels_tr.power_to_mels(x_spec_pinv)
    else:
        if mels_tr.name == "yamnet":
            raise Exception("It is not possible to train regular Mel spectrogram for YamNet (as opposed to logMel Spectrogram") 
        if mels_tr.name == "pann":
            x_mels_pinv = mels_tr.power_to_mels_no_db(x_spec_pinv)

    x_mels_pinv = x_mels_pinv.squeeze(0)

    return(x_mels_pinv)

def pinv_fb(x, tho_tr, target_tr, reshape=None, device=torch.device("cpu"), input_is_db=True):
    """Convert a third octave spectrogram to a mel spectrogram using 
    a pseudo-inverse method.

    Parameters
    ----------
    x : torch.Tensor
        input third octave spectrogram of size (batch size, third octave 
                                                transform time bins, 
                                                third octave transform
                                                frequency bins)
        
    tho_tr : ThirdOctaveTransform instance
        third octave transform used as input (see ThirdOctaveTransform class)
    
    mels_tr : mels transform classes instance
        mels bands transform to match (see PANNMelsTransform for example)
    
    reshape : int
        if not set to None, will reshape the input tensor to match the given
        reshape value in terms of time bins, doing a linear interpolation on
        the time axis.
        
    device : str
        set the data to the correct device (gpu or cpu)
    
    input_is_db: bool
        if set to true, will consider that the given input is in dB scale.

    Returns
    -------
    x_mels_pinv : torch.Tensor
        mel spectrogram of size (batch size, mel transform time bins, 
                                 mel transform frequency bins)
    """

    #x_phi_inv: (2049,29)
    x_phi_inv = tho_tr.inv_tho_basis_torch

    #remove the log component from the input third octave spectrogram
    x_power = tho_tr.db_to_power_torch(x, device=device)
    
    #compensate the energy loss due to windowing
    x_power = tho_tr.compensate_energy_loss(x_power, target_tr)
    
    #put tensor to correct device
    x_phi_inv.to(x.dtype)
    x_power.to(x.dtype)

    x_phi_inv = x_phi_inv.to(device)
    x_power = x_power.to(device)

    #add one dimension to the pseudo-inverted matrix 
    #for the batch size of the input
    x_phi_inv = x_phi_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)

    #permute third octave transform time bins and third octave transform 
    # frequency bins to allow matrix multiplication
    x_power = torch.permute(x_power, (0,2,1))

    if tho_tr.tho_freq:
        x_spec_pinv = torch.matmul(x_phi_inv, x_power)
        
    else:
        x_mel_inv = librosa.filters.mel(sr=32000, n_fft=4096, fmin=50, fmax=14000, n_mels=64)
        x_mel_inv = x_mel_inv.T
        x_mel_inv = torch.from_numpy(x_mel_inv)
        x_mel_inv = x_mel_inv.unsqueeze(0).repeat(x_power.shape[0],1,1)
        x_spec_pinv = torch.matmul(x_mel_inv, x_power)

    #eventually reshape time dimension to match 'reshape' value
    if reshape:
        #reshape on time axis
        x_spec_pinv = F.interpolate(x_spec_pinv, size=reshape[0], scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
        #permute again to have the dimensions in correct order 
        x_spec_pinv = torch.permute(x_spec_pinv, (0,2,1))
        #reshape on frequency axis
        x_spec_pinv = F.interpolate(x_spec_pinv, size=reshape[1], scale_factor=None, mode='linear', align_corners=None, recompute_scale_factor=None, antialias=False)
    else: 
        x_spec_pinv = x_spec_pinv

    if input_is_db:
        #from power spectrogram to mel spectrogram
        x_mels_pinv = target_tr.power_to_db(x_spec_pinv)
    else:
        x_mels_pinv = x_spec_pinv
        
    x_mels_pinv = x_mels_pinv.squeeze(0)

    return(x_mels_pinv)