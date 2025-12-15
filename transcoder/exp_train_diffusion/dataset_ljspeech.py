import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa
from diffusion_utils import create_folder, get_filename, create_logging, float32_to_int16, pad_or_truncate, read_metadata
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import sys
import torch
# Add the parent directory of the project directory to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
import utils.bands_transform as bt
import utils.pinv_transcoder as pt
import matplotlib.pyplot as plt
import utils.util as ut
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import yaml

np.random.seed(0)

def plot_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False):
    if vmin==None:
        vmin = torch.min(x_m)
    if vmax==None:
        vmax = torch.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    extlmax = 1

    plt.figure(figsize=(8, 5))

    if diff:
        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Power differences (dB)', rotation=90, labelpad=15)
    else:

        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Power (dB)', rotation=90, labelpad=15)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    if save:
        plt.savefig('figures_spectrograms/'+name)
    plt.show()

def train_test__valid_split_list(x, train_ratio, valid_ratio, eval_ratio):
    np.random.shuffle(x)
    train_idx = round(train_ratio*len(x))
    valid_idx = train_idx + round(valid_ratio*len(x))
    #eval_idx = valid_idx + round(eval_ratio*len(x))
    if train_ratio+valid_ratio+eval_ratio == 1:
        train = x[:train_idx]
        valid = x[train_idx:valid_idx]
        eval = x[valid_idx:]
    else:
        print('train_ratio, valid_ratio and eval_ratio must sum to 1')
    return(train, valid, eval)

def split_dataset(dataset, cities_train, cities_valid, cities_eval):
    """
    Splits the dataset into train, valid, and eval datasets based on the specified cities.

    Args:
        dataset (list): List of data containing the dataset.
        cities_train (list): List of cities for the train dataset.
        cities_valid (list): List of cities for the valid dataset.
        cities_eval (list): List of cities for the eval dataset.

    Returns:
        tuple: A tuple containing the train, valid, and eval datasets lists based on the specified cities.
    """
    dataset_f = dataset
    
    if cities_train:
        train_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_train]]
    
    if cities_valid:
        valid_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_valid]] 
    
    if cities_eval:
        eval_dataset = [x for x in dataset_f if True in
                    [y == x[2] for y in cities_eval]] 
        
    return(train_dataset, valid_dataset, eval_dataset)

def urban_dataset(dataset, devices_keep, classes_keep):
    """
    Filters the dataset based on the specified devices and classes.

    Args:
        dataset (list): List of data containing the dataset.
        devices_keep (list): List of devices to keep in the filtered dataset.
        classes_keep (list): List of classes to keep in the filtered dataset.

    Returns:
        list: The filtered dataset based on the specified devices and classes.

    Examples:
        >>> dataset = [...]
        >>> devices_keep = ['a', 's1']
        >>> classes_keep = ['metro', 'airport']
        >>> filtered_dataset = urban_dataset(dataset, devices_keep, classes_keep)
    """

    dataset_f = dataset
    
    if devices_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[-1] for y in devices_keep]]
    if classes_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[1] for y in classes_keep]]
    return(np.array(dataset_f))

def full_dataset(dataset, devices_keep):
    """
    Filters the dataset based on the specified devices.

    Args:
        dataset (list): List of data containing the dataset.
        devices_keep (list): List of devices to keep in the filtered dataset.

    Returns:
        list: The filtered dataset based on the specified devices.

    Examples:
        >>> dataset = [...]
        >>> devices_keep = ['a', 's1']
        >>> filtered_dataset = full_dataset(dataset, devices_keep)
    """
    dataset_f = dataset
    
    if devices_keep:
        dataset_f = [x for x in dataset_f if True in
                    [y == x[-1] for y in devices_keep]]
    
    return(np.array(dataset_f))

def pack_waveforms_to_hdf5(args):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    sample_rate = 32000

    df = pd.read_csv(args.audio_dataset + "/" + "metadata_with_split.csv", sep='|', index_col="file")

    #random shuffle so that the samples are select are random

    train_dataset = df[df["split"]=="train"].index.values
    train_dataset = ["wavs/"+item+".wav" for item in train_dataset]
    eval_dataset = df[df["split"]=="eval"].index.values
    eval_dataset = ["wavs/"+item+".wav" for item in eval_dataset]

    print('LENGTH OF EACH DATASET')
    total_length = len(train_dataset) + len(eval_dataset)
    train_percentage = len(train_dataset) / total_length
    eval_percentage = len(eval_dataset) / total_length

    print(f'train: {len(train_dataset)}, {round(train_percentage, 4)}%')
    print(f'eval: {len(eval_dataset)}, {round(eval_percentage, 4)}%')

    create_h5_file(args, train_dataset, "train", sample_rate)
    create_h5_file(args, eval_dataset, "eval", sample_rate)

# OLD ONE: used for the best trained model as to date 02/02/2023
# def create_h5_file(args, dataset, split, sr, chunk_len=1.36, hop_len=0.25):

#this one takes full advantage of the 10s audio, leaving just a few digits behing
# def create_h5_file(args, dataset, split, sr, chunk_len=1.36, hop_len=1.2342):
def create_h5_file(args, dataset, split, sr, chunk_len=1.36, hop_len=1.2342):

    '''
    chunk_len: the length of the chunk of audio that will be analysed by the network (in seconds)
    hop_len: the length of the hop (in seconds)
    '''
    audios_num = len(dataset)
    n_tho = 20
    ref_freq = 9
    flen = 4096
    hlen = 4000
    tho_tr = bt.ThirdOctaveTransform(sr=sr, flen=4096, hlen=4000, refFreq=ref_freq, n_tho=n_tho, db_delta=0)
    fb_tr = bt.FineBandsTransform(sr=sr, flen=1024, hlen=250, window='hann')
    #MT: added
    mels_tr = bt.GominMelsTransform()
    # mels_tr_test = bt.PANNMelsTransform()

    sr_target = 24000
    if "wav" in args.dataset_type:
        chunker_24k = ut.AudioChunks(n=round(sr_target*10), hop=round(sr_target*10))
        chunker_32k = ut.AudioChunks(n=round(sr*10), hop=round(sr*10))
        num_chunks, audio_truncated = chunker_24k.calculate_num_chunks(sr_target*10)
        audios_n_num = audios_num * num_chunks
    else:
        chunker_24k = ut.AudioChunks(n=round(sr_target*chunk_len), hop=round(sr_target*hop_len))
        chunker_32k = ut.AudioChunks(n=round(sr*chunk_len), hop=round(sr*hop_len))
        num_chunks, audio_truncated = chunker_24k.calculate_num_chunks(sr_target*10)
        audios_n_num = audios_num * num_chunks

    print(num_chunks)
    #if the audio is troncated (the size of chunks and the hop size doesn't allow to match the total size of the wav file),
    #then the troncated audio is also used for the training and validation dataset
    # if split != 'eval' or args.dataset_content == 'waveform':
    if split != 'eval':
        chunker_24k = ut.AudioChunks(n=round(sr_target*chunk_len), hop=round(sr_target*chunk_len))
        chunker_32k = ut.AudioChunks(n=round(sr*chunk_len), hop=round(sr*chunk_len))
        num_chunks, audio_truncated = chunker_24k.calculate_num_chunks(sr_target*10)
        audios_n_num = audios_num * num_chunks

    if audio_truncated:
        print('WARNING: the 10s audio is truncated because your hop length and chunk size doesn t allow to get a 10s file')

    #fmel_tr = bt.FineMelTransform(sr=sr, flen=2048, hlen=250, window='hann', n_mels=256)
    force_cpu = False
    useCuda = torch.cuda.is_available() and not force_cpu
    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        ltype = torch.cuda.LongTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        ltype = torch.LongTensor
        #MT: add
        device = torch.device("cpu")

    if not os.path.exists(args.waveforms_hdf5_path + '/' + args.dataset_type):
        # Create the directory recursively
        os.makedirs(args.waveforms_hdf5_path + '/' + args.dataset_type)

    n_freq = mels_tr.mel_bins
    n_time = mels_tr.get_stft_num_time_bin(chunk_len)
    if ("wav" in args.dataset_type) & (split == 'eval'):
        n_time_reshape = mels_tr.get_stft_num_time_bin(10)
    else:
        n_time_reshape = n_time
    n_time_hop = mels_tr.get_stft_num_time_bin(hop_len)

    data_settings = {
        'split' : 
        {
        'valid_ratio' : args.valid_ratio,
        'eval_ratio' : args.eval_ratio
        },
        'third_octave' : 
        {
        'sr' : sr,
        'n_tho' : n_tho,
        'ref_freq' : ref_freq,
        'flen' : flen,
        'hlen' : hlen,
        },
        'mel' :
        {
        'sr' : sr_target,
        'n_freq' : n_freq,
        'n_time' : n_time,
        'n_time_hop' : n_time_hop,
        'chunk_len' : chunk_len,
        'hop_len' : hop_len,
        }
    }

    with open(args.waveforms_hdf5_path + '/' + args.dataset_type + '/' + 'settings.yaml', 'w') as file:
        yaml.dump(data_settings, file)

    with h5py.File(args.waveforms_hdf5_path + '/' + args.dataset_type + '/' + split + '.h5', 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_n_num,)), dtype='S200')
        hf.create_dataset('logits', shape=((audios_n_num, 527)), dtype=np.float32)
        hf.create_dataset('pinv_spectrogram', shape=((audios_n_num, n_freq, n_time_reshape)), dtype=np.float32)

        #MT: if mel spectrogram
        #shape=((audios_n_num, nb_of_freq, nb_of_time_bins))

        if "wav" in args.dataset_type:
            if split != 'eval':
                hf.create_dataset('waveform', shape=((audios_n_num, round(sr_target*chunk_len))), dtype=np.float32)
            else:
                hf.create_dataset('waveform', shape=((audios_n_num, round(sr_target*10))), dtype=np.float32)
        else:
            hf.create_dataset('spectrogram', shape=((audios_n_num, n_freq, n_time)), dtype=np.float32)

        n_file = 0
        # Pack waveform & target of several audio clips to a single hdf5 file
        for n in range(audios_num):
            audio_path = args.audio_dataset +  dataset[n]
            audio_name = dataset[n].split('/')[1][:-4]

            if os.path.isfile(audio_path):
                logging.info('{} {}'.format(n, audio_path))
                (waveform_32k, _) = librosa.core.load(audio_path, sr=sr, mono=True)
                (waveform_24k, _) = librosa.core.load(audio_path, sr=sr_target, mono=True)

                if len(waveform_32k) != 320000:
                    print('ERROR: not a 10s file')

                audio_n_24k = chunker_24k.chunks_with_hop(waveform_24k)
                audio_n_32k = chunker_32k.chunks_with_hop(waveform_32k)

                for idx, (audio_24k, audio_32k) in enumerate(zip(audio_n_24k, audio_n_32k)):

                    #MT: for mel spectrograms
                    spec = mels_tr.wave_to_mels(audio_24k).squeeze(dim=0).T.type(dtype).cpu().numpy()

                    thirdo_spec = torch.from_numpy(tho_tr.wave_to_third_octave(audio_32k)).T.unsqueeze(dim=0).type(dtype)

                    spec_from_thirdo = pt.pinv(thirdo_spec, tho_tr, mels_tr, reshape=n_time_reshape).cpu().numpy()

                    hf['audio_name'][n*num_chunks+idx] = audio_name + '___' + str(idx)
                    hf['pinv_spectrogram'][n*num_chunks+idx] = spec_from_thirdo

                    if "wav" in args.dataset_type:
                        hf['waveform'][n*num_chunks+idx] = audio_24k
                    else:
                        hf['spectrogram'][n*num_chunks+idx] = spec
            else:
                print('NO FILE')
            n_file += 1
            print('\r' + f'{n_file} / {audios_num} files have been processed in {split} dataset',end=' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dataset', type=str, default='/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/LJ-SPEECH_DataSet_resampled/', help='Directory where downloaded audio is saved.')
    parser.add_argument('--waveforms_hdf5_path', type=str, default='/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/6-diffusionFromThirdo-Data/spectral_data/', help='Path to save out packed hdf5.')
    parser.add_argument('--valid_ratio', type=float, default=0.005,
                        help='The ratio of data to use for validation')
    parser.add_argument('--eval_ratio', type=float, default=0.005,
                        help='The ratio of data to use for evaluation')
    parser.add_argument('--dataset_type', type=str, default='trash',
                        help='Whether to filter the data')
    args = parser.parse_args()

    pack_waveforms_to_hdf5(args)
