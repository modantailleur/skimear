import numpy as np
import h5py
import csv
import time
import logging
import torch
import os
import sys
import yaml
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)
import utils.util as ut
import librosa
import pandas as pd
import glob
#from utilities import int16_to_float32


def read_black_list(black_list_csv):
    """Read audio names from black list. 
    """
    with open(black_list_csv, 'r') as fr:
        reader = csv.reader(fr)
        lines = list(reader)

    black_list_names = ['Y{}.wav'.format(line[0]) for line in lines]
    return black_list_names

class DcaseTask1Dataset1D(object):
    def __init__(self, hdf5_path, sample_rate=32000):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.sample_rate = sample_rate
        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, 'r') as hf:
            self.audio_names = [audio_name for audio_name in hf['audio_name'][:]]
        self.len_dataset = len(self.audio_names)

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][idx]
            waveform = hf['waveform'][idx]
            pinv_waveform = hf['pinv_waveform'][idx]

        return waveform, pinv_waveform, audio_name

    
    def __len__(self):
        return self.len_dataset
    
    def resample(self, waveform):
        """Resample.

        Args:
          waveform: (clip_samples,)

        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0 :: 2]
        elif self.sample_rate == 8000:
            return waveform[0 :: 4]
        else:
            raise Exception('Incorrect sample rate!')

class WavDataset(object):
    def __init__(self, setting_data, data_path, subset='train', sample_rate=32000):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.sample_rate = sample_rate
        self.subset_path = data_path + '/' + subset + '.h5' 
        # self.data_setting_path = data_path + 'setting.yaml'
        # with open(self.data_setting_path, 'r') as file:
        #     self.data_setting = yaml.load(file, Loader=yaml.FullLoader)
        with h5py.File(self.subset_path, 'r') as hf:
            self.audio_names = [audio_name for audio_name in hf['audio_name'][:]]

        if setting_data is None:
            data_setting_path = data_path + 'setting.yaml'
            with open(data_setting_path, 'r') as file:
                self.data_setting = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.setting_data = setting_data

        self.len_dataset = len(self.audio_names)

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(self.subset_path, 'r') as hf:
            audio_name = hf['audio_name'][idx]
            waveform = hf['waveform'][idx]
            tho_spectrogram = hf['pinv_spectrogram'][idx]
            logits = hf['logits'][idx]

        return idx, waveform, tho_spectrogram, logits, audio_name
    
    def __len__(self):
        return self.len_dataset
    
    def resample(self, waveform):
        """Resample.

        Args:
          waveform: (clip_samples,)

        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0 :: 2]
        elif self.sample_rate == 8000:
            return waveform[0 :: 4]
        else:
            raise Exception('Incorrect sample rate!')

class DcaseTask1Dataset(object):
    def __init__(self, setting_data, data_path, subset='train', sample_rate=32000):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.sample_rate = sample_rate
        self.subset_path = data_path + '/' + subset + '.h5' 
        # self.data_setting_path = data_path + 'setting.yaml'
        # with open(self.data_setting_path, 'r') as file:
        #     self.data_setting = yaml.load(file, Loader=yaml.FullLoader)
        with h5py.File(self.subset_path, 'r') as hf:
            self.audio_names = [audio_name for audio_name in hf['audio_name'][:]]

        if setting_data is None:
            data_setting_path = data_path + 'setting.yaml'
            with open(data_setting_path, 'r') as file:
                self.data_setting = yaml.load(file, Loader=yaml.FullLoader)
        else:
            self.setting_data = setting_data

        self.len_dataset = len(self.audio_names)

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """
        # hdf5_path = meta['hdf5_path']
        # index_in_hdf5 = meta['index_in_hdf5']

        with h5py.File(self.subset_path, 'r') as hf:
            audio_name = hf['audio_name'][idx]
            #waveform = hf['waveform'][idx]
            spectrogram = hf['spectrogram'][idx]
            tho_spectrogram = hf['pinv_spectrogram'][idx]
            logits = hf['logits'][idx]
            # mel_spectrogram = hf['mel_spectrogram'][idx]
            # mel_tho_spectrogram = hf['mel_pinv_spectrogram'][idx]

            #MT: to change (obvisously)
            #tho_spectrogram = torch.rand((1, 1025, 64))
            # waveform = self.resample(waveform)
            # target = hf['spectrogram'][index_in_hdf5].astype(np.float32)

        # data_dict = {
        #     'audio_name': audio_name, 'waveform': waveform, 'spectrogram': spectrogram}
            
        #return spectrogram, tho_spectrogram, audio_name, mel_spectrogram, mel_tho_spectrogram
        #return torch.Tensor([]), torch.Tensor([]), audio_name, mel_spectrogram, mel_tho_spectrogram
        return idx, spectrogram, tho_spectrogram, logits, audio_name, torch.Tensor([]), torch.Tensor([])

    
    def __len__(self):
        return self.len_dataset
    
    def resample(self, waveform):
        """Resample.

        Args:
          waveform: (clip_samples,)

        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0 :: 2]
        elif self.sample_rate == 8000:
            return waveform[0 :: 4]
        else:
            raise Exception('Incorrect sample rate!')

class MelDataset(object):
    def __init__(self, eval_dataset, output_mel_path, output_mel_name, setting_data):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.setting_data = setting_data
        output_mels_fname = np.memmap(output_mel_path+output_mel_name + 'fname.dat', dtype='S100', mode ='r',shape=(eval_dataset.len_dataset))
        output_mels_fname_idx = np.array([int(s.decode('utf-8').split('___')[1]) for s in output_mels_fname])
        self.output_mels_fname_wav = np.array([s.decode('utf-8').split('___')[0] for s in output_mels_fname])
        n_chunks_per_file = np.max(output_mels_fname_idx) + 1
        # self.output_mels_aggragated_fname_wav = np.unique(self.output_mels_fname_wav)
        self.output_mels_aggragated_fname_wav = self.output_mels_fname_wav[::n_chunks_per_file]

        self.new_shape = (eval_dataset.len_dataset // n_chunks_per_file, setting_data['mel']['n_freq'], setting_data['mel']['n_time'] + setting_data['mel']['n_time_hop'] * (n_chunks_per_file - 1))
        self.output_concat_mels = np.memmap(output_mel_path+output_mel_name + 'aggregated.dat', dtype=np.float64,
                        mode='r', shape=self.new_shape)

        self.len_dataset = eval_dataset.len_dataset // n_chunks_per_file

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        audio_name = self.output_mels_aggragated_fname_wav[idx]
        # audio_name = ['aaa']
        spectrogram = torch.from_numpy(np.copy(self.output_concat_mels[idx]))

        return idx, spectrogram, audio_name

    
    def __len__(self):
        return self.len_dataset

# Function to recursively find all .flac files in a directory
def find_audio_files(directory):
    audio_files = []
    for extension in ['wav', 'mp3', 'flac']:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith("." + extension):
                    audio_files.append(os.path.join(root, file))
    return audio_files

def padd_or_cut(audio, reference):
    # If audio_gt is longer than 320000, truncate it
    if len(audio) > len(reference):
        return(audio[:len(reference)])
    elif len(audio) < len(reference):
        audio = audio[:len(reference)]
        # Calculate the number of zeros to pad
        padding_length = len(reference) - len(audio)
        # Pad the audio_gt array with zeros
        return(np.pad(audio, (0, padding_length), mode='constant'))
    else:
        return(audio)

# Function to read and concatenate all .txt files in current directory and subdirectories
def concatenate_txt_files_to_df(folder_path):
    # List to store dataframes
    dfs = []
    
    # Find all .txt files in current directory and subdirectories
    txt_files = glob.glob(os.path.join(folder_path, '**', '*.txt'), recursive=True)
    
    for file in txt_files:
        # Read the file line by line and split into ID and Text columns
        with open(file, 'r') as f:
            lines = f.readlines()
            data = []
            for line in lines:
                # Split only on the first occurrence of a space
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    data.append(parts)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['file', 'script1'])
        dfs.append(df)
    
    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    return concatenated_df

class AudioOutputsDataset(object):
    def __init__(self, setting_data, audio_path, audio_path_oracle, audio_path_gt, evalset, vocoder="gomin", sr=32000):
        """
        Initialize the AudioOutputsDataset.

        Args:
            setting_data (object): Setting data.
            audio_path (str): Path to the audio files to be evaluated (audio --> 1/3oct --> mels --> audio).
            audio_path_oracle (str): Path to oracle audio files (audio --> mels --> audio).
            audio_path_gt (str): Path to ground truth audio files (audio).
            vocoder (str, optional): Vocoder name (for mels --> audio). Can be either "gomin" or "grifflim". Defaults to "gomin".
        """
        self.setting_data = setting_data
        self.sr = sr
        self.evalset = evalset

        # Initialize dictionaries to store audio waveforms and file names
        self.audio_dict = {}

        # Add the vocoder path to the audio path (gomin and grifflim audio outputs are stored in different folders)
        self.audio_path = os.path.join(audio_path, vocoder) 
        self.audio_path_oracle = os.path.join(audio_path_oracle, vocoder)
        self.audio_path_gt = audio_path_gt

        # Process audio files to be evaluated
        for filename in os.listdir(self.audio_path):
            full_filename = os.path.join(self.audio_path, filename)
            fname = filename.split('___')
            fname = fname[1] if len(fname) == 2 else fname[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            self.audio_dict[fname] = audio

        # Initialize dictionaries to store oracle audio waveforms and file names
        self.audio_dict_oracle = {}

        # Process oracle audio files
        for filename in os.listdir(self.audio_path_oracle):
            full_filename = os.path.join(self.audio_path_oracle, filename)
            fname = filename.split('___')
            fname = fname[1] if len(fname) == 2 else fname[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            self.audio_dict_oracle[fname] = audio

        # Process groudtruth audio files (from the dataset audio folder). 
        self.audio_dict_gt = {}
        self.filename_gt = find_audio_files(self.audio_path_gt)

        for key in self.audio_dict:
            index = next((i for i, element in enumerate(self.filename_gt) if key in element), None)
            fname = self.filename_gt[index]
            audio_gt, _ = librosa.load(fname, sr=self.sr)
            audio_gt = padd_or_cut(audio_gt, self.audio_dict[key])
            self.audio_dict_gt[key] = audio_gt

        #sort the dictionnaries (to have every dict in same order)
        self.audio_dict_oracle = {key: self.audio_dict_oracle[key] for key in sorted(self.audio_dict_oracle)}
        self.audio_dict = {key: self.audio_dict[key] for key in sorted(self.audio_dict)}
        self.audio_dict_gt = {key: self.audio_dict_gt[key] for key in sorted(self.audio_dict_gt)}

        #transform the dicts into list for the __getitem__
        self.fname_l = list(self.audio_dict_oracle.keys())
        self.audio_l = [self.audio_dict[key] for key in self.audio_dict.keys()]
        self.audio_l_oracle = [self.audio_dict_oracle[key] for key in self.audio_dict_oracle.keys()]
        self.audio_l_gt = [self.audio_dict_gt[key] for key in self.audio_dict_gt.keys()]

        self.len_dataset = len(self.fname_l)

        if "ljspeech" in self.evalset:
            self.metadata_file = self.audio_path_gt + "/metadata_with_split.csv"
            df = pd.read_csv(self.metadata_file, sep="|")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values
        if self.evalset == "librispeech":
            df = concatenate_txt_files_to_df(self.audio_path_gt + "/test-clean/")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        name = self.fname_l[idx]
        audio = self.audio_l[idx]
        audio_oracle = self.audio_l_oracle[idx]
        audio_gt = self.audio_l_gt[idx]
        transcript = self.transcript_l[idx]
        return idx, audio, audio_oracle, audio_gt, name, transcript

    def __len__(self):
        return self.len_dataset

class RandomAudioOutputsDataset(object):
    def __init__(self, setting_data, audio_path, audio_path_oracle, audio_path_gt, evalset, vocoder="gomin", sr=32000):
        """
        Initialize the AudioOutputsDataset.

        Args:
            setting_data (object): Setting data.
            audio_path (str): Path to the audio files to be evaluated (audio --> 1/3oct --> mels --> audio).
            audio_path_oracle (str): Path to oracle audio files (audio --> mels --> audio).
            audio_path_gt (str): Path to ground truth audio files (audio).
            vocoder (str, optional): Vocoder name (for mels --> audio). Can be either "gomin" or "grifflim". Defaults to "gomin".
        """
        self.setting_data = setting_data
        self.sr = sr
        self.evalset = evalset

        # Initialize dictionaries to store audio waveforms and file names
        self.audio_dict = {}

        # Add the vocoder path to the audio path (gomin and grifflim audio outputs are stored in different folders)
        self.audio_path = os.path.join(audio_path, vocoder) 
        self.audio_path_oracle = os.path.join(audio_path_oracle, vocoder)
        self.audio_path_gt = audio_path_gt

        # Process audio files to be evaluated
        for filename in os.listdir(self.audio_path_oracle):
            full_filename = os.path.join(self.audio_path_oracle, filename)
            fname = filename.split('___')
            fname = fname[1] if len(fname) == 2 else fname[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            audio = 2 * np.random.rand(audio.shape[0]) -1
            self.audio_dict[fname] = audio

        # Initialize dictionaries to store oracle audio waveforms and file names
        self.audio_dict_oracle = {}

        # Process oracle audio files
        for filename in os.listdir(self.audio_path_oracle):
            full_filename = os.path.join(self.audio_path_oracle, filename)
            fname = filename.split('___')
            fname = fname[1] if len(fname) == 2 else fname[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            self.audio_dict_oracle[fname] = audio

        # Process groudtruth audio files (from the dataset audio folder). 
        self.audio_dict_gt = {}
        self.filename_gt = find_audio_files(self.audio_path_gt)

        for key in self.audio_dict:
            index = next((i for i, element in enumerate(self.filename_gt) if key in element), None)
            fname = self.filename_gt[index]
            audio_gt, _ = librosa.load(fname, sr=self.sr)
            audio_gt = padd_or_cut(audio_gt, self.audio_dict[key])
            self.audio_dict_gt[key] = audio_gt

        #sort the dictionnaries (to have every dict in same order)
        self.audio_dict_oracle = {key: self.audio_dict_oracle[key] for key in sorted(self.audio_dict_oracle)}
        self.audio_dict = {key: self.audio_dict[key] for key in sorted(self.audio_dict)}
        self.audio_dict_gt = {key: self.audio_dict_gt[key] for key in sorted(self.audio_dict_gt)}

        #transform the dicts into list for the __getitem__
        self.fname_l = list(self.audio_dict_oracle.keys())
        self.audio_l = [self.audio_dict[key] for key in self.audio_dict.keys()]
        self.audio_l_oracle = [self.audio_dict_oracle[key] for key in self.audio_dict_oracle.keys()]
        self.audio_l_gt = [self.audio_dict_gt[key] for key in self.audio_dict_gt.keys()]

        self.len_dataset = len(self.fname_l)

        if "ljspeech" in self.evalset:
            self.metadata_file = self.audio_path_gt + "/metadata_with_split.csv"
            df = pd.read_csv(self.metadata_file, sep="|")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values
        if self.evalset == "librispeech":
            df = concatenate_txt_files_to_df(self.audio_path_gt + "/test-clean/")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        name = self.fname_l[idx]
        audio = self.audio_l[idx]
        audio_oracle = self.audio_l_oracle[idx]
        audio_gt = self.audio_l_gt[idx]
        transcript = self.transcript_l[idx]
        return idx, audio, audio_oracle, audio_gt, name, transcript

    def __len__(self):
        return self.len_dataset

class DiffwaveAudioOutputsDataset(object):
    def __init__(self, setting_data, audio_path, audio_path_oracle, audio_path_gt, evalset, sr=32000):
        """
        Initialize the AudioOutputsDataset.

        Args:
            setting_data (object): Setting data.
            audio_path (str): Path to the audio files to be evaluated (audio --> 1/3oct --> mels --> audio).
            audio_path_oracle (str): Path to oracle audio files (audio --> mels --> audio).
            audio_path_gt (str): Path to ground truth audio files (audio).
            vocoder (str, optional): Vocoder name (for mels --> audio). Can be either "gomin" or "grifflim". Defaults to "gomin".
        """
        self.setting_data = setting_data
        self.sr = sr
        self.evalset = evalset

        # Initialize dictionaries to store audio waveforms and file names
        self.audio_dict = {}

        # Add the vocoder path to the audio path (gomin and grifflim audio outputs are stored in different folders)
        self.audio_path = os.path.join(audio_path) 
        self.audio_path_gt = audio_path_gt

        # Process audio files to be evaluated
        for filename in os.listdir(self.audio_path):
            full_filename = os.path.join(self.audio_path, filename)
            fname = filename.split('___')
            fname = fname[1] if len(fname) == 2 else fname[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            audio = librosa.util.normalize(audio)
            self.audio_dict[fname] = audio

        # Process groudtruth audio files (from the dataset audio folder). 
        self.audio_dict_gt = {}
        self.filename_gt = find_audio_files(self.audio_path_gt)

        for key in self.audio_dict:
            index = next((i for i, element in enumerate(self.filename_gt) if key in element), None)
            fname = self.filename_gt[index]
            audio_gt, _ = librosa.load(fname, sr=self.sr)
            audio_gt = padd_or_cut(audio_gt, self.audio_dict[key])
            audio_gt = librosa.util.normalize(audio_gt)
            self.audio_dict_gt[key] = audio_gt

        #sort the dictionnaries (to have every dict in same order)
        self.audio_dict = {key: self.audio_dict[key] for key in sorted(self.audio_dict)}
        self.audio_dict_gt = {key: self.audio_dict_gt[key] for key in sorted(self.audio_dict_gt)}

        #transform the dicts into list for the __getitem__
        self.fname_l = list(self.audio_dict_gt.keys())
        self.audio_l = [self.audio_dict[key] for key in self.audio_dict.keys()]
        self.audio_l_gt = [self.audio_dict_gt[key] for key in self.audio_dict_gt.keys()]

        self.len_dataset = len(self.fname_l)

        if "ljspeech" in self.evalset:
            self.metadata_file = self.audio_path_gt + "/metadata_with_split.csv"
            df = pd.read_csv(self.metadata_file, sep="|")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values
        if self.evalset == "librispeech":
            df = concatenate_txt_files_to_df(self.audio_path_gt + "/test-clean/")
            df['file'] = pd.Categorical(df['file'], categories=self.fname_l, ordered=True)
            df = df.sort_values('file')
            df.dropna(subset=['file'], inplace=True)
            self.transcript_l = df["script1"].values

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        name = self.fname_l[idx]
        audio = self.audio_l[idx]
        audio_gt = self.audio_l_gt[idx]
        transcript = self.transcript_l[idx]
        return idx, audio, audio_gt, name, transcript

    def __len__(self):
        return self.len_dataset


class OutputsDataset(object):
    def __init__(self, setting_data, audio_path, pann_path, audio_path_oracle, audio_path_gt, pann_path_oracle, vocoder="gomin"):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.setting_data = setting_data
        self.sr = 32000
        # Initialize dictionaries to store audio waveforms and file names
        self.audio_dict = {}
        # self.audio_fname_dict = {}

        self.audio_path = os.path.join(audio_path, vocoder) 
        self.audio_path_oracle = os.path.join(audio_path_oracle, vocoder)
        self.audio_path_gt = audio_path_gt
        self.pann_path = os.path.join(pann_path, vocoder)
        self.pann_path_oracle = os.path.join(pann_path_oracle, vocoder)

        # Process regular audio files
        for filename in os.listdir(self.audio_path):
            full_filename = os.path.join(self.audio_path, filename)
            fname = filename.split('___')[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            self.audio_dict[fname] = audio
            # self.audio_fname_dict[fname] = full_filename

        # Initialize dictionaries to store predicted annotations and file names
        self.pann_pred_dict = {}
        # self.pann_fname_dict = {}

        # Process pann output files
        for filename in os.listdir(self.pann_path):
            full_filename = os.path.join(self.pann_path, filename)
            fname = filename.split('___')[2]
            fname, _ = os.path.splitext(fname)
            pann_pred = np.load(full_filename)
            self.pann_pred_dict[fname] = pann_pred
            # self.pann_fname_dict[fname] = full_filename

        # Initialize dictionaries to store oracle audio waveforms and file names
        self.audio_dict_oracle = {}
        # self.audio_fname_dict_oracle = {}

        # Process oracle audio files
        for filename in os.listdir(self.audio_path_oracle):
            full_filename = os.path.join(self.audio_path_oracle, filename)
            fname = filename.split('___')[2]
            fname, _ = os.path.splitext(fname)
            audio, _ = librosa.load(full_filename, sr=self.sr)
            self.audio_dict_oracle[fname] = audio
            # self.audio_fname_dict_oracle[fname] = full_filename

        # Initialize dictionaries to store oracle predicted annotations and file names
        self.pann_pred_dict_oracle = {}
        # self.pann_fname_dict_oracle = {}

        # Process pann output files for oracle
        for filename in os.listdir(self.pann_path_oracle):
            full_filename = os.path.join(self.pann_path_oracle, filename)
            fname = filename.split('___')[2]
            fname, _ = os.path.splitext(fname)
            pann_pred = np.load(full_filename)
            self.pann_pred_dict_oracle[fname] = pann_pred
            # self.pann_fname_dict_oracle[fname] = full_filename

        self.audio_dict_gt = {}
        self.filename_gt = find_audio_files(self.audio_path_gt)
        for key in self.audio_dict:
            index = next((i for i, element in enumerate(self.filename_gt) if key in element), None)
            fname = self.filename_gt[index]
            audio_gt, _ = librosa.load(fname, sr=self.sr)
            audio_gt = padd_or_cut(audio_gt, self.audio_dict[key])
            self.audio_dict_gt[key] = audio_gt

        #sort the dictionnaries
        self.pann_pred_dict_oracle = {key: self.pann_pred_dict_oracle[key] for key in sorted(self.pann_pred_dict_oracle)}
        self.audio_dict_oracle = {key: self.audio_dict_oracle[key] for key in sorted(self.audio_dict_oracle)}
        self.pann_pred_dict = {key: self.pann_pred_dict[key] for key in sorted(self.pann_pred_dict)}
        self.audio_dict = {key: self.audio_dict[key] for key in sorted(self.audio_dict)}
        self.audio_dict_gt = {key: self.audio_dict_gt[key] for key in sorted(self.audio_dict_gt)}

        if (set(self.pann_pred_dict_oracle.keys()) != set(self.audio_dict_oracle.keys()) or
            set(self.pann_pred_dict_oracle.keys()) != set(self.pann_pred_dict.keys()) or
            set(self.pann_pred_dict_oracle.keys()) != set(self.audio_dict.keys())):
            raise Exception("Keys of the dictionaries should be identical.")

        self.fname_l = list(self.pann_pred_dict_oracle.keys())
        self.audio_l = [self.audio_dict[key] for key in self.audio_dict.keys()]
        self.pann_pred_l = [self.pann_pred_dict[key] for key in self.pann_pred_dict.keys()]
        self.audio_l_oracle = [self.audio_dict_oracle[key] for key in self.audio_dict_oracle.keys()]
        self.pann_pred_l_oracle = [self.pann_pred_dict_oracle[key] for key in self.pann_pred_dict_oracle.keys()]
        self.audio_l_gt = [self.audio_dict_gt[key] for key in self.audio_dict_gt.keys()]

        self.len_dataset = len(self.fname_l)

    def __getitem__(self, idx):
        """Load waveform and target of an audio clip.
        
        Args:
          meta: {
            'hdf5_path': str, 
            'index_in_hdf5': int}

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': (classes_num,)}
        """

        name = self.fname_l[idx]
        pann_pred = self.pann_pred_l[idx]
        audio = self.audio_l[idx]
        pann_pred_oracle = self.pann_pred_l_oracle[idx]
        audio_oracle = self.audio_l_oracle[idx]
        audio_gt = self.audio_l_gt[idx]
        return idx, audio, pann_pred, audio_oracle, pann_pred_oracle, audio_gt, name

    def __len__(self):
        return self.len_dataset

class Base(object):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv, random_seed):
        """Base class of train sampler.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        if black_list_csv:
            self.black_list_names = read_black_list(black_list_csv)
        else:
            self.black_list_names = []

        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name for audio_name in hf['audio_name'][:]]
            self.spectrograms = [spec for spec in hf['spectrogram'][:]]
            self.waveforms = [spec for spec in hf['waveform'][:]]
        
        self.audio_nums = len(self.audio_names)
        #(self.audios_num, self.classes_num) = self.targets.shape
        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class TrainSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(TrainSampler, self).__init__(indexes_hdf5_path, batch_size, 
            black_list_csv, random_seed)
        
        self.indexes = np.arange(self.audios_num)
            
        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= self.audios_num:
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)
                
                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    # batch_meta.append({
                    #     'hdf5_path': self.hdf5_paths[index], 
                    #     'index_in_hdf5': self.indexes_in_hdf5[index]})
                    batch_meta.append({
                        'waveform': self.waveforms[index], 
                        'spectrogram': self.spectrograms[index]})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state
            
    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class BalancedTrainSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None, 
        random_seed=1234):
        """Balanced sampler. Generate batch meta for training. Data are equally 
        sampled from different sound classes.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        super(BalancedTrainSampler, self).__init__(indexes_hdf5_path, 
            batch_size, black_list_csv, random_seed)
        
        self.samples_num_per_class = np.sum(self.targets, axis=0)
        logging.info('samples_num_per_class: {}'.format(
            self.samples_num_per_class.astype(np.int32)))
        
        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        
        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(self.targets[:, k] == 1)[0])
            
        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])
        
        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                index = self.indexes_per_class[class_id][pointer]
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_num_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append({
                        'hdf5_path': self.hdf5_paths[index], 
                        'index_in_hdf5': self.indexes_in_hdf5[index]})
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes_per_class': self.indexes_per_class, 
            'queue': self.queue, 
            'pointers_of_classes': self.pointers_of_classes}
        return state
            
    def load_state_dict(self, state):
        self.indexes_per_class = state['indexes_per_class']
        self.queue = state['queue']
        self.pointers_of_classes = state['pointers_of_classes']


class AlternateTrainSampler(Base):
    def __init__(self, indexes_hdf5_path, batch_size, black_list_csv=None,
        random_seed=1234):
        """AlternateSampler is a combination of Sampler and Balanced Sampler. 
        AlternateSampler alternately sample data from Sampler and Blanced Sampler.
        
        Args:
          indexes_hdf5_path: string          
          batch_size: int
          black_list_csv: string
          random_seed: int
        """
        self.sampler1 = TrainSampler(indexes_hdf5_path, batch_size, 
            black_list_csv, random_seed)

        self.sampler2 = BalancedTrainSampler(indexes_hdf5_path, batch_size, 
            black_list_csv, random_seed)

        self.batch_size = batch_size
        self.count = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            self.count += 1

            if self.count % 2 == 0:
                batch_meta = []
                i = 0
                while i < batch_size:
                    index = self.sampler1.indexes[self.sampler1.pointer]
                    self.sampler1.pointer += 1

                    # Shuffle indexes and reset pointer
                    if self.sampler1.pointer >= self.sampler1.audios_num:
                        self.sampler1.pointer = 0
                        self.sampler1.random_state.shuffle(self.sampler1.indexes)
                    
                    # If audio in black list then continue
                    if self.sampler1.audio_names[index] in self.sampler1.black_list_names:
                        continue
                    else:
                        batch_meta.append({
                            'hdf5_path': self.sampler1.hdf5_paths[index], 
                            'index_in_hdf5': self.sampler1.indexes_in_hdf5[index]})
                        i += 1

            elif self.count % 2 == 1:
                batch_meta = []
                i = 0
                while i < batch_size:
                    if len(self.sampler2.queue) == 0:
                        self.sampler2.queue = self.sampler2.expand_queue(self.sampler2.queue)

                    class_id = self.sampler2.queue.pop(0)
                    pointer = self.sampler2.pointers_of_classes[class_id]
                    self.sampler2.pointers_of_classes[class_id] += 1
                    index = self.sampler2.indexes_per_class[class_id][pointer]
                    
                    # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                    if self.sampler2.pointers_of_classes[class_id] >= self.sampler2.samples_num_per_class[class_id]:
                        self.sampler2.pointers_of_classes[class_id] = 0
                        self.sampler2.random_state.shuffle(self.sampler2.indexes_per_class[class_id])

                    # If audio in black list then continue
                    if self.sampler2.audio_names[index] in self.sampler2.black_list_names:
                        continue
                    else:
                        batch_meta.append({
                            'hdf5_path': self.sampler2.hdf5_paths[index], 
                            'index_in_hdf5': self.sampler2.indexes_in_hdf5[index]})
                        i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'sampler1': self.sampler1.state_dict(), 
            'sampler2': self.sampler2.state_dict()}
        return state

    def load_state_dict(self, state):
        self.sampler1.load_state_dict(state['sampler1'])
        self.sampler2.load_state_dict(state['sampler2'])


class EvaluateSampler(object):
    def __init__(self, indexes_hdf5_path, batch_size):
        """Evaluate sampler. Generate batch meta for evaluation.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf['hdf5_path'][:]]
            self.indexes_in_hdf5 = hf['index_in_hdf5'][:]
            self.targets = hf['target'][:].astype(np.float32)
            
        self.audios_num = len(self.audio_names)

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 
             'index_in_hdf5': int}
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(pointer, 
                min(pointer + batch_size, self.audios_num))

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append({
                    'audio_name': self.audio_names[index], 
                    'hdf5_path': self.hdf5_paths[index], 
                    'index_in_hdf5': self.indexes_in_hdf5[index], 
                    'target': self.targets[index]})

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...}, 
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}
    
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict