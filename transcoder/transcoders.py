import numpy as np
import yaml
import numpy as np
import numpy as np
import torch.utils.data
import torch
import utils.baseline_inversion as bi
import transcoder.models_transcoder as md_tr
import yaml
from pann.pann_mel_inference import PannMelInference
from utils.util import get_transforms
import utils.bands_transform as bt
import transcoder.pinv_transcoder as pt
import librosa
from transcoder.exp_train_diffusion.diffusion_models import UNet2DModel
from transcoder.exp_train_diffusion.diffusion_inference import DDPMInference, ScoreSdeVeInference
from diffusers import DDPMScheduler, ScoreSdeVeScheduler
from transcoder.gomin.models import GomiGAN, DiffusionWrapper
from transcoder.gomin.config import GANConfig, DiffusionConfig
import os
import utils.util as ut
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import requests
from pathlib import Path

class ThirdOctaveToMelTranscoderPinv():

    def __init__(self, model_path, model_name, device, classifier='PANN'):

        self.device = device

        with open(model_path + "/" + model_name + '_settings.yaml') as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        self.input_shape = settings_model.get('input_shape')
        self.output_shape = settings_model.get('output_shape')

        #from settings of the model
        classifier = settings_model.get('mels_type')

        self.tho_tr, self.mels_tr = get_transforms(sr=32000, 
                                            flen=4096,
                                            hlen=4000,
                                            classifier=classifier,
                                            device=device)


    def thirdo_to_mels_1s(self, x, dtype=torch.FloatTensor):
        x_inf = bi.pinv(x, self.tho_tr, self.mels_tr, reshape=self.output_shape[0], dtype=dtype, device=self.device)
        x_inf = x_inf[0].T.numpy()
        return(x_inf)

    def wave_to_thirdo_to_mels_1s(self, x, dtype=torch.FloatTensor):
        x_tho = self.tho_tr.wave_to_third_octave(x)
        x_tho = torch.from_numpy(x_tho.T)
        x_tho = x_tho.unsqueeze(0)
        x_tho = x_tho.type(dtype)
        x_mels_inf = self.thirdo_to_mels_1s(x_tho)
        return(x_mels_inf)
    
    def wave_to_thirdo_to_mels(self, x):
        chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        for x_i in x_sliced:
            x_mels_inf = self.wave_to_thirdo_to_mels_1s(x_i)
            x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
        return(x_mels_inf_tot)

class ThirdOctaveToMelTranscoder():
    '''
    Contains all the functions to transcode third-octave to mels and to make
    PANN predictions. 
    '''
    def __init__(self, model_type, model_name, model_path, device, flen=4096, hlen=4000, pann_type='ResNet38'):
        '''
        Parameters:
        - model_type: str
            Determines the architecture of the transcoder model.
        - model_name: str
            Specifies the filename (excluding the path) of the pre-trained model to be loaded.
        - model_path: str
            Specifies the directory where the model file and its corresponding settings file are stored.
        - device: torch.device
            Specifies the computing device (CPU or GPU) where the PyTorch model will be loaded and executed.
        - flen: int, optional (default=4096)
            Frame length for the third-octave spectrograms. 4096 is a fast third-octave spectrogram.
        - hlen: int, optional (default=4000)
            Hop length for the the third-octave spectrograms. 4000 is a fast third-octave spectrogram.
        - pann_type: str, optional (default='ResNet38')
            Type of PANN model to use. Only applicable if classifier is 'PANN'.
            Specifies the architecture or type of PANN (Pre-trained Audio Neural Network) model to use for classification.
        '''
        self.device = device

        settings_filename = f"{model_name}_settings.yaml"
        model_filename = f"{model_name}"

        settings_fp = model_path + "/" + settings_filename
        model_fp = model_path + "/" + model_filename

        # If the YAML doesn't exist, fetch both YAML and model from Hugging Face.
        if not Path(settings_fp).exists() or not Path(model_fp).exists():
            # Convert provided "blob/main" URLs to direct "resolve/main" for downloading.
            yaml_url = ("https://huggingface.co/modantailleur/fast-to-ear/resolve/main/"
                             "classifier%3DPANN%2Bdataset%3Dfull%2Bdilation%3D1%2Bepoch%3D200%2Bkernel_size%3D5"
                             "%2Blearning_rate%3D-3%2Bnb_channels%3D64%2Bnb_layers%3D5%2Bprop_logit%3D100%2Bstep"
                             "%3Dtrain%2Btranscoder%3Dcnn_pinv%2Bts%3D1_model_settings.yaml")
            model_url = ("https://huggingface.co/modantailleur/fast-to-ear/resolve/main/"
                              "classifier%3DPANN%2Bdataset%3Dfull%2Bdilation%3D1%2Bepoch%3D200%2Bkernel_size%3D5"
                              "%2Blearning_rate%3D-3%2Bnb_channels%3D64%2Bnb_layers%3D5%2Bprop_logit%3D100%2Bstep"
                              "%3Dtrain%2Btranscoder%3Dcnn_pinv%2Bts%3D1_model")

            def _download(url, dst):
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                with open(dst, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            # Download only what’s missing
            if not Path(settings_fp).exists():
                _download(yaml_url, settings_fp)
            if not Path(model_fp).exists():
                _download(model_url, model_fp)

        # Now load settings (file must exist by here)
        with open(settings_fp, "r") as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        input_shape = settings_model.get('input_shape')
        output_shape = settings_model.get('output_shape')

        cnn_kernel_size = settings_model.get('cnn_kernel_size')
        cnn_dilation = settings_model.get('cnn_dilation')
        cnn_nb_layers = settings_model.get('cnn_nb_layers')
        cnn_nb_channels = settings_model.get('cnn_nb_channels')

        #from settings of the model
        classifier = settings_model.get('mels_type')

        self.tho_tr, self.mels_tr = get_transforms(sr=32000, 
                                                flen=flen,
                                                hlen=hlen,
                                                classifier=classifier,
                                                device=device)

        if flen == 32758 and hlen == 32000:
            # time of a chunk as input of the transcoder
            self.tr_input_len = 10
        if flen == 4096 and hlen == 4000:
            # time of a chunk as input of the transcoder
            self.tr_input_len = 1
        
        if model_type == "cnn_pinv":
            self.model = md_tr.CNN(input_shape=input_shape, output_shape=output_shape, tho_tr=self.tho_tr, mels_tr=self.mels_tr, kernel_size=cnn_kernel_size, dilation=cnn_dilation, nb_layers=cnn_nb_layers, nb_channels=cnn_nb_channels, device=device)

        state_dict = torch.load(model_path + "/" + model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)

        if classifier == 'PANN':
            self.classif_inference = PannMelInference(verbose=False, device=device, pann_type=pann_type)

    def thirdo_to_mels_1s(self, x, torch_output=False):
        '''
        Transforms a 1-s third-octave spectrogram (size 29, 8) into a
        Mel spectrogram (size 64, 101 for PANN). 

        Parameters:
        - x: ndarray or torch.Tensor
        The input third-octave spectrogram with a shape of (29, 8). If torch_output is True,
        x should be a torch.Tensor; otherwise, it should be an ndarray.
        - torch_output: bool, optional (default=False)
        Specifies whether the output should be a torch.Tensor or not. 

        Returns:
        - ndarray or torch.Tensor
        The resulting Mel spectrogram with a shape of (64, 101) suitable for PANN classification task.
        If torch_output is True, the output is a torch.Tensor, else it is an ndarray
        '''
        if torch_output:
            x_inf = self.model(x)
        else:
            x_inf = self.model(x).detach()
            x_inf = x_inf[0].T.cpu().numpy()
        return(x_inf)

    def wave_to_thirdo_to_mels_1s(self, x, dtype=torch.FloatTensor):
        '''
        Converts a 1-s waveform into a third-octave spectrogram, 
        and transcode it into a Mel spectrogram with the CNN-transcoder.

        Parameters:
        - x: ndarray
            The input 1-s waveform data with a sample rate of 32kHz for PANN, and 16kHz for YamNet.
        - dtype: torch.dtype, optional (default=torch.FloatTensor)
            Specifies the data type for the torch.Tensor conversion. Default is torch.FloatTensor.

        Returns:
        - ndarray
            The resulting 1-s long Mel spectrogram (mel_bins, mel_frames) suitable for PANN classification tasks.
        '''

        x_tho = self.tho_tr.wave_to_third_octave(x)
        x_tho = torch.from_numpy(x_tho.T)
        x_tho = x_tho.unsqueeze(0)
        x_tho = x_tho.type(dtype)
        x_mels_inf = self.thirdo_to_mels_1s(x_tho)
        return(x_mels_inf)

    def wave_to_thirdo_to_mels(self, x, n_mels_frames_to_remove=1):
        '''
        Converts a waveform of any length into a third-octave spectrogram, 
        and transcode it into a Mel spectrogram with the CNN-transcoder.

        Parameters:
        - x: ndarray or torch.Tensor
            The input waveform data with a sample rate of 32kHz for PANN, and 16kHz for YamNet.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).

        Returns:
        - ndarray
            The resulting Mel spectrogram (mel_bins, mel_frames) suitable for PANN classification tasks.
        '''
        chunk_size = self.tho_tr.sr
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        for k, x_i in enumerate(x_sliced):
            if x_i.shape[0] >= self.tho_tr.sr:
                x_mels_inf = self.wave_to_thirdo_to_mels_1s(x_i)
                if k == len(x_sliced)-1:   
                    x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
                else:
                    x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)
            
        x_mels_inf_tot = np.array(x_mels_inf_tot)
        return(x_mels_inf_tot)

    def wave_to_mels_to_logit(self, x, frame_duration=10, weighted_avg=True, mean=True):
        '''
        Converts a waveform of any length into a Mel spectrogram, and give it 
        as input of the classifier (PANN or YamNet).
        
        Parameters:
        - x: ndarray or torch.Tensor
            The input waveform data with a sample rate of 32kHz for PANN, and 16kHz for YamNet.
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38is optimal with 10s chunks.
        - weighted_avg: bool, optional (default=True)
            If True, calculates the weighted average of logits based on the length of each frame.
            If False, uses the logits from each frame without averaging. If mean is False, this
            parameter is forced to be False.
        - mean: bool, optional (default=True)
            If True, calculates the mean of logits across all Mel frames. If False, uses the logits
            from each frame without averaging (useful for CNN14 for detection). If mean is False, weighted_avg is forced to be False.
        Returns:
        - ndarray
            The resulting Mel spectrogram (mel_bins, mel_frames) suitable for PANN classification tasks.
        '''
        if mean == False:
            weighted_avg = False

        chunk_size = self.tho_tr.sr * frame_duration
        x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
        x_mels_inf_tot = []
        x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
        x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
        weights = []
        for x_i in x_sliced:
            if len(x_i) >= self.tho_tr.sr:
                x_mels_inf = self.mels_tr.wave_to_mels(x_i)
                x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
                if (((x_i.shape[0] == chunk_size) or (weighted_avg==True) or (mean==False)) & (x_i.shape[0]>=32000)):
                    weight = x_i.shape[0]/chunk_size
                    weights.append(weight)
                    x_logits = self.classifier_prediction(x_mels_inf, mean=mean)
                    x_logits_tot =np.concatenate((x_logits_tot, x_logits), axis=1)

        x_mels_inf_tot = np.array(x_mels_inf_tot)
        x_logits_tot = np.array(x_logits_tot)

        if weighted_avg:
            x_logits_tot = np.average(x_logits_tot, axis=1, weights=weights)
            x_logits_tot = np.expand_dims(x_logits_tot, axis=1)

        x_logits_tot = x_logits_tot.T

        return(x_mels_inf_tot, x_logits_tot)
    
    def classifier_prediction(self, x, mean=True, torch_input=False, torch_output=False, output_embedding=False):
        '''
        Performs classification inference on input data.

        Parameters:
        - x: ndarray or torch.Tensor
            The input data for classification inference. If torch.Tensor, it is expected to be
            of type torch.FloatTensor. If ndarray, it will be converted to a torch.Tensor.
        - mean: bool, optional (default=True)
            If True, calculates the mean of logits across all Mel frames. If False, uses the logits
            from each frame without averaging (useful for CNN14 for detection).
        - torch_input: bool, optional (default=False)
            If True, assumes 'x' is a torch.Tensor. If False, converts 'x' to a torch.Tensor before
            passing it for inference.
        - torch_output: bool, optional (default=False)
            If True, returns the output as a torch.Tensor. If False, converts the torch.Tensor to
            a numpy array before returning.

        Returns:
        - ndarray or torch.Tensor
            The classification logits or probabilities. The shape depends on the 'mean' parameter:
            - If mean is True, the shape is (n_labels,)
            - If mean is False, the shape is (mel_frames, n_labels)
        '''
        if torch_input:
            temp = torch.unsqueeze(x, dim=0)
            if output_embedding:
                x_logits = self.classif_inference.get_embedding(temp, no_grad=True)
            else:
                x_logits = self.classif_inference.simple_inference(temp, no_grad=True, mean=mean)
        else:
            temp = torch.from_numpy(np.expand_dims(np.expand_dims(x.T, axis=0), axis=0))
            temp =  temp.to(torch.float32)
            if output_embedding:
                x_logits = self.classif_inference.get_embedding(temp, no_grad=True)
            else:
                x_logits = self.classif_inference.simple_inference(temp, no_grad=True, mean=mean)
        if not mean:
            x_logits = torch.unsqueeze(x_logits, dim=0)
        if not torch_output:
            x_logits = x_logits.detach().cpu().numpy().T
        return(x_logits)

    def mels_to_logit(self, x, mean=True, frame_duration=10, n_mels_frames_per_s=100, n_mels_frames_to_remove=1, weighted_avg=True):
        '''
        Converts Mel spectrogram frames into classification logits.

        Parameters:
        - x: ndarray 
            The input Mel spectrogram (mel_bins, mel_frames)
        - mean: bool, optional (default=True)
            If True, calculates the mean of logits across all Mel frames. If False, uses the logits
            from each frame without averaging (useful for CNN14 for detection).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - weighted_avg: bool, optional (default=True)
            If True, calculates the weighted average of logits based on the length of each frame.
            If False, uses the logits from each frame without averaging. If mean is False, this
            parameter is forced to be False.

        Returns:
        - ndarray
            The classification logits. The shape depends on the 'mean' parameter:
            - If mean is True, the shape is (n_labels,)
            - If mean is False, the shape is (time_frames, n_labels)

        '''
        # if mels are sliced into batches of 1s audio (for YamNet)
        if len(x.shape) > 2:
            X = []
            for i in range(x.shape[-1]):
                xi = x[:, :, 0, i]
                temp = torch.from_numpy(np.expand_dims(np.expand_dims(xi.T, axis=0), axis=0))
                temp =  temp.to(torch.float32)
                x_logits = self.classif_inference.simple_inference(temp, no_grad=True, mean=mean)
                if not mean:
                    x_logits = torch.unsqueeze(x_logits, dim=0)
                x_logits = x_logits.numpy()
                X.append(x_logits)
            X = np.array(X)
            X = X.reshape(-1, X.shape[-1])
        else:
            x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
            weights = []
            if frame_duration != 1:
                chunk_size = frame_duration*n_mels_frames_per_s
                x_mels_sliced = [x[:, i:i+chunk_size] for i in range(0, x.shape[1], chunk_size)]
                for k, x_i in enumerate(x_mels_sliced):
                    #avoid giving to the model chunks of audio with too short length
                    if (((x_i.shape[1] == chunk_size) or (weighted_avg==True) or (mean==False)) & (x_i.shape[1]>=n_mels_frames_per_s+n_mels_frames_to_remove)):
                        weight = x_i.shape[1]/chunk_size
                        weights.append(weight)
                        x_logits = self.classifier_prediction(x_i, mean=mean)
                        x_logits_tot = np.concatenate((x_logits_tot, x_logits), axis=1)
            if weighted_avg:
                x_logits_tot = np.average(x_logits_tot, axis=1, weights=weights)
                x_logits_tot = np.expand_dims(x_logits_tot, axis=1)
            X = x_logits_tot
        return(X)

    def thirdo_to_mels_to_embeddings(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], 2048, 0)).to(self.device)

        for k, x_i in enumerate(x_sliced):
            if x_i.shape[1] >= n_thirdo_frames_per_s:
                x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
                if k == len(x_sliced)-1:   
                    x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
                else:
                    x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        chunk_size = frame_duration*n_mels_frames_per_s
        x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
        for k, x_i in enumerate(x_mels_sliced):
            if x_i.shape[1] == chunk_size:
                x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True, output_embedding=True)
                x_logits = x_logits.unsqueeze(dim=-1)
                x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)
        
        return(x_mels_inf_tot, x_logits_tot)

    def thirdo_to_mels_to_logit(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], self.classif_inference.n_labels, 0)).to(self.device)
        for k, x_i in enumerate(x_sliced):
            if x_i.shape[1] >= n_thirdo_frames_per_s:
                x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
                if k == len(x_sliced)-1:   
                    x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
                else:
                    x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        chunk_size = frame_duration*n_mels_frames_per_s
        x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
        for k, x_i in enumerate(x_mels_sliced):
            if x_i.shape[1] == chunk_size:
                x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True)
                x_logits = x_logits.unsqueeze(dim=-1)
                x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)
        
        return(x_mels_inf_tot, x_logits_tot)
    
    def embedding_prediction(self, x, mean=True, torch_input=False, torch_output=False):
        '''
        Performs classification inference on input data.

        Parameters:
        - x: ndarray or torch.Tensor
            The input data for classification inference. If torch.Tensor, it is expected to be
            of type torch.FloatTensor. If ndarray, it will be converted to a torch.Tensor.
        - mean: bool, optional (default=True)
            If True, calculates the mean of logits across all Mel frames. If False, uses the logits
            from each frame without averaging (useful for CNN14 for detection).
        - torch_input: bool, optional (default=False)
            If True, assumes 'x' is a torch.Tensor. If False, converts 'x' to a torch.Tensor before
            passing it for inference.
        - torch_output: bool, optional (default=False)
            If True, returns the output as a torch.Tensor. If False, converts the torch.Tensor to
            a numpy array before returning.

        Returns:
        - ndarray or torch.Tensor
            The classification logits or probabilities. The shape depends on the 'mean' parameter:
            - If mean is True, the shape is (n_labels,)
            - If mean is False, the shape is (mel_frames, n_labels)
        '''
        if torch_input:
            temp = torch.unsqueeze(x, dim=0)
            x_logits = self.classif_inference.get_embedding(temp, no_grad=True)
        else:
            temp = torch.from_numpy(np.expand_dims(np.expand_dims(x.T, axis=0), axis=0))
            temp =  temp.to(torch.float32)
            x_logits = self.classif_inference.get_embedding(temp, no_grad=True)
        if not mean:
            x_logits = torch.unsqueeze(x_logits, dim=0)
        if not torch_output:
            x_logits = x_logits.detach().cpu().numpy().T
        return(x_logits)

    def mels_to_embedding(self, x, mean=True, frame_duration=10, n_mels_frames_per_s=100, n_mels_frames_to_remove=1, weighted_avg=True):
        '''
        Converts Mel spectrogram frames into classification logits.

        Parameters:
        - x: ndarray 
            The input Mel spectrogram (mel_bins, mel_frames)
        - mean: bool, optional (default=True)
            If True, calculates the mean of logits across all Mel frames. If False, uses the logits
            from each frame without averaging (useful for CNN14 for detection).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - weighted_avg: bool, optional (default=True)
            If True, calculates the weighted average of logits based on the length of each frame.
            If False, uses the logits from each frame without averaging. If mean is False, this
            parameter is forced to be False.

        Returns:
        - ndarray
            The classification logits. The shape depends on the 'mean' parameter:
            - If mean is True, the shape is (n_labels,)
            - If mean is False, the shape is (time_frames, n_labels)

        '''
        # if mels are sliced into batches of 1s audio (for YamNet)
        if len(x.shape) > 2:
            X = []
            for i in range(x.shape[-1]):
                xi = x[:, :, 0, i]
                temp = torch.from_numpy(np.expand_dims(np.expand_dims(xi.T, axis=0), axis=0))
                temp =  temp.to(torch.float32)
                x_logits = self.classif_inference.get_embedding(temp, no_grad=True, mean=mean)
                if not mean:
                    x_logits = torch.unsqueeze(x_logits, dim=0)
                x_logits = x_logits.numpy()
                X.append(x_logits)
            X = np.array(X)
            X = X.reshape(-1, X.shape[-1])
        else:
            x_logits_tot = np.empty((self.classif_inference.n_embedding, 0))
            weights = []
            if frame_duration != 1:
                chunk_size = frame_duration*n_mels_frames_per_s
                x_mels_sliced = [x[:, i:i+chunk_size] for i in range(0, x.shape[1], chunk_size)]
                for k, x_i in enumerate(x_mels_sliced):
                    #avoid giving to the model chunks of audio with too short length
                    if (((x_i.shape[1] == chunk_size) or (weighted_avg==True) or (mean==False)) & (x_i.shape[1]>=n_mels_frames_per_s+n_mels_frames_to_remove)):
                        weight = x_i.shape[1]/chunk_size
                        weights.append(weight)
                        x_logits = self.embedding_prediction(x_i, mean=mean)
                        x_logits_tot = np.concatenate((x_logits_tot, x_logits), axis=1)
            if weighted_avg:
                x_logits_tot = np.average(x_logits_tot, axis=1, weights=weights)
                x_logits_tot = np.expand_dims(x_logits_tot, axis=1)
            X = x_logits_tot
        return(X)











    # MIGHT BE USEFUL 
    # def wave_to_thirdo_to_logits(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, mean=True):
    #     chunk_size = self.tho_tr.sr
    #     x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
    #     x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
    #     for k, x_i in enumerate(x_sliced):
    #         x_mels_inf = self.wave_to_thirdo_to_mels_1s(x_i)
    #         if k == len(x_sliced)-1:   
    #             x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
    #         else:
    #             x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)
            
    #     x_mels_inf_tot = np.array(x_mels_inf_tot)

    #     x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
    #     if frame_duration != 1:
    #         chunk_size = frame_duration*n_mels_frames_per_s
    #         x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
    #         for k, x_i in enumerate(x_mels_sliced):
    #             #MT: recently added to avoid giving to the model random chunks of audio
    #             if x_i.shape[1] == chunk_size:
    #                 x_logits = self.mels_to_logit(x_i, torch_input=False, mean=mean)
    #                 x_logits_tot = np.concatenate((x_logits_tot, x_logits), axis=1)

    #     return(x_mels_inf_tot, x_logits_tot)

class MelDataset(object):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class ThirdOctaveToMelTranscoderDiffusion():
    '''
    Contains all the functions to transcode third-octave to mels and to make
    PANN predictions. 
    '''
    def __init__(self, model_name, model_path, device, dtype):
        '''
        Parameters:
        - model_name: str
            Specifies the filename (excluding the path) of the pre-trained model to be loaded.
        - model_path: str
            Specifies the directory where the model file and its corresponding settings file are stored.
        - device: torch.device
            Specifies the computing device (CPU or GPU) where the PyTorch model will be loaded and executed.
        - flen: int, optional (default=4096)
            Frame length for the third-octave spectrograms. 4096 is a fast third-octave spectrogram.
        - hlen: int, optional (default=4000)
            Hop length for the the third-octave spectrograms. 4000 is a fast third-octave spectrogram.
        - pann_type: str, optional (default='ResNet38')
            Type of PANN model to use. Only applicable if classifier is 'PANN'.
            Specifies the architecture or type of PANN (Pre-trained Audio Neural Network) model to use for classification.
        '''
        self.device = device
        self.dtype = dtype
        model_raw_name, _ = os.path.splitext(model_path + "/" + model_name)
        if 'chkpt' in model_raw_name:
            fname = model_raw_name.split("__")[0] + '_settings.yaml'
        else:
            fname = model_raw_name + '_settings.yaml'
        with open(fname) as file:
            settings_model = yaml.load(file, Loader=yaml.FullLoader)

        self.settings_model = settings_model

        self.model = UNet2DModel(
            #WARNING: This is only for square matrices, need to find a solution in case not square
            sample_size=settings_model.get('sample_size'),  # the target image resolution
            in_channels=2,  # the number of input channels, 3 for RGB images
            out_channels=1,  # the number of output channels
            layers_per_block=settings_model.get('layers_per_block'),  # how many ResNet layers to use per UNet block
            block_out_channels=settings_model.get('block_out_channels'),  # the number of output channes for each UNet block
            down_block_types=settings_model.get('down_block_types'), 
            up_block_types=settings_model.get('up_block_types'),
        )

        state_dict = torch.load(model_path + "/" + model_name, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # self.diffusion_inference = diffusion_inference
        if settings_model.get('schedule') == 'VE':
            self.noise_scheduler = ScoreSdeVeScheduler(num_train_timesteps=settings_model.get('diff_steps'))
            self.diffusion_inference = ScoreSdeVeInference(self.model, self.noise_scheduler,settings_model.get('diff_steps'))

        if settings_model.get('schedule') == 'DDPM':
            self.noise_scheduler = DDPMScheduler(num_train_timesteps=settings_model.get('diff_steps'), beta_schedule='sigmoid')
            self.diffusion_inference = DDPMInference(self.model, self.noise_scheduler,settings_model.get('diff_steps'))

        self.gomin_model = GomiGAN.from_pretrained(
        pretrained_model_path="transcoder/gomin/gan_state_dict.pt", **GANConfig().__dict__
        )
        self.gomin_model.eval()
        self.gomin_model.to(device)

        #chunkers used to chunk audio and spectrograms into overlapping macroscopic frames
        self.audio_chunker_mel = ut.AudioChunks(n=round(self.settings_model.get('mel')['sr']*self.settings_model.get('mel')['chunk_len']), hop=round(self.settings_model.get('mel')['sr']*self.settings_model.get('mel')['hop_len']))
        self.gomin_mel_chunker = ut.AudioChunks(n=self.settings_model.get('mel')['n_time'], hop=self.settings_model.get('mel')['n_time_hop'])
        self.audio_chunker_thopinv = ut.AudioChunks(n=round(self.settings_model.get('third_octave')['sr']*self.settings_model.get('mel')['chunk_len']), hop=round(self.settings_model.get('third_octave')['sr']*self.settings_model.get('mel')['hop_len']))
        self.thirdo_chunker = ut.AudioChunks(n=11, hop=10)

        self.tho_tr = bt.ThirdOctaveTransform(sr=self.settings_model.get('third_octave')['sr'], flen=4096, hlen=4000, refFreq=self.settings_model.get('third_octave')['ref_freq'], n_tho=self.settings_model.get('third_octave')['n_tho'], db_delta=0)
        self.mels_tr = bt.GominMelsTransform()

    def load_mel_chunks(self, fname, output_type='mel'):
        if output_type == 'thirdopinvmel':
            x = librosa.load(fname, sr=self.settings_model.get('third_octave')['sr'])[0]
            audio_n = self.audio_chunker_thopinv.chunks_with_hop(x)
            thirdo_specs = []
            for audio in audio_n:
                thirdo_spec = torch.from_numpy(self.tho_tr.wave_to_third_octave(audio)).T.unsqueeze(dim=0).type(self.dtype)
                spec_from_thirdo = pt.pinv(thirdo_spec, self.tho_tr, self.mels_tr, reshape=self.settings_model.get('mel')['n_time'])
                thirdo_specs.append(spec_from_thirdo)
            thirdo_specs = np.array(thirdo_specs)
            return(thirdo_specs)  
                  
        if output_type == 'gominmel':
            x = librosa.load(fname, sr=self.settings_model.get('mel')['sr'])[0]
            audio_n = self.audio_chunker_mel.chunks_with_hop(x)
            specs = []
            for audio in audio_n:
                spec = self.mels_tr.wave_to_mels(audio).squeeze(dim=0).T.type(self.dtype).cpu().numpy()
                specs.append(spec)
            specs = np.array(specs)
            return(specs)
        
    def load_thirdo_chunks(self, x):
        thirdo_n = self.thirdo_chunker.chunks_with_hop(x)
        thirdo_specs = []
        for thirdo in thirdo_n:
            spec_from_thirdo = pt.pinv(thirdo.unsqueeze(0), self.tho_tr, self.mels_tr, reshape=self.settings_model.get('mel')['n_time'])
            thirdo_specs.append(spec_from_thirdo)
        thirdo_specs = np.array(thirdo_specs)
        return(thirdo_specs)  
    
    def mels_to_mels(self, x, torch_output=False, batch_size=8):
        dataset = MelDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        tqdm_it=tqdm(dataloader, desc='EVALUATION')
        x_infs = None
        for (x) in tqdm_it:
            if torch_output:
                with torch.no_grad():
                    x = x.unsqueeze(dim=1)
                    x = x.to(self.device)
                    x = x * 2 - 1
                    x_inf = self.diffusion_inference.inference(x, device=self.device)
                    # wav_gomin = self.gomin_model(x_inf.squeeze(dim=1))
                    if x_infs is None:
                        x_infs = x_inf
                    else:
                        x_infs = torch.cat((x_infs, x_inf))
            else:
                x_inf = self.diffusion_inference.inference(x, device=self.device).detach()
                x_inf = x_inf[0].T.cpu().numpy()
        x_infs = x_infs.squeeze(dim=1).cpu().numpy()
        return(x_infs)

    def mels_to_audio(self, x, torch_output=False, batch_size=8):
        """
        Input x should be a 3D tensor of shape (batch_size, mel_bins, time_frames)
        """
        dataset = MelDataset(x)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        tqdm_it=tqdm(dataloader, desc='EVALUATION')
        wavs = None
        for (x) in tqdm_it:
            if torch_output:
                with torch.no_grad():
                    x = x.to(self.device)
                    x = x.type(self.dtype)
                    wav_gomin = self.gomin_model(x)
                    if wavs is None:
                        wavs = wav_gomin
                    else:
                        wavs = torch.cat((wavs, wav_gomin))
            else:
                x_inf = self.diffusion_inference.inference(x, device=self.device).detach()
                x_inf = x_inf[0].T.cpu().numpy()
        wavs = wavs.cpu().numpy()
        return(wavs)
    
    def thirdo_to_mels_to_audio(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], self.classif_inference.n_labels, 0)).to(self.device)
        for k, x_i in enumerate(x_sliced):
            x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
            if k == len(x_sliced)-1:   
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
            else:
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        if frame_duration != 1:
            chunk_size = frame_duration*n_mels_frames_per_s
            x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
            for k, x_i in enumerate(x_mels_sliced):
                if x_i.shape[1] == chunk_size:
                    x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True)
                    x_logits = x_logits.unsqueeze(dim=-1)
                    x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)

        return(x_mels_inf_tot, x_logits_tot)


    def thirdo_to_mels_to_logit(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, n_thirdo_frames_per_s=8):
        '''
        Converts a 1-second third-octave spectrogram into Mel spectrogram frames and corresponding logits.

        Parameters:
        - x: torch.Tensor
            The input third-octave data with a shape of (batch_size, time_frames, frequency_bins).
        - frame_duration: int, optional (default=10)
            The duration in seconds for each frame when slicing the waveform. Default is 10 seconds
            because PANN ResNet38 is optimal with 10s chunks.
        - n_mels_frames_to_remove: int, optional (default=1)
            The number of Mel spectrogram frames to remove from the end of each chunk to avoid overlap artifacts.
            Default is 1 (for PANN).
        - n_mels_frames_per_s: int, optional (default=100)
            The number of Mel spectrogram frames expected per second. Used to determine the chunk size.
            Default is 100 frames per second (for PANN).

        Returns:
        - Tuple of torch.Tensor
            A tuple containing:
            1. The Mel spectrogram frames with a shape of (batch_size, time_frames, mel_bins).
            2. The corresponding logits with a shape of (batch_size, n_labels), where n_labels
            is the number of classification labels (527 for PANN).
        '''
        n_thirdo_frames_per_s = 8
        x_sliced = [x[:, i:i+n_thirdo_frames_per_s, :] for i in range(0, x.shape[1], n_thirdo_frames_per_s)]
        x_mels_inf_tot = torch.empty((x.shape[0], 0, self.mels_tr.mel_bins)).to(self.device)
        x_logits_tot = torch.empty((x.shape[0], self.classif_inference.n_labels, 0)).to(self.device)
        for k, x_i in enumerate(x_sliced):
            x_mels_inf = self.thirdo_to_mels_1s(x_i, torch_output=True)
            if k == len(x_sliced)-1:   
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf), axis=1)
            else:
                x_mels_inf_tot = torch.cat((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)

        if frame_duration != 1:
            chunk_size = frame_duration*n_mels_frames_per_s
            x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size, :] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
            for k, x_i in enumerate(x_mels_sliced):
                if x_i.shape[1] == chunk_size:
                    x_logits = self.classifier_prediction(x_i, torch_input=True, torch_output=True)
                    x_logits = x_logits.unsqueeze(dim=-1)
                    x_logits_tot =torch.concatenate((x_logits_tot, x_logits), axis=-1)

        return(x_mels_inf_tot, x_logits_tot)



    # MIGHT BE USEFUL 
    # def wave_to_thirdo_to_logits(self, x, frame_duration=10, n_mels_frames_to_remove=1, n_mels_frames_per_s=100, mean=True):
    #     chunk_size = self.tho_tr.sr
    #     x_sliced = [x[i:i+chunk_size] for i in range(0, len(x), chunk_size)]
    #     x_mels_inf_tot = np.empty((self.mels_tr.mel_bins, 0))
    #     for k, x_i in enumerate(x_sliced):
    #         x_mels_inf = self.wave_to_thirdo_to_mels_1s(x_i)
    #         if k == len(x_sliced)-1:   
    #             x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf), axis=1)
    #         else:
    #             x_mels_inf_tot = np.concatenate((x_mels_inf_tot, x_mels_inf[:, :-n_mels_frames_to_remove]), axis=1)
            
    #     x_mels_inf_tot = np.array(x_mels_inf_tot)

    #     x_logits_tot = np.empty((self.classif_inference.n_labels, 0))
    #     if frame_duration != 1:
    #         chunk_size = frame_duration*n_mels_frames_per_s
    #         x_mels_sliced = [x_mels_inf_tot[:, i:i+chunk_size] for i in range(0, x_mels_inf_tot.shape[1], chunk_size)]
    #         for k, x_i in enumerate(x_mels_sliced):
    #             #MT: recently added to avoid giving to the model random chunks of audio
    #             if x_i.shape[1] == chunk_size:
    #                 x_logits = self.mels_to_logit(x_i, torch_input=False, mean=mean)
    #                 x_logits_tot = np.concatenate((x_logits_tot, x_logits), axis=1)

    #     return(x_mels_inf_tot, x_logits_tot)