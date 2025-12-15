import torch
import numpy as np
import librosa
from torchaudio.transforms import Fade
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS, CONVTASNET_BASE_LIBRI2MIX
import random 
import torchaudio
from utils.random_splicing import random_splicing
from speechbrain.inference.interfaces import Pretrained
import torch.nn.functional as F

class RandomAudioChunks():
    def __init__(self, segmin, segmax, seghopmin_perc=0.10, seghopmax_perc=0.90, fade=True):
        # window size limits
        self.segmin = segmin
        self.segmax = segmax
        # hop size limits as percentages of the window size
        self.seghopmin_perc = seghopmin_perc
        self.seghopmax_perc = seghopmax_perc
        # whether to apply a fade in and fade out on each chunk or not
        self.fade = fade

        # To store window lengths and hop lengths
        self.wins = []
        self.diffhops = []
        self.pos_l = []
        self.pos_r = []

    def chunks_with_hop(self, lst, mixframe=True, chunk_size=32000):

        L = []
        idx = 0
        while idx < len(lst):
            # Choose a random window size between segmin and segmax
            window_size = random.randint(self.segmin, self.segmax)
            self.pos_l.append(idx)
            self.pos_r.append(idx+window_size)
            self.wins.append(window_size)  # Store window size
            L.append(lst[self.pos_l[-1]:self.pos_r[-1]])
            idx = idx + window_size
        L[-1] = lst[self.pos_l[-1]:len(lst)]
        self.wins[-1] = len(L[-1])
        self.pos_r[-1] = len(lst)

        if mixframe:
            combined = list(zip(L, self.wins, self.pos_l, self.pos_r))
            chunk_size = max(1, int(chunk_size // (len(lst) / len(L))))   # Number of chunks per group to shuffle
            shuffled_combined = []

            for i in range(0, len(combined), chunk_size):
                group = combined[i:i + chunk_size]  # Get the group of chunks
                random.shuffle(group)  # Shuffle only this group if shuffle is True
                shuffled_combined.extend(group)  # Add the group (shuffled or not) to the result

            L, self.wins, self.pos_l, self.pos_r = zip(*shuffled_combined)
            L = list(L)
            self.wins = list(self.wins)
            self.pos_l = list(self.pos_l)
            self.pos_r = list(self.pos_r)

        self.newpos_l = list(self.pos_l)
        self.newpos_r = list(self.pos_r)

        for i in range(1, len(L)):
            minwinsize = min([self.wins[i], self.wins[i-1]])
            hop_perc = random.uniform(self.seghopmin_perc, self.seghopmax_perc)
            diffhop = int((1-hop_perc) * minwinsize)
            self.diffhops.append(diffhop)
            diffhop_l = diffhop // 2
            diffhop_r = diffhop - diffhop_l

            self.newpos_l[i] = self.newpos_l[i] - diffhop_l
            self.newpos_r[i-1] = self.newpos_r[i-1] + diffhop_r

        for i, (pos_l, pos_r, newpos_l, newpos_r) in enumerate(zip(self.pos_l, self.pos_r, self.newpos_l, self.newpos_r)):
            if newpos_l < 0:
                L[i] = np.concatenate((L[i][pos_l:2*pos_l-newpos_l][::-1],L[i],lst[pos_r:newpos_r]))

            elif newpos_r > len(lst):
                L[i] = np.concatenate((lst[newpos_l:pos_l], L[i], lst[2*pos_r-newpos_r:pos_r][::-1]))

            else:
                L[i] = np.concatenate((lst[newpos_l:pos_l],L[i],lst[pos_r:newpos_r]))
            self.wins[i] = len(L[i])

        return L

    def concat_with_hop(self, L):
        # Initialize the output list with the correct size
        total_length = sum(len(chunk) for chunk in L) - sum(diffhop for diffhop in self.diffhops)

        lst = np.zeros(shape=total_length)

        # Keep track of the position in the output list
        current_position = 0

        for i, chunk in enumerate(L):
            chunk_length = len(chunk)

            # Calculate diffhop as 5% of the current chunk size (overlap)
            if (i == (len(L)-1)):
                diffhop = 0
            else:
                diffhop = self.diffhops[i]

            # If fade is enabled, calculate the fade in/out windows
            if self.fade:
                bef = np.linspace(0, 1, diffhop)
                aft = np.linspace(1, 0, diffhop)
                mid = np.ones(chunk_length - diffhop)
                pond_g = np.concatenate((bef, mid))
                pond_d = np.concatenate((mid, aft))
            else:
                pond_g = np.ones(chunk_length)
                pond_d = np.ones(chunk_length)

            # If it's the first chunk, simply copy it to the output list
            if i == 0:
                lst[current_position:current_position + chunk_length] = pond_d * chunk
                current_position += chunk_length
            else:
                # Calculate the starting position in the output list considering overlap
                overlap_position = current_position - diffhop

                # Apply the overlap
                if overlap_position + chunk_length > len(lst):
                    break

                if overlap_position < 0:
                    break

                lst[overlap_position:overlap_position+chunk_length] += pond_g * pond_d * chunk

                # Update the position for the next chunk
                current_position += (chunk_length - diffhop)
        
        return lst

class AudioChunks():
    def __init__(self, n, hop, fade=True):
        #number of elements in each chunk
        self.n = n
        #size of hop
        self.hop = hop
        self.diffhop = n - hop

        #whether to apply a fade in and fade out on each chunk or not
        self.fade = fade

    def calculate_num_chunks(self, wavesize):
        num_chunks = 1
        idx = 0
        audio_truncated=False
        
        if self.n == self.diffhop:
            step = self.n
        else:
            step = self.n-self.diffhop

        for i in range(self.n, wavesize-self.n+self.diffhop, step):
            num_chunks += 1
            idx = i

        if idx+2*(self.n-self.diffhop) == wavesize:
            num_chunks += 1
        else:
            audio_truncated=True

        if self.n == self.diffhop:
            if self.n*num_chunks == wavesize:
                audio_truncated=False
            else:
                audio_truncated=True
            
        return(num_chunks, audio_truncated)
    
    def chunks_with_hop(self, lst):
        if isinstance(lst, np.ndarray):
            return self._chunks_with_hop_np(lst)
        elif isinstance(lst, torch.Tensor):
            return self._chunks_with_hop_torch(lst)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _chunks_with_hop_np(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return np.array(L)

    def _chunks_with_hop_torch(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return torch.stack(L)

    def concat_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_with_hop_np(self, L):
        lst = np.zeros(shape=L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[1] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))
        else:
            pond_g = np.ones(L.shape[1])
            pond_d = np.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def _concat_with_hop_torch(self, L):
        lst = torch.zeros(L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[1] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))
        else:
            pond_g = torch.ones(L.shape[1])
            pond_d = torch.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def concat_spec_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_spec_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_spec_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_spec_with_hop_np(self, L):
        lst = np.zeros(shape=(L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[2] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))

            pond_g = np.tile(pond_g, (L.shape[1], 1))
            pond_d = np.tile(pond_d, (L.shape[1], 1))
        else:
            pond_g = np.ones((L.shape[1], L.shape[2]))
            pond_d = np.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

    def _concat_spec_with_hop_torch(self, L):
        lst = torch.zeros((L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[2] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))

            pond_g = pond_g.repeat(L.shape[1], 1)
            pond_d = pond_d.repeat(L.shape[1], 1)
        else:
            pond_g = torch.ones((L.shape[1], L.shape[2]))
            pond_d = torch.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

def roll_with_crossfade(audio, roll_amount, fade_duration=None):
    """
    Rolls an audio signal with a crossfade at the roll position to avoid clicks.
    
    Parameters:
        audio (numpy.ndarray): The audio signal (1D numpy array for mono, 2D for stereo).
        roll_amount (int): Number of samples to roll the audio by.
        fade_duration (int): Duration of the crossfade in samples.
        
    Returns:
        numpy.ndarray: The rolled audio signal with crossfade applied.
    """
    if fade_duration is None:
        fade_duration = len(audio) // 10
    if roll_amount < fade_duration:
        fade_duration = roll_amount
    if roll_amount > len(audio) - fade_duration:
        fade_duration = len(audio) - roll_amount

    # Roll the audio
    rolled_audio = np.roll(audio, roll_amount)
    
    # Create crossfade window
    fade_in = np.linspace(0, 1, fade_duration)
    fade_out = np.linspace(1, 0, fade_duration)

    # Apply crossfade at the roll boundary        
    rolled_audio[roll_amount-fade_duration:roll_amount] = \
        rolled_audio[roll_amount-fade_duration:roll_amount] * fade_out + \
        rolled_audio[roll_amount:roll_amount+fade_duration][::-1] * fade_in

    return rolled_audio


def blurring_mfcc(audio, sr, n_mfcc=5):
    """
    Apply MFCC transformation, keep only the first n_mfcc components, 
    and reverse the MFCC to reconstruct the audio.

    Parameters:
        audio (numpy.ndarray): Input audio signal.
        sr (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCC components to keep.

    Returns:
        numpy.ndarray: Reconstructed audio signal.
    """
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Reconstruct the spectrogram from the MFCCs
    mfccs_reconstructed = librosa.feature.inverse.mfcc_to_mel(mfccs)

    # Reconstruct the audio signal from the spectrogram
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(mfccs_reconstructed, sr=sr)

    return audio_reconstructed

class SourceSeparation:
    def __init__(self, method="music", device="cpu"):
        """
        Initialize the SourceSeparation class.

        Args:
            method (str): The method to use for source separation. 
                            Options are "music" or "environmental".
        """
        if method not in ["music", "env"]:
            raise ValueError("Invalid method. Choose 'music' or 'env'.")
        self.method = method
        if method == "music":
            bundle = HDEMUCS_HIGH_MUSDB_PLUS if method == "music" else CONVTASNET_BASE_LIBRI2MIX
            self.model = bundle.get_model()
            self.sr = bundle.sample_rate
        else:
            self.model = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')
            self.model.device = device
            self.sr = 16000

        self.device = device
        self.model = self.model.to(self.device)

    def separate(self, audio, sr, segment=10.0, overlap=0.1):
        """
        Separate the sources from the input audio.

        Args:
            audio (numpy.ndarray): Input audio signal.
            sr (int): Sampling rate of the audio.
            segment (float): Segment length in seconds.
            overlap (float): Overlap between segments as a fraction.

        Returns:
            dict: A dictionary containing separated sources.
        """
        if self.method == "music":

            waveform = torch.tensor(audio, dtype=torch.float32).repeat(2, 1).to(self.device)
            ref = waveform.mean(0)
            waveform = (waveform - ref.mean()) / ref.std()  # Normalize
            sources = separate_sources(self.model, waveform[None], device=self.device, segment=segment, overlap=overlap, sample_rate=sr)[0]
            sources = sources * ref.std() + ref.mean()
            sources_list = self.model.sources
            sources = list(sources)
            audios = dict(zip(sources_list, sources))
            voice = audios['vocals']
            other = sum(audios[key] for key in audios.keys() if key != 'vocals')
            y_voice = voice.detach().cpu().numpy()[0]
            y_other = other.detach().cpu().numpy()[0]
        else:
            sr = self.sr
            sr_model = 16000

            # We download the audio file from our storage. Feel free to download another file and use audio from a specific path
            waveform = torch.tensor(audio, dtype=torch.float32)
            waveform = waveform.to(self.device)
            waveform = waveform.unsqueeze(0)

            input_new_sr = waveform

            # resample the data if needed
            if sr != sr_model:
                print(
                    "Resampling the audio from {} Hz to {} Hz".format(
                        sr, sr_model
                    )
                )
                tf = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=sr_model
                ).to(self.model.device)
                input_new_sr = waveform.mean(dim=0, keepdim=True)
                input_new_sr = tf(input_new_sr)

            voice = self.model.separate_file(input=input_new_sr)

            # Resample the separated voice to the original sampling rate if needed
            if sr != sr_model:
                print(
                    "Resampling the separated voice from {} Hz to {} Hz".format(
                        sr_model, sr
                    )
                )
                tf_reverse = torchaudio.transforms.Resample(
                    orig_freq=sr_model, new_freq=sr
                ).to(voice.device)
                voice = tf_reverse(voice)

            # Remove the projection to get the pseudo-background, otherwise the voice is not scaled and 
            # thus input-voice doesn't give a proper background
            y_other = remove_projection(waveform, voice).detach().cpu().numpy()[0]
            y_voice = voice.detach().cpu().numpy()[0]

        return y_voice, y_other

def blurring_random_splicing(y_voice, sr, segmin, segmax, seghopmin_perc=1.0, seghopmax_perc=1.0, p_reverse=0.0, framehopmin_perc=0, mixframe=True, framemin=None):

    meta_chunker = RandomAudioChunks(segmin=segmin, segmax=segmax, seghopmin_perc=seghopmin_perc, seghopmax_perc=seghopmax_perc)
    y_n_voice = meta_chunker.chunks_with_hop(y_voice, mixframe=False)
    y_n_r_voice = [random_splicing(y_segment, sr, p_reverse=p_reverse, mixframe=mixframe, overlap=(1-framehopmin_perc), frame_length=framemin, top_db=6) for y_segment in y_n_voice]
    y_r_voice = meta_chunker.concat_with_hop(y_n_r_voice)

    return y_r_voice

def blurring_reverse(y_voice,  segmin, segmax, seghopmin_perc=0.50, seghopmax_perc=0.90, mixframe=True, chunk_size=32000, randphase=True, reverse="forward"):
    """
    Apply random time reversal to the audio signal. 
    Not used in the paper, but alternate solution to the segmentation of Burkhardt, without any RMSE computation.
    """
    chunker = RandomAudioChunks(segmin=segmin, segmax=segmax, seghopmin_perc=seghopmin_perc, seghopmax_perc=seghopmax_perc)

    # SHORT TERM TIME RANDOM TIME REVERSAL: chunk array into chunks of random windows, shuffle them and invert the windows
    y_n_voice = chunker.chunks_with_hop(y_voice, mixframe=mixframe, chunk_size=chunk_size)

    if reverse == "backward":
        order = [-1 for _ in y_n_voice]
    elif reverse == "forward":
        order = [1 for _ in y_n_voice]
    else:
        order = [random.choice([-1, 1]) for _ in y_n_voice]

    if randphase:
        y_n_r_voice = [roll_with_crossfade(y[::order_i], random.randint(len(y) // 10, len(y) - len(y) // 10), 
                                    fade_duration=random.randint(len(y) // 10, len(y) // 5)) \
                    for _, (y, order_i) in enumerate(zip(y_n_voice, order))]
    else:
        y_n_r_voice = [y[::order_i] for _, (y, order_i) in enumerate(zip(y_n_voice, order))]

    y_r_voice = chunker.concat_with_hop(y_n_r_voice)

    return(y_r_voice)

def blur_audio_for_figure(y, sr, segmin, segmax, framemin, seghopmin_perc=0.95, seghopmax_perc=0.95, framehopmin_perc=0.80, mixframe=True, reverse="forward",
                                                      device="cpu", source_sep=None, vad_model=None, vad_sr=None, vad_speech_index=None, vad_speech_ths=0.5, method='reverse'):
    """
    reverse: "random", "forward", "backward"
    """

    if source_sep is not None:
        y_voice, y_other = source_sep.separate(y, sr)
    else:
        y_voice = y
        y_other = np.zeros_like(y)

    if method == 'random_splicing_reverse':
        p_reverse = 1.0 if reverse == "backward" else 0.0 if reverse == "forward" else 0.5
        y_r_voice = blurring_random_splicing(y_voice, sr, segmin, segmax, seghopmin_perc, seghopmax_perc, p_reverse=p_reverse, framehopmin_perc=framehopmin_perc, mixframe=mixframe, framemin=framemin)
    elif method == 'mfcc':
        y_r_voice = blurring_mfcc(y_voice, sr)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    # chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))
    chunker_thresh = AudioChunks(n=round(0.5*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)
    y_n_is_speech = np.zeros(len(y_n), dtype=bool)
        
    # Resample the audio to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=vad_sr)
    y_beats = resampler(torch.tensor(y_n, dtype=torch.float32))

    # Normalize y_beats to max absolute value
    y_beats = y_beats / torch.max(torch.abs(y_beats))

    # Pad y_beats with zeros until it reaches 0.5s
    target_length = int(0.5 * 16000)  # 0.5 seconds at 16kHz
    if y_beats.shape[0] < target_length:
        padding = target_length - y_beats.shape[0]
        y_beats = torch.nn.functional.pad(y_beats, (0, padding))

    if vad_model is not None:
        # Predict with the model
        with torch.no_grad():
            input_tensor = y_beats.clone().detach().to(device)
            padding_mask = torch.zeros(input_tensor.size(0), input_tensor.size(1)).bool().to(device)  # Assuming no padding for the single audio
            speech_logit = vad_model.extract_features(input_tensor, padding_mask=padding_mask)[0][:, vad_speech_index]

        y_n_is_speech = speech_logit > vad_speech_ths
    else:
        y_n_is_speech = np.ones(len(y_n), dtype=bool)
    
    y_n_t = []

    for y_i, y_i_r_voice, y_i_other, y_i_is_speech in zip(y_n, y_n_r_voice, y_n_other, y_n_is_speech):
        if y_i_is_speech:
            y_n_t.append(y_i_r_voice + y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    if len(y_combined) < len(y):
        y_combined = np.concatenate((y_combined, y[len(y_combined):]))

    return(y_combined, y_voice, y_r_voice, y_other)


def blur_audio(y, sr, segmin, segmax, framemin, seghopmin_perc=0.95, seghopmax_perc=0.95, framehopmin_perc=0.80, mixframe=True, reverse="forward",
                                                      device="cpu", source_sep=None, vad_model=None, vad_sr=None, vad_speech_index=None, vad_speech_ths=0.5, method='reverse'):
    """
    reverse: "random", "forward", "backward"
    """

    if source_sep is not None:
        y_voice, y_other = source_sep.separate(y, sr)
    else:
        y_voice = y.copy()
        y_other = np.zeros_like(y)

    if method == 'random_splicing_reverse':
        p_reverse = 1.0 if reverse == "backward" else 0.0 if reverse == "forward" else 0.5
        y_r_voice = blurring_random_splicing(y_voice, sr, segmin, segmax, seghopmin_perc, seghopmax_perc, p_reverse=p_reverse, framehopmin_perc=framehopmin_perc, mixframe=mixframe, framemin=framemin)
    elif method == 'mfcc':
        y_r_voice = blurring_mfcc(y_voice, sr)

    max_length = len(y)
    y_r_voice = np.pad(y_r_voice, (0, max(0, max_length - len(y_r_voice))), 'constant')[:max_length]
    y_other = np.pad(y_other, (0, max(0, max_length - len(y_other))), 'constant')[:max_length]

    # chunker_thresh = AudioChunks(n=round(0.500*sr), hop=round(0.440*sr))
    chunker_thresh = AudioChunks(n=round(0.5*sr), hop=round(0.440*sr))

    y_n = chunker_thresh.chunks_with_hop(y)
    y_n_r_voice = chunker_thresh.chunks_with_hop(y_r_voice)
    y_n_other = chunker_thresh.chunks_with_hop(y_other)
    y_n_is_speech = np.zeros(len(y_n), dtype=bool)
        
    # Resample the audio to 16kHz
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=vad_sr)
    y_beats = resampler(torch.tensor(y_n, dtype=torch.float32))

    # Normalize y_beats to max absolute value
    y_beats = y_beats / torch.max(torch.abs(y_beats))

    # Pad y_beats with zeros until it reaches 0.5s
    target_length = int(0.5 * 16000)  # 0.5 seconds at 16kHz
    if y_beats.shape[0] < target_length:
        padding = target_length - y_beats.shape[0]
        y_beats = torch.nn.functional.pad(y_beats, (0, padding))

    if vad_model is not None:
        # Predict with the model
        with torch.no_grad():
            input_tensor = y_beats.clone().detach().to(device)
            padding_mask = torch.zeros(input_tensor.size(0), input_tensor.size(1)).bool().to(device)  # Assuming no padding for the single audio
            speech_logit = vad_model.extract_features(input_tensor, padding_mask=padding_mask)[0][:, vad_speech_index]
        y_n_is_speech = speech_logit > vad_speech_ths
    else:
        y_n_is_speech = np.ones(len(y_n), dtype=bool)
    
    y_n_t = []

    for y_i, y_i_r_voice, y_i_other, y_i_is_speech in zip(y_n, y_n_r_voice, y_n_other, y_n_is_speech):
        temp = y_i_r_voice + y_i_other
        if y_i_is_speech:
            y_n_t.append(y_i_r_voice + y_i_other)
        else:
            y_n_t.append(y_i)

    # Convert y_combined to a numpy array if needed
    y_n_t = np.array(y_n_t)
    y_combined = chunker_thresh.concat_with_hop(y_n_t)

    # 20ms fade out for the remaining part
    if len(y_combined) < len(y):
        fade_length = int(0.02*sr)  # 20ms fade
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        y_combined = np.concatenate((
            y_combined[:-fade_length],
            y_combined[-fade_length:] * fade_out +
            y[len(y_combined)-fade_length:len(y_combined)] * fade_in,
            y[len(y_combined):]
        ))

    return(y_combined)

def separate_sources(
    model,
    mix,
    segment=10.0,
    overlap=0.1,
    device=None,
    sample_rate=32000,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def separate_sources_env(
    model,
    mix,
    device=None,
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    mixture = mix.reshape(1, 1, -1)
    estimated_sources = model(mixture)
    return estimated_sources


class SepformerSeparation(Pretrained):
    """A "ready-to-use" speech separation model.

    Uses Sepformer architecture.

    Example
    -------
    >>> tmpdir = getfixture("tmpdir")
    >>> model = SepformerSeparation.from_hparams(
    ...     source="speechbrain/sepformer-wsj02mix",
    ...     savedir=tmpdir)
    >>> mix = torch.randn(1, 400)
    >>> est_sources = model.separate_batch(mix)
    >>> print(est_sources.shape)
    torch.Size([1, 400, 2])
    """

    MODULES_NEEDED = ["encoder", "masknet", "decoder"]

    def separate_batch(self, mix):
        """Run source separation on batch of audio.

        Arguments
        ---------
        mix : torch.Tensor
            The mixture of sources.

        Returns
        -------
        tensor
            Separated sources
        """

        # Separation
        mix = mix.to(self.device)
        mix_w = self.mods.encoder(mix)
        est_mask = self.mods.masknet(mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.mods.decoder(sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]
        return est_source

    def separate_file(self, input):
        """Separate sources from file.

        Arguments
        ---------
        path : str
            Path to file which has a mixture of sources. It can be a local
            path, a web url, or a huggingface repo.
        savedir : path
            Path where to store the wav signals (when downloaded from the web).
        Returns
        -------
        tensor
            Separated sources
        """

        batch = input.to(self.device)

        est_sources = self.separate_batch(batch)
        est_sources = (
            est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
        )
        return est_sources[:, :, 0]

    def forward(self, mix):
        """Runs separation on the input mix"""
        return self.separate_batch(mix)
    
def remove_projection(input, voice):
    # Flatten (remove batch) and normalize
    x = input[0]
    v = voice[0]

    # Project input onto voice
    scale = torch.dot(x, v) / torch.dot(v, v)
    projected = scale * v

    # Subtract projection to get pseudo-background
    background = x - projected
    return background.unsqueeze(0)