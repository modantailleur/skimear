from abc import ABC, abstractmethod
import logging
from typing import Literal
import numpy as np
import soundfile
import os
import sys
from urllib.parse import unquote

import torch
from torch import nn
from pathlib import Path
from hypy_utils.downloader import download_file
import importlib.metadata

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pann import models as panns
from transcoder.transcoders import ThirdOctaveToMelTranscoder

log = logging.getLogger(__name__)


class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int, min_len: int = -1, device: str = torch.device("cpu")):
        """
        Args:
            name (str): A unique identifier for the model.
            num_features (int): Number of features in the output embedding (dimensionality).
            sr (int): Sample rate of the audio.
            min_len (int, optional): Enforce a minimal length for the audio in seconds. Defaults to -1 (no minimum).
        """
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.min_len = min_len
        self.device = device

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        
        # Enforce minimum length
        wav_data = self.enforce_min_len(wav_data)

        return wav_data
    
    def enforce_min_len(self, audio: np.ndarray) -> np.ndarray:
        """
        Enforce a minimum length for the audio. If the audio is too short, output a warning and pad it with zeros.
        """
        if self.min_len < 0:
            return audio
        if audio.shape[0] < self.min_len * self.sr:
            log.warning(
                f"Audio is too short for {self.name}.\n"
                f"The model requires a minimum length of {self.min_len}s, audio is {audio.shape[0] / self.sr:.2f}s.\n"
                f"Padding with zeros."
            )
            audio = np.pad(audio, (0, int(np.ceil(self.min_len * self.sr - audio.shape[0]))))
            print()
        return audio

class CLAPLaionModel(ModelLoader):
    """
    CLAP model from https://github.com/LAION-AI/CLAP
    """
    
    def __init__(self, type: Literal['audio', 'music'], device: str = "cpu", emb_ckpt_dir: str = None):

        super().__init__(f"clap-laion-{type}", 512, 48000, device=device)
        self.type = type

        if type == 'audio':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt'
        elif type == 'music':
            url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt'

        if emb_ckpt_dir is None:
            self.emb_ckpt_dir = Path(__file__).parent / ".model-checkpoints"
        else:
            self.emb_ckpt_dir = Path(emb_ckpt_dir)
        
        self.model_file = self.emb_ckpt_dir / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)
            
        # Patch the model file to remove position_ids (will raise an error otherwise)
        # This key must be removed for CLAP version <= 1.1.5
        # But it must be kept for CLAP version >= 1.1.6
        package_name = "laion_clap"
        from packaging import version
        ver = version.parse(importlib.metadata.version(package_name))
        if ver < version.parse("1.1.6"):
            self.patch_model_430(self.model_file)
        else:
            self.unpatch_model_430(self.model_file)


    def patch_model_430(self, file: Path):
        """
        Patch the model file to remove position_ids (will raise an error otherwise)
        This is a new issue after the transformers 4.30.0 update
        Please refer to https://github.com/LAION-AI/CLAP/issues/127
        """
        # Create a "patched" file when patching is done
        patched = file.parent / f"{file.name}.patched.430"
        if patched.exists():
            return
        
        log.warning("Patching LAION-CLAP's model checkpoints")
        
        # Load the checkpoint from the given path
        ck = torch.load(file, map_location="cpu")

        # Extract the state_dict from the checkpoint
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck

        # Delete the specific key from the state_dict
        sd.pop("module.text_branch.embeddings.position_ids", None)

        # Save the modified state_dict back to the checkpoint
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd

        # Save the modified checkpoint
        torch.save(ck, file)
        log.warning(f"Saved patched checkpoint to {file}")
        
        # Create a "patched" file when patching is done
        patched.touch()
            

    def unpatch_model_430(self, file: Path):
        """
        Since CLAP 1.1.6, its codebase provided its own workarounds that isn't compatible
        with our patch. This function will revert the patch to make it compatible with the new
        CLAP version.
        """
        patched = file.parent / f"{file.name}.patched.430"
        if not patched.exists():
            return
        
        # The below is an inverse operation of the patch_model_430 function, so comments are omitted
        log.warning("Unpatching LAION-CLAP's model checkpoints")
        ck = torch.load(file, map_location="cpu")
        unwrap = isinstance(ck, dict) and "state_dict" in ck
        sd = ck["state_dict"] if unwrap else ck
        sd["module.text_branch.embeddings.position_ids"] = 0
        if isinstance(ck, dict) and "state_dict" in ck:
            ck["state_dict"] = sd
        torch.save(ck, file)
        log.warning(f"Saved unpatched checkpoint to {file}")
        patched.unlink()
        
        
    def load_model(self, verbose=False):
        import laion_clap

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny' if self.type == 'audio' else 'HTSAT-base')
        self.model.load_ckpt(self.model_file, verbose=False)
        self.model.to(self.device)

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:

        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.get_audio_embedding_from_data(x = chunk, use_tensor=True)
                embeddings.append(emb)

        emb = torch.stack(embeddings, dim=0)
        emb = torch.mean(emb, dim=0)

        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

class MSCLAPModel(ModelLoader):
    """
    CLAP model from https://github.com/microsoft/CLAP
    """
    def __init__(self, type: Literal['2023'], device: str = "cpu", emb_ckpt_dir: str = None):
        super().__init__(f"clap-{type}", 1024, 44100, device=device)
        self.type = type

        if type == '2023':
            url = 'https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth'

        if emb_ckpt_dir is None:
            self.emb_ckpt_dir = Path(__file__).parent / ".model-checkpoints"
        else:
            self.emb_ckpt_dir = Path(emb_ckpt_dir)

        self.model_file = self.emb_ckpt_dir / url.split('/')[-1]

        # Download file if it doesn't exist
        if not self.model_file.exists():
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            download_file(url, self.model_file)

        
    def load_model(self):
        from msclap import CLAP
        
        use_cuda = "cuda" in str(self.device)
        self.model = CLAP(self.model_file, version = self.type, use_cuda=use_cuda)

    def _get_embedding(self, audio: np.ndarray, pad: bool = True) -> np.ndarray:

        # Split the audio into 7s chunks with 1s hop
        chunk_size = 7 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # zero-pad chunks to make equal length
        clen = [x.shape[1] for x in chunks]
        chunks = [np.pad(ch, ((0,0), (0,np.max(clen) - ch.shape[1]))) for ch in chunks]

        self.model.default_collate(chunks)

        # Calculate embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.clap.audio_encoder(chunk)[0]
                embeddings.append(emb)

        # Concatenate the embeddings
        emb = torch.stack(embeddings, dim=0)
        emb = torch.mean(emb, dim=0)

        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

class PANNsModel(ModelLoader):
    """
    Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

    Specify the model to use (cnn14-32k, cnn14-16k, wavegram-logmel).
    You can also specify wether to send the full provided audio or 1-s chunks of audio (cnn14-32k-1s). This was shown 
    to have a very low impact on performances.
    """
    def __init__(self, variant: Literal['cnn14-32k', 'wavegram-logmel'], device: str = "cpu", emb_ckpt_dir: str = None):
        super().__init__(f"panns-{variant}", 2048, 
                         sr=16000 if variant == 'cnn14-16k' else 32000, device=device)
        self.variant = variant
        if emb_ckpt_dir is None:
            self.emb_ckpt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".model-checkpoints/")
        else:
            self.emb_ckpt_dir = os.path(emb_ckpt_dir)

    def load_model(self):
        os.makedirs(self.emb_ckpt_dir, exist_ok=True)

        # Mapping variants to checkpoint files
        ckpt_urls = {
            'cnn14-16k': "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
            'cnn14-32k': "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
            'wavegram-logmel': "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
        }
        ckpt_files = {key: unquote(url.split('/')[-1]) for key, url in ckpt_urls.items()}

        # Check and download the specific checkpoint file if not present
        ckpt_file = ckpt_files.get(self.variant, None)

        if ckpt_file:
            ckpt_path = os.path.join(self.emb_ckpt_dir, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"Downloading checkpoint for {self.variant}...")
                os.system(f"wget -P {self.emb_ckpt_dir} {ckpt_urls[self.variant]}")

        features_list = ["2048", "logits"]

        # Load the corresponding model and checkpoint
        if self.variant == 'cnn14-16k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Cnn14_16k_mAP=0.438.pth"))
            self.model.load_state_dict(state_dict["model"])

        elif self.variant == 'cnn14-32k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Cnn14_mAP=0.431.pth"))
            self.model.load_state_dict(state_dict["model"])

        elif 'wavegram-logmel' in self.variant:
            self.model = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Wavegram_Logmel_Cnn14_mAP=0.439.pth"))
            self.model.load_state_dict(state_dict["model"])

        self.model.eval()
        self.model.to(self.device)

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'PANNs: {self.num_params}')

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        
        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        emb_str = '2048' if 'cnn14' in self.variant else 'embedding'
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.forward(chunk)[emb_str]
                embeddings.append(emb)

        emb = torch.stack(embeddings, dim=0)
        emb = torch.mean(emb, dim=0)

        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)

class PANNsThirdoModel(ModelLoader):
    """
    This model computes fast third-octave on the audio, then uses the transcoder [1] to transform 
    fast third-octaves into higher Mel spectro-temporal super-resolution, and then uses PANNs [2] to
    generate the audio embeddings. It is just used to simulate what would happen whenever you record
    fast third-octaves instead of audio.

    [1] Tailleur, M., Lagrange, M., Aumond, P., & Tourre, V. (2023, September). Spectral trancoder: using 
    pretrained urban sound classifiers on undersampled spectral representations. In 8th Workshop on Detection 
    and Classification of Acoustic Scenes and Events (DCASE).

    [2] Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.
    """
    def __init__(self, device: str = "cpu"):
        super().__init__(f"panns-thirdo", 2048, 32000, device=device)

    def load_model(self):
        model_path = "./transcoder_models"
        transcoder_name = 'cnn_pinv'
        cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
        self.transcoder = ThirdOctaveToMelTranscoder(transcoder_name, cnn_logits_name, model_path, device=self.device, pann_type='CNN14')

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        
        embeddings = None
        for audio_sample in audio:
            x_mels_deep_bce = self.transcoder.wave_to_thirdo_to_mels(audio_sample)
            embedding = self.transcoder.mels_to_embedding(x_mels_deep_bce).T[0]
            embedding = torch.from_numpy(embedding).to(self.device)
            if embeddings is None:
                embeddings = embedding.unsqueeze(0)
            else:
                embeddings = torch.cat((embeddings, embedding.unsqueeze(0)), dim=0)

        return embeddings

class VGGishModel(ModelLoader):
    """
    S. Hershey et al., "CNN Architectures for Large-Scale Audio Classification", ICASSP 2017
    """
    def __init__(self, use_pca=False, use_activation=False, audio_len=None, device: str = "cpu", emb_ckpt_dir: str = Path(__file__).parent / ".model-checkpoints"):
        super().__init__("vggish", 128, 16000, audio_len, device=device)
        self.use_pca = use_pca
        self.use_activation = use_activation

    def load_model(self):
        self.model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        if not self.use_pca:
            self.model.postprocess = False
        if not self.use_activation:
            self.model.embeddings = nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'VGGish: {self.num_params}')

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        embeddings = []
        for chunk in audio:
            emb = self.model.forward(chunk, self.sr)
            emb = emb.mean(dim=0)
            embeddings.append(emb)
        emb = torch.stack(embeddings, dim=0)

        return emb