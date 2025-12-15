import argparse
import librosa
import torch
import os
import numpy as np
from datetime import datetime, timedelta
import h5py
from pann.pann_inference import PANNsModel
from transcoder.transcoders import ThirdOctaveToMelTranscoder
from embs.model_loader import CLAPLaionModel, PANNsModel, PANNsThirdoModel, MSCLAPModel, VGGishModel
import sys

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

# Now you can safely import BEATs
from BEATs import BEATs, BEATsConfig

def get_beats_logits(audio_path, logits_path, all_models=None, sr=32000, device="cpu", batch_size=1):
    """
    Extracts logits (model outputs) from audio data stored in an HDF5 file using the BEATs model and saves them to an output HDF5 file.

    This function reads audio chunks from the input HDF5 file, processes them in batches, and computes logits
    using a BEATs model (either loaded from a checkpoint or provided via `all_models`). The resulting logits
    and corresponding datetime values are stored in a new HDF5 file at the specified output path.

    Args:
        audio_path (str): Path to the input HDF5 file containing audio data and datetime information.
        logits_path (str): Path to the output HDF5 file where logits and datetime will be saved.
        all_models (optional): An object providing a method `load_model("beats")` to load the BEATs model.
            If None, the model is loaded from a local checkpoint.
        sr (int, optional): Sample rate of the input audio. Default is 32000.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Default is "cpu".
        batch_size (int, optional): Number of audio samples to process per batch. Default is 1.

    Returns:
        None. The function saves logits and datetime information to the specified output HDF5 file.

    Note: BEATs doesn't provide audio embeddings per se, but rather a set of logits that can be used for classification tasks.
    It is put in the get_embeddings.py file for convenience, as its structure resembles the get_embeddings function.
    """

    if all_models is None:
        beats_ckpt_relative_path = "./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)
        # Load the fine-tuned checkpoints
        checkpoint = torch.load(beats_ckpt_full_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
        BEATs_model.to(device)
    else:
        BEATs_model = all_models.load_model("beats")

    with h5py.File(audio_path, "r") as input_hf:
        audio_dataset = input_hf["audio"]
        input_datetime_dataset = input_hf["datetime"]
        num_rows = audio_dataset.shape[0]  # Get the total number of rows in the input dataset

        # Open the output HDF5 file for writing
        with h5py.File(logits_path, "w") as output_hf:
            # Create pre-allocated datasets for embeddings and datetime values
            logits_dim = 527  # Assuming the embedding dimension
            logit_dataset = output_hf.create_dataset("logits", 
                                                          (num_rows, logits_dim), 
                                                          dtype="float32")
            output_hf.create_dataset("datetime", data=input_datetime_dataset)

            # batch loop
            # Iterate through audio chunks and compute embeddings
            for start_idx in range(0, num_rows, batch_size):
                end_idx = min(start_idx + batch_size, num_rows)
                audio_chunk = audio_dataset[start_idx:end_idx]

                y_beats = librosa.resample(audio_chunk, orig_sr=sr, target_sr=16000)

                # Predict with the model
                with torch.no_grad():
                    input = torch.tensor(y_beats).to(device)
                    padding_mask = torch.zeros(input.size(0), input.size(1)).bool().to(device)  # Assuming no padding for the single audio
                    logit = BEATs_model.extract_features(input, padding_mask=padding_mask)[0].detach().cpu().numpy()

                # Store the embedding and timestamp directly at the pre-allocated index
                logit_dataset[start_idx:end_idx] = logit

                print(f"CHUNK {start_idx}/{num_rows}                ", end="\r")

def get_embeddings(audio_path, embeddings_path, emb="clap", all_models=None, sr=32000, device="cpu", batch_size=64):
    """
    Extracts audio embeddings from an HDF5 file containing audio data and saves them to another HDF5 file.

    This function loads audio data from the specified input HDF5 file, processes it in batches, and computes embeddings
    using the selected audio embedding model. The resulting embeddings and corresponding datetime information are stored
    in the output HDF5 file.
    Args:
        audio_path (str): Path to the input HDF5 file containing audio data and datetime information.
        embeddings_path (str): Path to the output HDF5 file where embeddings and datetime will be saved.
        emb (str, optional): The embedding model to use. Options include "clap", "pann", "pann-thirdo", "msclap", "vggish".
            Defaults to "clap".
        all_models (object, optional): Preloaded models object. If provided, loads the specified model from this object.
            Defaults to None.
        sr (int, optional): Sample rate of the input audio. Defaults to 32000.
        device (str, optional): Device to run the model on ("cpu" or "cuda"). Defaults to "cpu".
        batch_size (int, optional): Number of audio samples to process per batch. Defaults to 64.
    Returns:
        None
    Notes:
        - The function assumes the input HDF5 file contains datasets named "audio" and "datetime".
        - Embeddings are computed using the specified model and stored in the output file under the "embeddings" dataset.
        - The function prints progress information during processing.
    """

    if all_models is None:
        if emb == "clap":
            clap_model = CLAPLaionModel(type="audio", device=device)
        elif emb == "pann":
            clap_model = PANNsModel(variant="wavegram-logmel", device=device)
        elif emb == "pann-thirdo":
            clap_model = PANNsThirdoModel(device=device)
        elif emb == "msclap":
            clap_model = MSCLAPModel(type="2023", device=device)
        elif emb == "vggish":
            clap_model = VGGishModel(device=device)
        clap_model.load_model()
    else:
        clap_model = all_models.load_model(emb)
    
    with h5py.File(audio_path, "r") as input_hf:
        audio_dataset = input_hf["audio"]
        input_datetime_dataset = input_hf["datetime"]
        num_rows = audio_dataset.shape[0]  # Get the total number of rows in the input dataset

        # Open the output HDF5 file for writing
        with h5py.File(embeddings_path, "w") as output_hf:
            # Create pre-allocated datasets for embeddings and datetime values
            embedding_dataset = output_hf.create_dataset("embeddings", 
                                                          (num_rows, clap_model.num_features), 
                                                          dtype="float32")
            output_hf.create_dataset("datetime", data=input_datetime_dataset)

            # batch loop
            # Iterate through audio chunks and compute embeddings
            for start_idx in range(0, num_rows, batch_size):
                end_idx = min(start_idx + batch_size, num_rows)
                audio_chunk = audio_dataset[start_idx:end_idx]

                if sr != clap_model.sr:
                    y_clap = librosa.resample(audio_chunk, orig_sr=sr, target_sr=clap_model.sr)
                else:
                    y_clap = audio_chunk
                
                # Predict with the model
                with torch.no_grad():
                    embedding = clap_model._get_embedding(y_clap)
                    embedding = embedding.detach().cpu().numpy()

                # Store the embedding and timestamp directly at the pre-allocated index
                embedding_dataset[start_idx:end_idx] = embedding

                print(f"CHUNK {start_idx}/{num_rows}                ", end="\r")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files.")
    parser.add_argument("audio_path", type=str, help="Path to the h5 file that contains the audios.")
    parser.add_argument("embeddings_path", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("--emb", type=str, default="clap", help="Embedding type to use (default: clap).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the embedding computation.")
    parser.add_argument("--sr", type=int, default=32000, help="Sample rate (default to 32000).")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('USING DEVICE:', DEVICE)

    if args.emb in ["clap", 'msclap', "pann", "pann-thirdo", "vggish"]:
        get_embeddings(args.audio_path, args.embeddings_path, args.emb, None, args.sr, DEVICE, args.batch_size)
