import h5py
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
# from panns_model import PANNsModel
import librosa
from dtaidistance import dtw, dtw_ndim
import torch
from utils.summary_utils import batch_embeddings_generator,\
    get_embedding_from_row, get_logit_from_row, logits_generator
import pandas as pd
import os
import sys
from embs.model_loader import CLAPLaionModel, MSCLAPModel, PANNsModel, VGGishModel
import warnings
import json
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")
warnings.filterwarnings("ignore", message="Some weights of.*are newly initialized")
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release.*")
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated.*")
warnings.filterwarnings("ignore", message="Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized.*")

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

# Now you can safely import BEATs
from BEATs import BEATs, BEATsConfig

def get_summary_emb_from_data_manual(summary_path, block_length, emb, all_models=None, sr=32000, device="cpu", batch_size=12, emb_ckpt_dir=None):

    if all_models is None:
        if emb == "clap":
            clap_model = CLAPLaionModel(type="audio", device=device, emb_ckpt_dir=emb_ckpt_dir)
        elif emb == "pann":
            clap_model = PANNsModel(variant="wavegram-logmel", device=device, emb_ckpt_dir=emb_ckpt_dir)
        elif emb == "msclap":
            clap_model = MSCLAPModel(type="2023", device=device, emb_ckpt_dir=emb_ckpt_dir)
        elif emb == "vggish":
            clap_model = VGGishModel(device=device, emb_ckpt_dir=emb_ckpt_dir)
        clap_model.load_model()
    else:
        clap_model = all_models.load_model(emb)

    # Load the audio file
    y, _ = librosa.load(summary_path, sr=sr, mono=True)

    # Define chunk parameters
    block_samples = int(block_length * sr)
    overlap_samples = int(3 * sr)  # 3-second overlap
    step_samples = block_samples - overlap_samples

    # Determine the start indices for each chunk
    start_indices = range(0, len(y) - block_samples + 1, step_samples)

    embeddings = []
    audio_chunks = []

    # Prepare audio chunks
    for start_idx in start_indices:
        end_idx = start_idx + block_samples
        audio_chunk = y[start_idx:end_idx]

        # Resample to 48kHz if necessary
        y_clap = librosa.resample(audio_chunk, orig_sr=sr, target_sr=clap_model.sr)
        audio_chunks.append(y_clap)

    # Process in batches
    for i in range(0, len(audio_chunks), batch_size):
        batch = audio_chunks[i:i + batch_size]
        batch = np.array(batch)
        if batch.shape == 2:
            batch = np.expand_dims(batch, axis=1)  # Adding batch dimension if it's 2D

        with torch.no_grad():
            embedding = clap_model._get_embedding(batch)
            embedding = embedding.detach().cpu().numpy()
            embeddings.extend(embedding)

        print(f"Processed batch {i // batch_size + 1}/{(len(audio_chunks) + batch_size - 1) // batch_size}", end="\r")

    embeddings = np.array(embeddings)
    return embeddings

def get_summary_logits_from_data_manual(summary_path, block_length, all_models=None, sr=32000, device="cpu", verbose=False, batch_size=12):
    
    if all_models is None:
        # Load BEATs model
        beats_ckpt_relative_path = "./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)
        checkpoint = torch.load(beats_ckpt_full_path)

        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
        BEATs_model.to(device)
    else:
        BEATs_model = all_models.load_model("beats")

    # Load the audio file
    y, _ = librosa.load(summary_path, sr=sr)

    # Define chunk parameters
    block_samples = int(block_length * sr)
    overlap_samples = int(3 * sr)  # 3-second overlap
    step_samples = block_samples - overlap_samples

    # Determine the start indices for each chunk
    start_indices = range(0, len(y) - block_samples + 1, step_samples)

    logits_list = []
    audio_chunks = []

    # Prepare audio chunks
    for start_idx in start_indices:
        end_idx = start_idx + block_samples
        audio_chunk = y[start_idx:end_idx]

        # Resample to 16kHz if necessary
        y_beats = librosa.resample(audio_chunk, orig_sr=sr, target_sr=16000)
        audio_chunks.append(y_beats)

    # Process in batches
    for i in range(0, len(audio_chunks), batch_size):
        batch = np.array(audio_chunks[i:i + batch_size])
        input_tensor = torch.from_numpy(batch).to(device)  # Batch dimension added

        padding_mask = torch.zeros(input_tensor.size(0), input_tensor.size(1)).bool().to(device)  # No padding

        with torch.no_grad():
            logits = BEATs_model.extract_features(input_tensor, padding_mask=padding_mask)[0]

            if verbose:
                # Load the ontology
                ontology_path = "./utils/audioset_ontology.json"
                with open(ontology_path, "r", encoding="utf-8", errors="replace") as f:
                    ontology_data = json.load(f)

                # Create a mapping from Freebase IDs to human-readable names
                id_to_name = {entry["id"]: entry["name"] for entry in ontology_data}

                for j, (top5_label_prob, top5_label_idx) in enumerate(zip(*logits.topk(k=5))):
                    top5_label = [
                        id_to_name.get(checkpoint['label_dict'][label_idx.item()], "Unknown label")
                        for label_idx in top5_label_idx
                    ]
                    print(f'Top 5 predicted labels of the {j}th audio are {top5_label} with probability of {top5_label_prob}')

            logits = logits.detach().cpu().numpy()
            logits_list.extend(logits)

        print(f"Processed batch {i // batch_size + 1}/{(len(audio_chunks) + batch_size - 1) // batch_size}", end="\r")

    # Compute logits summary and max logits summary
    logits_summary = np.array(logits_list)
    max_logits_summary = logits_summary.max(axis=0)

    return logits_summary, max_logits_summary

def get_data_emb(embeddings_path, clusters_path, start_datetime, end_datetime, period=15):
    # If the file doesn't exist or clusters_path is None, generate mean_embeddings as before
    embeddings = np.array(
        [batch[0] for batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=1, start_datetime=start_datetime, end_datetime=end_datetime)],
        dtype=object
    )

    mean_embeddings = np.array([np.mean(x, axis=0) for x in embeddings])
    # try:
    #     if clusters_path is None:
    #         raise FileNotFoundError("clusters_path is None")

    #     # Split clusters_path and replace the last part with "meanembeddings.npy"
    #     path_parts = clusters_path.split("_")
    #     path_parts[-1] = "meanembeddings.npy"
    #     meanembeddings_path = "_".join(path_parts)
        
    #     # Load mean_embeddings from the numpy file
    #     mean_embeddings = np.load(meanembeddings_path, allow_pickle=True)
    # except (FileNotFoundError, AttributeError):
    #     # If the file doesn't exist or clusters_path is None, generate mean_embeddings as before
    #     embeddings = np.array(
    #         [batch[0] for batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=1, start_datetime=start_datetime, end_datetime=end_datetime)],
    #         dtype=object
    #     )

    #     mean_embeddings = np.array([np.mean(x, axis=0) for x in embeddings])

    return(mean_embeddings)

def get_data_logits(logits_path, clusters_path, start_datetime, end_datetime):

    try:
        if clusters_path is None:
            raise FileNotFoundError("clusters_path is None")
        
        # Split clusters_path and replace the last part with "maxlogits.npy"
        path_parts = clusters_path.split("_")
        path_parts[-1] = "maxlogits.npy"
        maxlogits_path = "_".join(path_parts)

        # Load max_logits from the numpy file
        max_logits = np.load(maxlogits_path)
    except (FileNotFoundError, AttributeError):
        # If the file doesn't exist, generate max_logits as before
        full_out = np.array(list(logits_generator(logits_path, start_datetime=start_datetime, end_datetime=end_datetime)), dtype=object)
        logits = np.array([full_out[elem, 0] for elem in range(len(full_out))])
        max_logits = np.max(logits, axis=0)

    return max_logits


def get_summary_emb_from_data(summary_setting_path, embeddings_path):
    
    rows_to_retrieve = pd.read_csv(summary_setting_path)['row'].dropna().astype(int).values
    embeddings_summary = np.array([get_embedding_from_row(embeddings_path, row) for row in rows_to_retrieve])

    return(embeddings_summary)

def get_summary_logits_from_data(summary_setting_path, logits_path):
    
    rows_to_retrieve = pd.read_csv(summary_setting_path)['row'].dropna().astype(int).values
    logits_summary = np.array([get_logit_from_row(logits_path, row) for row in rows_to_retrieve])
    max_logits_summary = np.max(logits_summary, axis=0)

    return(logits_summary, max_logits_summary)

#WITH CALIBRATION
def get_emb_dtw(mean_embeddings_data, mean_embeddings_summary):

    distance = dtw_ndim.distance(mean_embeddings_data, mean_embeddings_summary)

    path = dtw.warping_path(mean_embeddings_data, mean_embeddings_summary, use_ndim=True)
    axis_1_values = [x[1] for x in path]
    unique_values, counts = np.unique(axis_1_values, return_counts=True)
    value_counts_dict = dict(zip(unique_values, counts))
    print(value_counts_dict)

    output = distance

    print('DTW PANN DISTANCE')
    print(output)

    return(output)

def ms_iou(max_logits, max_logits_summary, top=50):

    ious = []
    step = 0.01
    for threshold in [step*k for k in range(1, int(1/step))]:

        within_threshold = max_logits > threshold
        within_threshold_summary = max_logits_summary > threshold

        num_classes = np.sum(within_threshold_summary)

        if top is not None:
            if (num_classes < top) or (num_classes > len(max_logits) - top):
                continue

        intersection = np.logical_and(within_threshold, within_threshold_summary)
        union = np.logical_or(within_threshold, within_threshold_summary)

        if np.sum(union) == 0:
            continue

        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)

    mean_iou = np.mean(ious)
    return(mean_iou)

def get_logit_max_similarity(max_logits_data, max_logits_summary, top_pann_to_keep=50, verbose=False):

    if verbose:
        # Load BEATs model
        beats_ckpt_relative_path = "./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
        beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)
        checkpoint = torch.load(beats_ckpt_full_path)

        torch_max_logits = torch.tensor(max_logits_data).unsqueeze(0)
        torch_max_logits_summary = torch.tensor(max_logits_summary).unsqueeze(0)

        # Load the ontology
        ontology_path = "./utils/audioset_ontology.json"
        with open(ontology_path, "r", encoding="utf-8", errors="replace") as f:
            ontology_data = json.load(f)

        # Create a mapping from Freebase IDs to human-readable names
        id_to_name = {entry["id"]: entry["name"] for entry in ontology_data}

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*torch_max_logits.topk(k=20))):
            top5_label = [
                id_to_name.get(checkpoint['label_dict'][label_idx.item()], "Unknown label")
                for label_idx in top5_label_idx
            ]
            print('MAX LOGITS FULL-LENGTH AUDIO')
            print(top5_label)
            print(top5_label_prob)

        for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*torch_max_logits_summary.topk(k=20))):
            top5_label = [
                id_to_name.get(checkpoint['label_dict'][label_idx.item()], "Unknown label")
                for label_idx in top5_label_idx
            ]
            print('MAX LOGITS SUMMARY')
            print(top5_label)
            print(top5_label_prob)

    mean_iou = ms_iou(max_logits_data, max_logits_summary, top=top_pann_to_keep)

    output = mean_iou

    return(output)
