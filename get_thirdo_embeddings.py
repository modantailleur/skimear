import argparse
import torch
import numpy as np
import h5py
from transcoder.transcoders import ThirdOctaveToMelTranscoder

def get_pann_thirdo_embeddings(thirdo_path, embeddings_path, batch_size=16):
    """
    Generate embeddings for audio files in the specified folder and save them to an HDF5 file.
    :param dataset_dir: Path to the folder containing the h5 files with fast third-octave data.
    :param embeddings_path: Path to the output HDF5 file
    :param block_length: Length of the audio block in seconds
    :param sr: Sampling rate to open the audio files. Pann uses 32kHz so sr should always be 32000.
                If the audio to store needs to be at another rate, the code needs to be modified.
    """

    MODEL_PATH = "./transcoder/ckpt"
    cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
    transcoder = 'cnn_pinv'
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        #MT: add
        device = torch.device("cpu")

    transcoder_deep_bce = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device, pann_type='CNN14')

    with h5py.File(thirdo_path, "r") as input_hf:
        thirdo_dataset = input_hf["thirdo"]
        input_datetime_dataset = input_hf["datetime"]
        num_rows = thirdo_dataset.shape[0]  # Get the total number of rows in the input dataset

        # Open the output HDF5 file for writing
        with h5py.File(embeddings_path, "w") as output_hf:
            # Create pre-allocated datasets for embeddings and datetime values
            # 8s thirdo data, and 8 thirdo bins per second
            num_output_rows = num_rows
            embedding_dataset = output_hf.create_dataset("embeddings", 
                                                          (num_output_rows, 2048), 
                                                          dtype="float32")
            output_datetime_dataset = output_hf.create_dataset("datetime", data=input_datetime_dataset)

            # Iterate through audio chunks and compute embeddings
            for start_idx in range(0, num_rows, batch_size):
                end_idx = min(start_idx + batch_size, num_rows)
                thirdo_chunk = thirdo_dataset[start_idx:end_idx]
                thirdo_chunk = torch.from_numpy(thirdo_chunk).float().to(device)

                # Predict with the model
                with torch.no_grad():
                    if thirdo_chunk.shape[1] < 8*8:
                        _ , embedding = transcoder_deep_bce.thirdo_to_mels_to_embeddings(thirdo_chunk, frame_duration=thirdo_chunk.shape[1]//8)
                    else:
                        _ , embedding = transcoder_deep_bce.thirdo_to_mels_to_embeddings(thirdo_chunk, frame_duration=8)
                    embedding = embedding.detach().cpu().numpy()
                    embedding = embedding.reshape(-1, embedding.shape[-1])
                    embedding = np.swapaxes(embedding, 0, 1)

                # Store the embedding and timestamp directly at the pre-allocated index
                embedding_dataset[start_idx:end_idx] = embedding

                print(f"CHUNK {start_idx}/{num_output_rows}                ", end="\r")

# def get_pann_thirdo_embeddings(thirdo_path, embeddings_path, batch_size=4800, db_offset=-88):
#     """
#     Generate embeddings for audio files in the specified folder and save them to an HDF5 file.
#     :param dataset_dir: Path to the folder containing the h5 files with fast third-octave data.
#     :param embeddings_path: Path to the output HDF5 file
#     :param block_length: Length of the audio block in seconds
#     :param sr: Sampling rate to open the audio files. Pann uses 32kHz so sr should always be 32000.
#                 If the audio to store needs to be at another rate, the code needs to be modified.
#     """

#     MODEL_PATH = "./reference_models"
#     cnn_logits_name = 'classifier=PANN+dataset=full+dilation=1+epoch=200+kernel_size=5+learning_rate=-3+nb_channels=64+nb_layers=5+prop_logit=100+step=train+transcoder=cnn_pinv+ts=1_model'
#     transcoder = 'cnn_pinv'
#     force_cpu = False
#     #manage gpu
#     useCuda = torch.cuda.is_available() and not force_cpu

#     if useCuda:
#         print('Using CUDA.')
#         #MT: add
#         device = torch.device("cuda:0")
#     else:
#         print('No CUDA available.')
#         #MT: add
#         device = torch.device("cpu")

#     transcoder_deep_bce = ThirdOctaveToMelTranscoder(transcoder, cnn_logits_name, MODEL_PATH, device=device, pann_type='CNN14')

#     with h5py.File(thirdo_path, "r") as input_hf:
#         dataset = input_hf["fast_125ms"][:]
#         dataset = np.asarray([tuple(row) for row in dataset])
#         thirdo_dataset = dataset[:, 1:] + db_offset  # Add db_offset to the third-octave data

#         # Add 8 columns of zeros at the beginning and 1 column of zeros at the end
#         # MT: ro remove after debugging
#         zeros_start = np.zeros((thirdo_dataset.shape[0], 8), dtype=thirdo_dataset.dtype)
#         zeros_end = np.zeros((thirdo_dataset.shape[0], 1), dtype=thirdo_dataset.dtype)
#         thirdo_dataset = np.hstack([zeros_start, thirdo_dataset, zeros_end])

#         epoch_dataset = dataset[:, 0]
#         df = pd.DataFrame({'epoch': epoch_dataset})
#         datetime_dataset = pd.to_datetime(df['epoch'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S.%f').values
#         num_rows = thirdo_dataset.shape[0]  # Get the total number of rows in the input dataset

#         # Open the output HDF5 file for writing
#         with h5py.File(embeddings_path, "w") as output_hf:
#             # Create pre-allocated datasets for embeddings and datetime values
#             # 8s thirdo data, and 8 thirdo bins per second
#             num_output_rows = num_rows // (8*8)
#             embedding_dataset = output_hf.create_dataset("embeddings", 
#                                                           (num_output_rows, 2048), 
#                                                           dtype="float32")
#             output_datetime_dataset = output_hf.create_dataset("datetime", (num_output_rows), dtype="S26")

#             # Iterate through audio chunks and compute embeddings
#             for start_idx in range(0, num_rows, batch_size):
#                 end_idx = min(start_idx + batch_size, num_rows)
#                 thirdo_chunk = thirdo_dataset[start_idx:end_idx]
#                 thirdo_chunk = torch.from_numpy(thirdo_chunk).float().to(device)
#                 thirdo_chunk = thirdo_chunk.unsqueeze(0)  # Add channel dimension
#                 # Predict with the model
#                 with torch.no_grad():
#                     if thirdo_chunk.shape[1] < 8*8:
#                         _ , embedding = transcoder_deep_bce.thirdo_to_mels_to_embeddings(thirdo_chunk, frame_duration=thirdo_chunk.shape[1]//8)
#                     else:
#                         _ , embedding = transcoder_deep_bce.thirdo_to_mels_to_embeddings(thirdo_chunk, frame_duration=8)
#                     embedding = embedding.detach().cpu().numpy()
#                     embedding = embedding.reshape(-1, embedding.shape[-1])
#                     embedding = np.swapaxes(embedding, 0, 1)

#                 output_start_index = start_idx // (8*8)
#                 output_end_index = end_idx // (8*8)
#                 # Store the embedding and timestamp directly at the pre-allocated index
#                 embedding_dataset[output_start_index:output_end_index] = embedding

#                 datatimes = datetime_dataset[start_idx:end_idx]
#                 datetimes = datatimes[::8*8]

#                 output_datetime_dataset[output_start_index:output_end_index] = datetimes.astype('S26')

#                 print(f"CHUNK {output_start_index}/{num_output_rows}                ", end="\r")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files.")
    parser.add_argument("thirdo_path", type=str, help="Path to the h5 file that contains the fast third-octave data.")
    parser.add_argument("embeddings_path", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the embedding computation.")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('USING DEVICE:', DEVICE)

    get_pann_thirdo_embeddings(args.thirdo_path, args.embeddings_path, batch_size=args.batch_size)
