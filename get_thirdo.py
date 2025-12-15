import librosa
import os
from datetime import datetime, timedelta
import h5py
import argparse
import pandas as pd
import numpy as np

def get_thirdo(dataset_path, output_path, block_length, db_offset=-88):

    with h5py.File(dataset_path, "r") as input_hf:
        dataset = input_hf["fast_125ms"][:]
        dataset = np.asarray([tuple(row) for row in dataset])

        thirdo_dataset = dataset[:, 1:] + db_offset  # Add db_offset to the third-octave data

        epoch_dataset = dataset[:, 0]
        df = pd.DataFrame({'epoch': epoch_dataset})
        datetime_dataset = pd.to_datetime(df['epoch'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S.%f').values
        num_rows = thirdo_dataset.shape[0]  # Get the total number of rows in the input dataset


        # Ensure the directory for output_path exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create the HDF5 file
        with h5py.File(output_path, "w") as hf:
            # Create datasets for embeddings and datetime values
            output_thirdo_dataset = hf.create_dataset("thirdo", (0,int(8*block_length),29), maxshape=(None,int(8*block_length),29), dtype="float32")
            output_datetime_dataset = hf.create_dataset("datetime", (0,), maxshape=(None,), dtype="S26")  # Adjust the string length as needed

            for idx in range(0, num_rows, 8 * block_length):
                if idx + 8 * block_length >= num_rows:
                    continue  # Skip if there are not enough rows for a full block
                thirdo_chunk = thirdo_dataset[idx:idx + 8 * block_length]
                input_time = datetime_dataset[idx] if idx < len(datetime_dataset) else b""
                # Store the chunk and its initial datetime
                # Append logit to logit dataset
                output_thirdo_dataset.resize(output_thirdo_dataset.shape[0] + 1, axis=0)
                output_thirdo_dataset[-1] = thirdo_chunk

                # Append datetime value to datetime dataset
                output_datetime_dataset.resize(output_datetime_dataset.shape[0] + 1, axis=0)
                output_datetime_dataset[-1] = input_time

# UNUSED CODE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files.")
    parser.add_argument("dataset_path", type=str, help="Path to the input thirdo h5 file.")
    parser.add_argument("thirdo_h5_dir", type=str, help="Path to the output thirdo HDF5 file.")
    parser.add_argument("block_length", type=int, nargs="?", default=8, help="Length of a block of audio (in seconds).")

    args = parser.parse_args()

    get_thirdo(args.dataset_path, args.thirdo_h5_dir, args.block_length)