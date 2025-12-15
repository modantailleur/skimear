import librosa
import os
from datetime import datetime, timedelta
import h5py
import argparse

def get_audio(dataset_dir, audio_path, block_length, sr=32000, orig_utc=0, target_utc=0):
    """
    Processes audio files in a specified directory, splits them into blocks, resamples them, and stores the resulting audio chunks along with their corresponding timestamps in an HDF5 file.

    Args:
        dataset_dir (str): Path to the directory containing audio files (.wav, .mp3, .WAV).
        audio_path (str): Path to the output HDF5 file where processed audio and timestamps will be stored.
        block_length (int): Length (in seconds) of each audio block to extract from the files.
        sr (int, optional): Target sampling rate for resampling audio. Defaults to 32000.
        orig_utc (int, optional): Original UTC offset of the audio files. Defaults to 0.
        target_utc (int, optional): Target UTC offset to adjust timestamps. Defaults to 0.

    Notes:
        - Audio files are expected to have their date and time encoded in the filename in the format: ..._YYYYMMDD_HHMMSS.ext
        - Each audio file is split into consecutive blocks of length `block_length` seconds.
        - Audio blocks shorter than the specified length are skipped.
        - Timestamps are adjusted according to the difference between `orig_utc` and `target_utc`.
        - The resulting audio chunks and their timestamps are stored in datasets named "audio" and "datetime" within the HDF5 file.
    """

    # Initialize an empty list to store audio arrays
    wav_generators = []
    srs = []
    init_times = []
    # Iterate through all files in the folder
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".WAV"):  # Assuming all audio files are in WAV format
            file_path = os.path.join(dataset_dir, filename)
            # Load the audio file
            sri = librosa.get_samplerate(file_path)
            audio_chunks = librosa.stream(file_path, block_length=block_length, frame_length=sri, hop_length=sri, mono=True)
            srs.append(sri)
            wav_generators.append(audio_chunks)
            # Parse the date and time from the file name
            date_str, time_str = filename.rsplit(".", 1)[0].rsplit("_", 2)[-2:]
            # Convert the date and time strings into a datetime object
            datetime_obj = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            # Adjust datetime_obj by the difference between target_utc and orig_utc
            datetime_obj += timedelta(hours=(target_utc - orig_utc))
            init_times.append(datetime_obj)

    # Sort init_times and corresponding wav_generators and srs by datetime_obj
    sorted_indices = sorted(range(len(init_times)), key=lambda i: init_times[i])
    init_times = [init_times[i] for i in sorted_indices]
    wav_generators = [wav_generators[i] for i in sorted_indices]
    srs = [srs[i] for i in sorted_indices]

    # Create the HDF5 file
    with h5py.File(audio_path, "w") as hf:
        # Create datasets for embeddings and datetime values
        audio_dataset = hf.create_dataset("audio", (0,int(sr*block_length)), maxshape=(None,int(sr*block_length)), dtype="float32")
        datetime_dataset = hf.create_dataset("datetime", (0,), maxshape=(None,), dtype="S26")  # Adjust the string length as needed

        for idx, (audio_chunks, orig_sr, init_time) in enumerate(zip(wav_generators, srs, init_times)):
            for idx_chunk, audio_chunk in enumerate(audio_chunks):
                y_sr = librosa.resample(audio_chunk, orig_sr=orig_sr, target_sr=sr)

                #checks if the audio chunk is long enough
                if len(y_sr) < block_length*sr:
                    continue
                seconds_to_add = block_length*idx_chunk
                delta = timedelta(seconds=seconds_to_add)
                y_time = init_time + delta

                # Append logit to logit dataset
                audio_dataset.resize(audio_dataset.shape[0] + 1, axis=0)
                audio_dataset[-1] = y_sr

                # Append datetime value to datetime dataset
                datetime_str = y_time.strftime("%Y-%m-%d %H:%M:%S.%f")
                datetime_dataset.resize(datetime_dataset.shape[0] + 1, axis=0)
                datetime_dataset[-1] = datetime_str

                print(f"FILE {idx}/{len(wav_generators)}, chunk {idx_chunk}          ", end="\r")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files.")
    parser.add_argument("dataset_dir", type=str, help="Path to the folder containing the audio files.")
    parser.add_argument("audio_h5_dir", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("block_length", type=int, nargs="?", default=8, help="Length of a block of audio (in seconds).")
    parser.add_argument("sr", type=int, nargs="?", default=32000, help="Sample rate, set by default to 32000.")

    args = parser.parse_args()

    get_audio(args.dataset_dir, args.audio_h5_dir, args.block_length, args.sr)