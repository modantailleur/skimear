
import argparse
from get_embeddings import get_beats_logits
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for audio files.")
    parser.add_argument("audio_path", type=str, help="Path to the h5 file that contains the audios.")
    parser.add_argument("logits_path", type=str, help="Path to the output HDF5 file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for the embedding computation.")
    parser.add_argument("--sr", type=int, default=32000, help="Sample rate (default to 32000).")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('USING DEVICE:', DEVICE)

    # COMPUTING BEATS LOGITS
    get_beats_logits(args.audio_path, args.logits_path, None, args.sr, DEVICE, args.batch_size)