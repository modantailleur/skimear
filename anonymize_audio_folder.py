from anonymizer import blur_audio, SourceSeparation
import librosa
import soundfile as sf
import numpy as np
import os 
import argparse
import torch
import shutil
from pydub import AudioSegment
import json
import sys

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

from BEATs import BEATs, BEATsConfig

def anonymize_audio(args):

    do_sep = args.do_sep
    do_vad = args.do_vad

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else f"{input_dir.rstrip(os.sep)}_anonymized"
    convert_to_mp3 = args.convert_to_mp3

    force_cpu = False
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_sep = SourceSeparation(args.sep_method, device=device) if do_sep else None

    ###########################################
    ###### BEATS loading

    beats_ckpt_relative_path = "./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    beats_ckpt_full_path = os.path.abspath(beats_ckpt_relative_path)

    # Load the fine-tuned checkpoints
    checkpoint = torch.load(beats_ckpt_full_path, map_location=device)

    if do_vad:
        cfg = BEATsConfig(checkpoint['cfg'])
        BEATs_model = BEATs(cfg)
        BEATs_model.load_state_dict(checkpoint['model'])
        BEATs_model.eval()
        BEATs_model.to(device)
    else:
        BEATs_model = None
    # Load the ontology
    ontology_path = "./utils/audioset_ontology.json"
    with open(ontology_path, "r") as f:
        ontology_data = json.load(f)

    # Create a mapping from Freebase IDs to human-readable names
    id_to_name = {entry["id"]: entry["name"] for entry in ontology_data}
    name_to_id = {entry["name"]: entry["id"] for entry in ontology_data}

    speech_id = name_to_id['Speech']
    speech_index = next((key for key, value in checkpoint['label_dict'].items() if value == speech_id), None)
    speech_ths = 0.3
    #####
    #################################################

    sr = source_sep.sr if source_sep is not None else 44100  # Target sample rate
    batch_duration = 120  # Duration of each batch in seconds
    batch_size = sr * batch_duration
    segmin = int(sr * args.segmin)
    segmax = int(sr * args.segmax)
    seghopmin_perc = args.seghopmin_perc
    seghopmax_perc = args.seghopmax_perc
    framemin = int(sr * args.framemin) if args.framemin is not None else None
    framehopmin_perc = args.framehopmin_perc
    mixframe = args.mixframe
    reverse = args.reverse

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        output_subfolder = os.path.join(output_dir, os.path.relpath(root, input_dir))
        os.makedirs(output_subfolder, exist_ok=True)

        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_subfolder, file)
            mp3_output_filename = os.path.splitext(file)[0] + ".mp3"
            mp3_output_path = os.path.join(output_subfolder, mp3_output_filename)

            if os.path.exists(mp3_output_path):
                print(f"MP3 file already exists: {mp3_output_path}, skipping...")
                continue

            if file.lower().endswith(('.wav', '.mp3', '.flac')):
                with sf.SoundFile(input_path) as sf_file:
                    original_sr = sf_file.samplerate
                    chunk_index = 0
                    with sf.SoundFile(output_path, mode='w', samplerate=sr, channels=1) as output_file:
                        while sf_file.tell() < len(sf_file):
                            chunk = sf_file.read(int(batch_duration * original_sr), dtype='float32')
                            if len(chunk) == 0:
                                break
                            if sf_file.channels > 1:
                                chunk = np.mean(chunk, axis=1)
                            resampled_chunk = librosa.resample(chunk, orig_sr=original_sr, target_sr=sr)
                            processed_chunk = blur_audio(resampled_chunk, sr, segmin, 
                                                                          segmax, framemin, seghopmin_perc=seghopmin_perc, 
                                                                          seghopmax_perc=seghopmax_perc, framehopmin_perc=framehopmin_perc,
                                                                          mixframe=mixframe,
                                                                          reverse=reverse, device=device, 
                                                                          source_sep=source_sep,
                                                                          vad_model=BEATs_model, 
                                                                          vad_sr=16000,
                                                                          vad_speech_index=speech_index,
                                                                          vad_speech_ths=speech_ths,
                                                                          method=args.method)
                            output_file.write(processed_chunk)
                            chunk_index += 1

                            # Print progress
                            print(f"Processing chunk {chunk_index + 1} at position {sf_file.tell() / original_sr:.2f} seconds")

                print(f"Stage 1: WAV processing complete! Saved to '{output_path}'")
                if convert_to_mp3:
                    audio_segment = AudioSegment.from_wav(output_path)
                    audio_segment.export(mp3_output_path, format="mp3", bitrate="128k")
                    os.remove(output_path)
                    print(f"Stage 2: Converted to MP3 and saved to '{mp3_output_path}'")
            else:
                shutil.copy2(input_path, output_path)
    print("Processing complete! All audio files processed successfully.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anonymize all audio files in the input folder.")
    parser.add_argument('-i', '--input_dir', required=True, help="Path to the input folder containing audio files.")
    parser.add_argument('-o', '--output_dir', default=None, help="Path to the output folder. Defaults to input folder with '_anonymized' suffix.")
    parser.add_argument('-mp3', '--convert_to_mp3', default=False, help="Convert output files to MP3 format. Defaults to False.")
    parser.add_argument('-vad', '--do_vad', type=str2bool, default=True, help="Enable voice activity detection (VAD). Defaults to True.")
    parser.add_argument('-segmin', '--segmin', type=float, default=2.0, help="Minimum window size in seconds. Defaults to 0.3.")
    parser.add_argument('-segmax', '--segmax', type=float, default=2.0, help="Maximum window size in seconds. Defaults to 1.0.")
    parser.add_argument('-seghopmin', '--seghopmin_perc', type=float, default=0.95, help="Minimum hop percentage. Defaults to 0.8.")
    parser.add_argument('-seghopmax', '--seghopmax_perc', type=float, default=0.95, help="Maximum hop percentage. Defaults to 0.9.")
    parser.add_argument('-framemin', '--framemin', type=float, default=0.00, help="Minimum frame size in seconds. Defaults to 0.6.")
    parser.add_argument('-framehopmin', '--framehopmin_perc', type=float, default=0.95, help="Maximum hop percentage. Defaults to 80%.")
    parser.add_argument('-mixframe', '--mixframe', type=str2bool, default=False, help="Enable mixing windows. Defaults to True.")
    parser.add_argument('-reverse', '--reverse', type=str, default="backward", choices=["backward", "forward", "random"], help="Reverse direction. Defaults to 'backward'.")
    parser.add_argument('-method', '--method', type=str, default="random_splicing_reverse", choices=["random_splicing_reverse", "mfcc"], help="Anonymization method. Defaults to 'reverse'.")
    parser.add_argument('-sep', '--do_sep', type=str2bool, default=True, help="Enable source separation. Use True or False.")
    parser.add_argument('-sepmethod', '--sep_method', type=str, default="music", choices=["music", "env"], help="Source separation method. Defaults to 'music'.")
    args = parser.parse_args()

    anonymize_audio(args)
