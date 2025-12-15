"""
Code originally by Oliver Pauly

Based on an idea by Klaus Scherer

K. R. Scherer, “Randomized splicing: A note on a simple technique for masking speech content” 
Journal of Experimental Research in Personality, vol. 5, pp. 155–159, 1971.

Evaluated in:
F. Burkhardt, Anna Derington, Matthias Kahlau, Klaus Scherer, Florian Eyben and Björn Schuller: Masking Speech Contents by Random Splicing: is Emotional Expression Preserved?, Proc. ICASSP, 2023

Modified by Modan Tailleur to allow cross-fading, and activation/deactivation of splicing. Original code available at:
https://github.com/felixbur/nkululeko/blob/main/nkululeko/augmenting/randomsplicing.py

"""

import librosa
import numpy as np


def random_splicing(
    signal,
    sr,
    p_reverse=0.0,
    top_db=12,
    overlap=0.0,
    frame_length=0.0, # Minimum frame length for silence detection
    mixframe=True  # Whether to mix windows or not
):
    """
    Randomly splice the signal, re-arrange, and apply cross-fading between segments.

    p_reverse: Probability of some samples to be in reverse order.
    top_db: Top db level for silence to be recognized.
    overlap: Percentage of overlap between segments for cross-fading.
    """
    max_signal = np.max(abs(signal))
    signal /= max_signal

    hop_length = frame_length if frame_length != 0.0 else int(sr*512/44100)
    frame_length = 4*frame_length if frame_length != 0.0 else int(sr*2048/44100)
    overlap_length = int(frame_length * overlap)

    indices = split_wav_naive(signal, sr, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    if mixframe:
        np.random.shuffle(indices)

    wav_spliced = remix_random_reverse(signal, indices, p_reverse=p_reverse, overlap_length=overlap_length)

    wav_spliced *= max_signal
    return wav_spliced


def split_wav_naive(wav, sr, top_db=12, frame_length=2048, hop_length=512):
    indices = librosa.effects.split(wav, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    indices = np.array(indices)
    # (re)add the silence-segments
    indices = indices.repeat(2)[1:-1].reshape((-1, 2))
    # add first segment
    indices = np.vstack(((0, indices[0][0]), indices))
    # add last segment
    indices = np.vstack((indices, [indices[-1][-1], wav.shape[0]]))

    return indices


def apply_crossfade(wav1, wav2, overlap_samples):
    """
    Applies a crossfade between two audio segments.
    """
    if overlap_samples == 0 or len(wav1) == 0 or len(wav2) == 0:
        return np.concatenate([wav1, wav2])
    
    if len(wav1) < overlap_samples or len(wav2) < overlap_samples:
        # If segments are too short for crossfading, concatenate directly
        return np.concatenate([wav1, wav2])
    
    fade_out = np.linspace(1, 0, overlap_samples)
    fade_in = np.linspace(0, 1, overlap_samples)
    
    wav1[-overlap_samples:] *= fade_out
    wav2[:overlap_samples] *= fade_in
    
    return np.concatenate([wav1, wav2])

def remix_random_reverse(wav, indices, p_reverse=0, overlap_length=512):
    wav_remix = []
    # overlap_length = int(overlap_length * np.min([end - start for start, end in indices if end - start > 0]))

    for i, seg in enumerate(indices):
        start = seg[0]
        end = seg[1]
        wav_seg = wav[start:end]

        if np.random.rand(1)[0] <= p_reverse:
            wav_seg = wav_seg[::-1]

        if len(wav_remix) == 0:
            wav_remix.append(wav_seg)
        else:
            # Apply cross-fade with the previous segment if it's not empty
            if len(wav_remix[-1]) > 0:
                wav_remix[-1] = apply_crossfade(wav_remix[-1], wav_seg, overlap_length)
            else:
                wav_remix.append(wav_seg)

    # Concatenate all segments
    wav_remix = np.hstack(wav_remix)

    return wav_remix