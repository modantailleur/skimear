# Audio Skim Generation for Environmental Audio Recordings

This is the code for our paper "Audio Skim Generation for Environmental Audio Recordings". 
Access our GitHub project page with audio examples [here](https://modantailleur.github.io/paperAudioSummary/).

## Setup

First, install requirements.txt using the following command in a new Python 3.9.19. environment:

```
pip install -r requirements.txt
```

The experiment plan is developped with [doce](https://doce.readthedocs.io/en/latest/). 
No need to download doce as a doce folder is already provided in the repository.

## Generate audio skims from your own audio files

You need to have a folder ready with your .wav files. The .wav file need to be named:
YourDatasetName_date_time.wav
With "date" in format "20240223" for the 23th of February of 2024 and "time" in format "185424" for "18hours:54min:24seconds"
Example: ModanRecording_20240223_185424.wav
Make sure that you don't have overlapping periods of time between your audio files names. The entirety of the folder will be considered
as the same audio environment. If you want to create summaries for different audio environments, create different audio folders.

You can enforce speech content privacy using method from [1] by running the following code (with /path/to/rawaudiodataset your input directory that contains all your audio files, and /path/to/audiodataset your output directory):

```
anonymize_audio_folder.py -i /path/to/rawaudiodataset -o /path/to/audiodataset
```

Once it is ready, to run the summary calculations on your customized full-length audio, use the following code.

To format the audio into a h5 file in 8s batches (audiodataset being a folder containing all the audio .wav files):
```
python get_audio.py /path/to/audiodataset /path/to/audio.h5
```

To generate the embeddings:
```
python get_embeddings.py /path/to/audio.h5 /path/to/embeddings.h5
```

To generate the clusters (K-means for every 15min intervals):
```
python get_clusters.py /path/to/embeddings.h5 /path/to/clusters.csv --period 15
```

To generate the summaries (with scen=0.5):
```
python get_summary.py /path/to/audio.h5 /path/to/embeddings.h5 /path/to/clusters.csv /path/to/summary.wav 0.5 
```

## Generate audio skims from your own fast third-octave dataset

You need to have a folder ready with your .h5 file that contains the data. The .wav file need to be named:
YourDatasetName_date_time.wav
With "date" in format "20240223" for the 23th of February of 2024 and "time" in format "185424" for "18hours:54min:24seconds"
Example: ModanRecording_20240223_185424.wav
Make sure that you don't have overlapping periods of time between your audio files names.

Once it is ready, to run the summary calculations on your customized dataset, use the following code.

To reformat the third octaves in 8s batches:

```
python get_thirdo.py /path/to/thirdo_unformatted.h5 /path/to/thirdo.h5
```

To generate the embeddings:

```
python get_thirdo_embeddings.py /path/to/thirdo.h5 /path/to/embeddings.h5
```

To generate the clusters (K-means for every 15min intervals):

```
python get_clusters.py /path/to/embeddings.h5 /path/to/clusters.csv --period 15
```

To generate the summaries (with scen=0.5):
```
python get_thirdo_summary.py /path/to/embeddings.h5 /path/to/clusters.csv /path/to/summary.csv 0.5 
```


## Generate accompanying video for your audio skim

Generate a clock for your audio skim in "./summaries_audiovisual/video/" using:

```
python3 ./summaries_audiovisual/create_audiovisual_summary_clock.py -i /path/to/input/audio.wav
```

Generate a dynamic audio-visual summary in "./video/" using:

```
python3 ./audiovisual_summary/create_audiovisual_summary.py -i /path/to/input/audio.wav
```

## References

[1] Tailleur, Modan, Mathieu Lagrange, Pierre Aumond, and Vincent Tourre (2025), Enforcing Speech Content Privacy in Environmental Sound Recordings using Segment-wise Waveform Reversal, doi: [10.48550/arXiv.2507.08412](10.48550/arXiv.2507.08412).