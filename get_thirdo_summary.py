import pandas as pd
import os
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write
import random
import argparse
from utils.util import blur_audio
import itertools
from scipy.interpolate import interp1d
from pydub import AudioSegment
from utils.summary_utils import batch_embeddings_generator, get_embedding_from_row, get_dummy_embedding_from_row,\
    get_audio_from_row, get_datetime_from_row, get_embedding_len, get_audio_len, logits_generator,\
    embeddings_generator, audio_generator, datetime_generator
import h5py
import torch
from transcoder.transcoders import ThirdOctaveToMelTranscoderDiffusion
import librosa

# Define a function to select the top N rows with the highest cluster_count
def select_top_n_per_period(df, n):
    return df.sort_values(by='cluster_count', ascending=False).head(n)

def randsel_summary(df, n_frames, embeddings_path, fid=0, seed=0):
    """"
    fid is a level between 0 and 1
    """
    scen = 1 - fid
    random.seed(seed)

    #Group by period and count the number of rows in each group
    period_counts = df.groupby('period').size()

    # Find the maximum number of rows in any period
    max_rows_per_period = period_counts.min()
    selected_rows_per_period = int((max_rows_per_period - 1) * scen + 1)

    df = df.groupby('period').apply(select_top_n_per_period, n=selected_rows_per_period).reset_index(drop=True)

    # Get unique periods
    unique_periods = df['period'].unique()
    
    # Randomly select n_periods different periods
    selected_periods = random.sample(list(unique_periods), n_frames)
    
    selected_elements = []
    
    for period in selected_periods:
        # Filter the dataframe for the current period
        period_df = df[df['period'] == period]
        
        # Randomly select one element within the current period
        selected_element = period_df.sample(n=1, random_state=seed)
        
        # Append the selected element to the list
        selected_elements.append(selected_element)
    
    # Concatenate the selected elements into a new dataframe
    final_df = pd.concat(selected_elements).reset_index(drop=True)

    return final_df

# See TASLP paper for more infor about beta and theta
def beta_x(x, s, k=6, a=20):
    if k == 0:
        output = 1
    else:
        output = a**(-((1-s)**k)*x)
    return output

def theta_x(x, s, k=6, a=20):
    if k == 0:
        output = 1
    else:
        output = a**(-((1-s)**k)*x)
    return output

class Optimal(Exception): pass
class GreedyAlgorithm:
    def __init__(self, combinations, embeddings, n_frames, ranks, betas, dist, scen, k_beta=6, k_theta=6, a_beta=20, a_theta=20):
        self.combinations = combinations
        self.selections = np.full_like(combinations, False, dtype=bool)
        self.embeddings = embeddings
        self.n_frames = n_frames
        self.ranks = ranks
        self.betas = betas

        self.dist = dist
        self.scen = scen
        self.k_beta = k_beta
        self.k_theta = k_theta
        self.a_beta = a_beta
        self.a_theta = a_theta

        #put a first element in the selection
        betas_3D = np.repeat(betas[:, :, np.newaxis], embeddings.shape[-1], axis=2)
        weighted_mean = np.average(embeddings, weights=betas_3D, axis=(0, 1))

        init_differences = np.array([[betas[i,j]*self.dist(weighted_mean, embeddings[i, j]) 
                                    for j in range(embeddings.shape[1])] for i in range(embeddings.shape[0])])
        init_differences[np.isnan(init_differences)] = - np.inf
        max_index = np.unravel_index(np.argmax(init_differences), init_differences.shape)
        self.selections[max_index] = True
        
    def _get_distance_matrix(self, selections, mitigate_border_effects=True):
        indices = np.argwhere(selections)
        indices = indices[indices[:, 1].argsort()]
        cur_row_indices = indices[:, 0]
        cur_delta_indices = indices[:, 1]
        cur_embeddings = self.embeddings[cur_row_indices, cur_delta_indices]

        distance_matrix = np.zeros((cur_embeddings.shape[0], cur_embeddings.shape[0]))

        if (mitigate_border_effects) & (cur_embeddings.shape[0] > 2):
            # This version is much simpler than the older version. Here we just devide the thetas by the sum of thetas,
            # this way it gives more weight to the elements that are in the border, as they have less close neighboors.
            for i in range(cur_embeddings.shape[0]):
                sum_thetas = 0
                for j in range(cur_embeddings.shape[0]):
                    if i == j:
                        continue
                    sum_thetas += theta_x(abs(i-j), self.scen, self.k_theta, self.a_theta)

                for j in range(cur_embeddings.shape[0]):
                    if i == j:
                        continue
                    thetas = theta_x(abs(i-j), self.scen, self.k_theta, self.a_theta) / sum_thetas
                    distance = self.dist(cur_embeddings[i], cur_embeddings[j])
                    metric = distance * thetas * self.betas[cur_row_indices[i], cur_delta_indices[i]] * self.betas[cur_row_indices[j], cur_delta_indices[j]]
                    distance_matrix[i, j] = metric

        else:
            # if we don't take into account border effects
            for i in range(cur_embeddings.shape[0]):
                for j in range(i + 1, cur_embeddings.shape[0]):
                    thetas = theta_x(abs(i-j), self.scen, self.k_theta, self.a_theta)
                    distance = self.dist(cur_embeddings[i], cur_embeddings[j])
                    metric = distance * thetas * self.betas[cur_row_indices[i], cur_delta_indices[i]] * self.betas[cur_row_indices[j], cur_delta_indices[j]]
                    distance_matrix[i, j] = metric
                    distance_matrix[j, i] = metric


        return(distance_matrix)

    def compute(self, n_iter=5, greedy_batch=1, verbose=True):
        # Executes a first loop on the greedy algorithm. The resulting combinations 
        # matrix is already optimized, but we can still improve it with multiple iterations
        selections, sum_differences_bef = self.loop(self.selections, greedy_batch=greedy_batch)

        new_selections = selections.copy()
        sum_differences_aft = sum_differences_bef

        # Execute multiple iterations to find the optimal solution
        try:
            for i in range(n_iter):
                if verbose:
                    print(f"Greedy Optim Iter: {i} / {n_iter}", end='\r')
                c = 0
                c_l = [c + k for k in range(greedy_batch)]
                while (sum_differences_aft <= sum_differences_bef):
                    temp_selections = new_selections.copy()
                    temp_selections = self.remove_element(new_selections, last_n=c_l)
                    temp_selections, sum_differences_aft = self.loop(temp_selections, greedy_batch=greedy_batch, verbose=False)
                    c += 1
                    if c > self.n_frames:
                        raise Optimal
                sum_differences_bef = sum_differences_aft
                new_selections = temp_selections.copy()
        except Optimal:
            if verbose:
                print('\n')
                print('FOUND OPTIMAL SOLUTION')
        if verbose:
            print('\n')
        selections = new_selections

        return(selections)

    def loop(self, selections, greedy_batch=1, verbose=True):

        new_selections = selections.copy()
        #GREEDY ALGORITHM
        n_per = 0
        while n_per < self.n_frames:
            if verbose:
                print(f"Greedy Loop: {n_per} / {self.n_frames}", end='\r')

            cur_greedy_batch = (self.n_frames - n_per) if (self.n_frames - n_per) < greedy_batch else greedy_batch

            # Find the indices of columns where there is only one non -1 element
            cur_remaining_indices = np.where(np.sum(new_selections != False, axis=0) != 1)[0]

            # Generate all combinations of n column indices
            column_combinations = list(itertools.combinations(cur_remaining_indices, cur_greedy_batch))

            # Generate all combinations of n row indices
            row_combinations = list(itertools.product(range(selections.shape[0]), repeat=cur_greedy_batch))

            all_combinations = []
            # Loop through each combination of column indices
            for col_indices in column_combinations:
                # Loop through each combination of row indices
                for row_indices in row_combinations:
                    # Get the elements from the selected rows and columns
                    for _, (col_idx, row_idx) in enumerate(zip(col_indices, row_indices)):
                        if self.combinations[row_idx, col_idx] == -1:
                            break
                    else:
                        all_combinations.append([(row_indices[i], col_indices[i]) for i in range(cur_greedy_batch)])

            best_selections = None
            best_distance = None
            for all_comb in all_combinations:
                temp_cur_selections = new_selections.copy()
                temp_cur_selections[tuple(zip(*all_comb))] = True

                if verbose:
                    print(f"Greedy Loop: {n_per} / {self.n_frames}", end='\r')

                distance = np.sum(self._get_distance_matrix(temp_cur_selections))
                if best_distance is None:
                    best_distance = distance
                    best_selections = temp_cur_selections.copy()
                else:
                    if distance > best_distance:
                        best_distance = distance
                        best_selections = temp_cur_selections.copy()

            new_selections = best_selections
            n_per = np.sum(np.sum(new_selections != False, axis=0) == 1)
        sum_differences = np.sum(self._get_distance_matrix(new_selections))
        if verbose:
            print('\n')

        return(new_selections, sum_differences)

    def remove_element(self, selections, last_n=0):
        """
        Remove the elements that is in rank last_n in term of least contribution to the sum of distances.
        last_n can be either a list or a single integer.
        """
        new_selections = selections.copy()
        # Find the indices of columns where there is only one non -1 element
        indices = np.argwhere(new_selections)
        cur_delta_indices = indices[:, 1]

        distance_matrix = self._get_distance_matrix(selections)

        summed_distance_matrix = np.sum(distance_matrix, axis=0)

        lowest_indices = np.argsort(summed_distance_matrix)[last_n]

        new_selections[:, cur_delta_indices[lowest_indices]] = False

        return new_selections

def greedy_summary(df, n_frames, embeddings_path, scen=0, limit_count=5, n_iter=5, greedy_batch=1, 
                   k_beta=6, k_theta=6, a_beta=20, a_theta=20):
    """"
    This function creates a complex greedy summary. The idea is to build a summary 
    by selecting the most diverse elements. So we want to find a combination 
    that maximizes the sum of distances between each elements. This sum is 
    weighted by the rank of each element within a time period, and weighted by 
    the relative position between time periods.

    To find this combination, we first put the element that is closer to the 
    weighted avg of all the embeddings of the matrix of elements. Then we iteratively
    add the element that maximizes the sum of distances of the current selection.
    We stop when we have n_frames elements in the selection.

    We can then optionnaly remove the elements from the selection that contributes 
    the less to the sum of distances, and reiterate the process at n_frames-1. 

    

    fid is a level between 0 and 1
    use limit_count = -1 if you don't want any limit in the number of elements 
    per cluster
    """

    # EXAMPLE OF PRINT FOR DEBUGGING
    # print('CURRENT COMBINATIONS')
    # # print(np.matrix(combinations))
    # for row in temp_cur_combination:
    #     print(" ".join(f"{val:05}" if val != -1 else "    -" for val in row))
    # print(cur_remaining_indices)

    dist = lambda p, q: np.linalg.norm(p - q)
    df_to_warp = df.copy()

    # Here the column "rank" becomes the scarcity score described in the TASLP paper.
    df_to_warp['max_cluster_count'] = df_to_warp.groupby('period')['cluster_count'].transform('max')
    df_to_warp['rank'] = (df_to_warp['max_cluster_count'] / df_to_warp['cluster_count']) - 1
    df_to_warp.drop(columns=['max_cluster_count'], inplace=True)

    ############
    ###########

    df_to_warp = df_to_warp.sort_values(by='clusters_datetimes')

    #create a list of periods
    periods = df_to_warp.groupby('period')
    # Get the number of unique periods
    num_periods = len(periods)
    # Find the maximum number of elements in any period
    max_elements_per_period = periods.size().max()

    # Create an empty numpy array with NaN as placeholders
    combinations = np.full((max_elements_per_period, num_periods), -1)
    cluster_counts = np.full((max_elements_per_period, num_periods), -1)
    ranks = np.full((max_elements_per_period, num_periods), -1.)

    dfs_sorted_by_period = []
    for i, (period, group) in enumerate(periods):
        sorted_group = group.sort_values(by='cluster_count', ascending=False).reset_index(drop=True)
        dfs_sorted_by_period.append(sorted_group)
        sorted_group['cluster_count'] = sorted_group['cluster_count'].apply(lambda x: -1 if x < limit_count else x)
        sorted_group['row'] = sorted_group.apply(lambda row: -1 if row['cluster_count'] == -1 else row['row'], axis=1)
        sorted_group['rank'] = sorted_group.apply(lambda row: -1 if row['cluster_count'] == -1 else row['rank'], axis=1)
        row_values = sorted_group['row'].values
        combinations[:len(row_values), i] = row_values

        cluster_count_values = sorted_group['cluster_count'].values
        cluster_counts[:len(cluster_count_values), i] = cluster_count_values

        rank_values = sorted_group['rank'].values
        ranks[:len(rank_values), i] = rank_values

    #replace limit_count by -1 if you don't want any limit
    embeddings = np.array([[get_dummy_embedding_from_row(embeddings_path) if combinations[i, j] == -1 else get_embedding_from_row(embeddings_path, combinations[i, j]) 
                            for j in range(combinations.shape[1])] 
                        for i in range(combinations.shape[0])])

    betas = np.array([[beta_x(ranks[i,j], scen, k=k_beta, a=a_beta) if ranks[i,j]!=-1. else -1. for j in range(ranks.shape[1])] for i in range(ranks.shape[0])])

    #GREEDY ALGORITHM
    greedy_alg = GreedyAlgorithm(combinations, embeddings, n_frames, ranks, betas, dist, scen,
                                 k_beta=k_beta, k_theta=k_theta, a_beta=a_beta, a_theta=a_theta)
    selections = greedy_alg.compute(n_iter=n_iter, greedy_batch=greedy_batch)

    unique_indices = []

    unique_indices = np.argwhere(selections)
    selected_rows = []
    # Iterate through unique_indices
    for i, j in unique_indices:
        # Select the corresponding DataFrame from dfs_sorted_by_period
        df = dfs_sorted_by_period[j]
        
        # Extract the row corresponding to index i
        selected_row = df.iloc[i]
        
        # Append the selected row to the list
        selected_rows.append(selected_row)

    # Concatenate all selected rows into a single DataFrame
    final_df = pd.concat(selected_rows, axis=1).T
    final_df = final_df.sort_values(by='clusters_datetimes')
    final_df = final_df.reset_index(drop=True)
    
    return final_df

# Function to apply fade-in and fade-out
def apply_fades(audio, fade_in_samples, fade_out_samples):
    np_audio = audio

    if fade_in_samples != 0:
        fade_in = np.sqrt(np.sin(np.linspace(0, np.pi / 2, fade_in_samples))**2)
        np_audio[:fade_in_samples] *= fade_in

    if fade_out_samples != 0:
        fade_out = np.sqrt(np.sin(np.linspace(np.pi / 2, 0, fade_out_samples))**2)
        np_audio[-fade_out_samples:] *= fade_out
    
    return np_audio

def concatenate_audios_with_overlap(audios, sr, overlap=3, initial_cut=0):
    """
    overlap: duration of overlap in seconds
    initial_cut: duration of initial cut in seconds
    """
    num_audios = len(audios)
    audio_length = len(audios[0])

    # Duration in samples
    overlap_samples = overlap * sr
    initial_cut_samples = initial_cut * sr

    # Prepare the final concatenated audio list
    concatenated_length = (audio_length - initial_cut_samples) + (num_audios - 1) * (audio_length - overlap_samples)
    concatenated_audio = np.zeros(concatenated_length)
    current_position = 0

    for i in range(0, num_audios):
        next_audio_segment = audios[i]
        
        if i == 0:
            # Apply fade in
            next_audio_segment = next_audio_segment[initial_cut_samples:]
            next_audio_segment = apply_fades(next_audio_segment, 0, overlap_samples)
        elif i == num_audios - 1:
            # Apply fade out
            next_audio_segment = apply_fades(next_audio_segment, overlap_samples, 0)
        else:
            # Apply fade in and fade out
            next_audio_segment = apply_fades(next_audio_segment, overlap_samples, overlap_samples)
        
        concatenated_audio[current_position:current_position + len(next_audio_segment)] += next_audio_segment
        current_position += (len(next_audio_segment) - overlap_samples)

    return concatenated_audio

def get_thirdo_greedy_summary(clusters_path, thirdo_path, embeddings_path, summary_path, sr, scen, start_datetime, end_datetime, block_length=8, n_frames=12, gain_db=0, 
                       blur=False, limit_count=5, n_iter=5, k_beta=6, k_theta=6, a_beta=20, a_theta=20, greedy_batch=1, verbose=True, audio_type="wav"):

    # Define the CSV file path
    csv_file_path = clusters_path

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    df['clusters_datetimes'] = pd.to_datetime(df['clusters_datetimes'])

    embeddings_path = embeddings_path

    if start_datetime is not None:
        df = df[df['clusters_datetimes'] >= start_datetime]
    if end_datetime is not None:
        df = df[df['clusters_datetimes'] <= end_datetime]

    fid = 1 - scen
    df_to_warp = greedy_summary(df, n_frames, embeddings_path, scen=scen, limit_count=limit_count, n_iter=n_iter, greedy_batch=greedy_batch, 
                                k_beta=k_beta, k_theta=k_theta, a_beta=a_beta, a_theta=a_theta)

    thirdos = np.array([get_thirdo_from_row(thirdo_path, df_to_warp.loc[k, "row"]) for k in range(len(df_to_warp))])

    if verbose:
        print(df_to_warp)
        
    ################
    ## CONVERT THIRDOS TO WAV

    #transcoder setup
    MODEL_PATH = "./transcoder/ckpt/"
    transcoder_name = 'tau_diff_steps=1000+epoch=70+learning_rate=-4_model.pt'
    force_cpu = False
    #manage gpu
    useCuda = torch.cuda.is_available() and not force_cpu

    if useCuda:
        print('Using CUDA.')
        dtype = torch.cuda.FloatTensor
        #MT: add
        device = torch.device("cuda:0")
    else:
        print('No CUDA available.')
        dtype = torch.FloatTensor
        #MT: add
        device = torch.device("cpu")

    transcoder = ThirdOctaveToMelTranscoderDiffusion(transcoder_name, MODEL_PATH, device=device, dtype=dtype)

    thirdos_torch = torch.from_numpy(thirdos).float().to(device)
    thirdos_torch = thirdos_torch[:, :, 8:-1]  # Remove the first 7 and last 1 columns
    audios = []
    for thirdos_torch_i in thirdos_torch:
        thirdopinvmel_chunks = transcoder.load_thirdo_chunks(thirdos_torch_i)
        diffusionmel_chunks = transcoder.mels_to_mels(thirdopinvmel_chunks, torch_output=True, batch_size=8)
        diffusionmel = transcoder.gomin_mel_chunker.concat_spec_with_hop(diffusionmel_chunks)
        diffusionmel = np.expand_dims(diffusionmel, axis=0)
        wav_diffusionmel = transcoder.mels_to_audio(diffusionmel, torch_output=True, batch_size=1)[0][0]
        # Resample from 24kHz to 32kHz
        wav_diffusionmel_32k = librosa.resample(wav_diffusionmel, orig_sr=24000, target_sr=sr)
        # Pad or trim wav_diffusionmel_32k to block_length seconds (block_length * 32000 samples)
        target_length = block_length * sr
        if wav_diffusionmel_32k.shape[0] < target_length:
            wav_diffusionmel_32k = np.pad(wav_diffusionmel_32k, (0, target_length - wav_diffusionmel_32k.shape[0]), mode='constant')
        else:
            wav_diffusionmel_32k = wav_diffusionmel_32k[:target_length]
        audios.append(wav_diffusionmel_32k)

    # Concatenate the audio files with overlap
    audio_files = concatenate_audios_with_overlap(audios, sr)

    # Apply gain to the audio files
    gain_linear = 10 ** (gain_db / 20)
    audio_files = audio_files * gain_linear

    # Blur the audio file voice
    if blur:
        audio_files = blur_audio(audio_files, sr)

    # Add first row with "start_datetime"
    first_row = pd.DataFrame({'clusters_datetimes': [start_datetime]})
    df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)

    # Add last row with "end_datetime"
    last_row = pd.DataFrame({'clusters_datetimes': [end_datetime]})
    df_to_warp = pd.concat([df_to_warp, last_row], ignore_index=True)

    df_to_warp.to_csv(os.path.splitext(summary_path)[0]+".csv", index=False)
    if audio_type == "mp3":
        # Ensure NumPy array is float32 (Pydub expects int16 for PCM)
        if audio_files.dtype != np.int16:
            audio_files = (audio_files * 32767).astype(np.int16)  # Convert from float to int16

        # Convert to bytes and create an AudioSegment with correct parameters
        audio_segment = AudioSegment(
            audio_files.tobytes(), frame_rate=sr, sample_width=2, channels=1
        )
        audio_segment.export(summary_path, format="mp3", bitrate="192k")
    else:
        write(summary_path, sr, audio_files)

def get_thirdo_from_row(embeddings_path, row_index):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        thirdo_dataset = hf["thirdo"]
        thirdo = thirdo_dataset[row_index]
        return thirdo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a greedy summary from clusters and embeddings.")
    parser.add_argument("thirdo_path", type=str, help="Path to the HDF5 file containing third-octave data.")
    parser.add_argument("embeddings_path", type=str, help="Path to the HDF5 file containing embeddings.")
    parser.add_argument("clusters_path", type=str, help="Path to the CSV file containing cluster data.")
    parser.add_argument("summary_path", type=str, help="Path to the output WAV file for the summary.")
    parser.add_argument("scen", type=float, help="Level of scenism (between 0 and 1).")
    parser.add_argument("--start_datetime", type=str, default=None, help="Start datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--end_datetime", type=str, default=None, help="End datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--sr", type=int, nargs="?", default=32000, help="Output sample rate, set by default to 32000.")
    parser.add_argument("--block_length", type=int, nargs="?", default=8, help="Length of a block of audio (in seconds).")
    parser.add_argument("--n_frames", type=int, default=12, help="Number of frames for the summary. 12 frames corresponds to a 1min summary if \
                         block_length=8 when the embeddings were generated, and 24 frames corresponds to a 2min summary.")
    parser.add_argument("--gain_db", type=float, default=0, help="Gain in dB for the output audio.")
    parser.add_argument("--limit_count", type=float, default=1, help="Minimum number of elements required to keep a cluster. \
                        The higher the limit_count, the less 'erratic' clusters will be kept.")
    parser.add_argument("--n_iter", type=int, default=0, help="Number of iterations for the greedy summary algorithm. Default is 0.")
    parser.add_argument("--greedy_batch", type=int, default=1, help="Batch size for the greedy algorithm. Default is 1.")
    parser.add_argument("--audio_type", type=str, default="wav", help="File format for the summary.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Enable verbose output. Default is True.")

    args = parser.parse_args()

    start_datetime = datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S") if args.start_datetime else None
    end_datetime = datetime.strptime(args.end_datetime, "%Y-%m-%d %H:%M:%S") if args.end_datetime else None

    get_thirdo_greedy_summary(args.clusters_path, args.thirdo_path, args.embeddings_path, args.summary_path, args.sr, args.scen, 
                       start_datetime, end_datetime, args.block_length, args.n_frames, args.gain_db, False, args.limit_count, args.n_iter, 6, 3, 5, 10, args.greedy_batch, args.verbose, args.audio_type)