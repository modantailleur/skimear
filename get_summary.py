import pandas as pd
import os
from datetime import datetime
import numpy as np
from scipy.io.wavfile import write
import random
import argparse
import itertools
from pydub import AudioSegment
from utils.summary_utils import get_embedding_from_row, get_dummy_embedding_from_row,\
    get_audio_from_row, get_datetime_from_row, logits_generator,\
    embeddings_generator, datetime_generator

# Define a function to select the top N rows with the highest cluster_count
def select_top_n_per_period(df, n):
    return df.sort_values(by='cluster_count', ascending=False).head(n)

# See paper for more info about beta and theta
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

    """
    Generates a summary DataFrame by selecting representative rows from the input DataFrame using a greedy algorithm.
    This function applies a greedy selection strategy to choose a subset of rows from the input DataFrame `df` 
    that best summarize the data across different periods and clusters. 
    It uses cluster counts, scarcity scores, and embeddings to guide the selection process, 
    ensuring faithfulness and highlightness. The selection is influenced by several hyperparameters 
    controlling the behavior of the greedy algorithm.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing data to be summarized. Must include columns such as 'period', 'cluster_count', 'row', and 'clusters_datetimes'.
        n_frames (int): Number of frames (rows) to select for the summary.
        embeddings_path (str): Path to the embeddings used for similarity calculations.
        scen (int, optional): Scenario index to control algorithm behavior. Default is 0.
        limit_count (int, optional): Minimum cluster count required for a row to be considered. Rows with lower counts are ignored. Default is 5.
        n_iter (int, optional): Number of iterations for the greedy algorithm. Default is 5.
        greedy_batch (int, optional): Batch size for greedy selection in each iteration. Default is 1.
        k_beta (int, optional): Hyperparameter for beta weighting in the algorithm. Default is 6.
        k_theta (int, optional): Hyperparameter for theta weighting in the algorithm. Default is 6.
        a_beta (int, optional): Scaling factor for beta weighting. Default is 20.
        a_theta (int, optional): Scaling factor for theta weighting. Default is 20.
    Returns:
        pd.DataFrame: A DataFrame containing the selected summary rows, sorted by 'clusters_datetimes' and reset index.
    Notes:
        - The function assumes the existence of helper functions such as `get_embedding_from_row`, `get_dummy_embedding_from_row`, `beta_x`, and a `GreedyAlgorithm` class.
        - The input DataFrame must be preprocessed to include necessary columns for clustering and period identification.
        - The summary aims to balance scarcity (rarity of clusters) and diversity (embedding-based similarity).
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
    """
    Applies fade-in and fade-out effects to an audio signal.
    This function modifies the input audio array by applying a fade-in effect to the beginning
    and a fade-out effect to the end, using smooth sinusoidal curves. The number of samples
    for each fade effect is specified by the corresponding arguments.
    Parameters:
        audio (np.ndarray): The input audio signal as a NumPy array.
        fade_in_samples (int): Number of samples over which to apply the fade-in effect.
        fade_out_samples (int): Number of samples over which to apply the fade-out effect.
    Returns:
        np.ndarray: The audio signal with fade-in and fade-out applied.
    """
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
    Concatenates a list of audio segments with specified overlap and optional initial cut.
    This function takes a list of audio arrays and concatenates them into a single audio array.
    Each audio segment is overlapped with the next by a specified duration (in seconds), and
    optional initial samples can be cut from the first segment. Fades are applied at the overlap
    regions to ensure smooth transitions between segments.
    Parameters
    ----------
    audios : list of np.ndarray
        List of 1D numpy arrays representing audio segments to concatenate.
    sr : int
        Sample rate of the audio segments (samples per second).
    overlap : int, optional
        Duration of overlap between consecutive audio segments, in seconds (default is 3).
    initial_cut : int, optional
        Duration to cut from the beginning of the first audio segment, in seconds (default is 0).
    Returns
    -------
    np.ndarray
        The concatenated audio array with overlaps and fades applied.
    Notes
    -----
    - The function assumes all audio segments have the same length and sample rate.
    - Fades are applied using the `apply_fades` function.
    - Overlapping regions are summed to create smooth transitions.
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

def get_greedy_summary(clusters_path, audio_path, embeddings_path, summary_path, sr, scen, start_datetime, end_datetime, n_frames=12,
                       limit_count=5, n_iter=5, k_beta=6, k_theta=6, a_beta=20, a_theta=20, greedy_batch=1, verbose=True, audio_type="wav"):

    """
    Generates a summary audio file by selecting representative audio segments from clustered data using a greedy algorithm,
    concatenates them with overlap, and saves the resulting audio and metadata. The summary generated is more or less faithful
    or scenic depending on the input "scen". It's the main method of our paper.

    Parameters
    ----------
    clusters_path : str
        Path to the CSV file containing cluster information and datetimes.
    audio_path : str
        Path to the audio file or directory containing the audio data.
    embeddings_path : str
        Path to the embeddings used for segment selection.
    summary_path : str
        Path where the generated summary audio file will be saved.
    sr : int
        Sample rate for audio processing.
    scen : int
        Scenario identifier used to determine selection logic.
    start_datetime : Any
        Start of the datetime range for filtering clusters.
    end_datetime : Any
        End of the datetime range for filtering clusters.
    n_frames : int, optional
        Number of audio segments to select for the summary (default is 12).
    limit_count : int, optional
        Minimum cluster count required for a row to be considered (default is 5).
    n_iter : int, optional
        Number of iterations for the greedy algorithm (default is 5).
    k_beta : int, optional
        Beta parameter for the greedy algorithm (default is 6).
    k_theta : int, optional
        Theta parameter for the greedy algorithm (default is 6).
    a_beta : int, optional
        Beta scaling factor for the greedy algorithm (default is 20).
    a_theta : int, optional
        Theta scaling factor for the greedy algorithm (default is 20).
    greedy_batch : int, optional
        Batch size for greedy selection (default is 1).
    verbose : bool, optional
        If True, prints summary information to the console (default is True).
    audio_type : str, optional
        Output audio format, either "wav" or "mp3" (default is "wav").

    Returns
    -------
    None
        The function saves the concatenated audio summary to `summary_path` and a CSV file with metadata.

    Notes
    -----
    - The function selects audio segments using a greedy algorithm to maximize representativeness and diversity.
    - Metadata about the selected segments (periods and row indices) is saved as a CSV file alongside the audio summary.
    - Supports saving the summary in either WAV or MP3 format.
    """

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

    audios = np.array([get_audio_from_row(audio_path, df_to_warp.loc[k, "row"]) for k in range(len(df_to_warp))])

    # Concatenate the audio files with overlap
    audio_files = concatenate_audios_with_overlap(audios, sr)

    if verbose:
        print(df_to_warp)

    # Add first row with "start_datetime"
    if start_datetime is not None:
        first_row = pd.DataFrame({'period': [start_datetime]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)
    else:
        first_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, 0)]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)

    # Add last row with "end_datetime"
    if start_datetime is not None:
        last_row = pd.DataFrame({'period': [end_datetime]})
        df_to_warp = pd.concat([df_to_warp, last_row], ignore_index=True)
    else:
        last_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, -1)]})
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

def get_randsel_summary(clusters_path, audio_path, embeddings_path, summary_path, sr, scen, start_datetime, end_datetime, 
                        n_frames=12, seed=0, audio_type="wav"):

    """
    Generates a summary audio file by randomly selecting audio segments from clustered data within a specified time range,
    concatenates them, and saves the resulting audio and metadata.

    Parameters
    ----------
    clusters_path : str
        Path to the CSV file containing cluster information and datetimes.
    audio_path : str
        Path to the audio file or directory containing the audio data.
    embeddings_path : str
        Path to the embeddings used for segment selection.
    summary_path : str
        Path where the generated summary audio file will be saved.
    sr : int
        Sample rate for audio processing.
    scen : int
        Scenario identifier used to determine selection logic.
    start_datetime : Any
        Start of the datetime range for filtering clusters.
    end_datetime : Any
        End of the datetime range for filtering clusters.
    n_frames : int, optional
        Number of audio segments to select for the summary (default is 12).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    audio_type : str, optional
        Output audio format, either "wav" or "mp3" (default is "wav").

    Returns
    -------
    None
        The function saves the concatenated audio summary to `summary_path` and a CSV file with metadata.

    Notes
    -----
    - The function randomly samples audio segments from clustered data within the specified datetime range.
    - Metadata about the selected segments (periods and row indices) is saved as a CSV file alongside the audio summary.
    - Supports saving the summary in either WAV or MP3 format.
    """

    # Define the CSV file path
    csv_file_path = clusters_path

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    df['clusters_datetimes'] = pd.to_datetime(df['clusters_datetimes'])

    embeddings_path = embeddings_path

    df = df[(df['clusters_datetimes'] >= start_datetime) & (df['clusters_datetimes'] <= end_datetime)]

    fid = 1 - scen
    df_to_warp = randsel_summary(df, n_frames, embeddings_path, fid=fid, seed=seed)

    audios = np.array([get_audio_from_row(audio_path, df_to_warp.loc[k, "row"]) for k in range(len(df_to_warp))])

    audio_files = concatenate_audios_with_overlap(audios, sr)

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

def get_random_summary(embeddings_path, audio_path, summary_path, sr, start_datetime, 
                       end_datetime, n_frames=12, seed=0, audio_type="wav"):

    """
    Generates a random audio summary by selecting random segments from an audio dataset based on embeddings,
    concatenates them, and saves the resulting audio and corresponding metadata.

    Parameters
    ----------
    embeddings_path : str
        Path to the embeddings file or directory used to select audio segments.
    audio_path : str
        Path to the audio file or directory containing the audio data.
    summary_path : str
        Path where the generated summary audio file will be saved.
    sr : int
        Sample rate for the audio processing.
    start_datetime : Any
        Start datetime for filtering the embeddings and audio segments.
    end_datetime : Any
        End datetime for filtering the embeddings and audio segments.
    n_frames : int, optional
        Number of random audio segments to select and concatenate (default is 12).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    audio_type : str, optional
        Type of audio file to save ("wav" or "mp3", default is "wav").

    Returns
    -------
    None
        The function saves the concatenated audio summary to `summary_path` and a CSV file with metadata.

    Notes
    -----
    - The function creates a summary by randomly sampling audio segments within the specified datetime range.
    - Metadata about the selected segments (periods and row indices) is saved as a CSV file alongside the audio summary.
    - Supports saving the summary in either WAV or MP3 format.
    """

    random.seed(seed)
    full_data = np.array(list(embeddings_generator(embeddings_path, start_datetime=start_datetime, end_datetime=end_datetime)), dtype=object)
    indices = np.array([full_data[elem, 2] for elem in range(len(full_data))])
    random_indices = random.sample(list(indices), n_frames)

    audios = np.array([get_audio_from_row(audio_path, k) for k in random_indices])
    audio_files = concatenate_audios_with_overlap(audios, sr)

    sum_datetimes = [get_datetime_from_row(embeddings_path, k) for k in random_indices]

    # Create DataFrame
    df_to_warp = pd.DataFrame({
        "period": sum_datetimes,
        "row": random_indices
    })

    # Add first row with "start_datetime"
    if start_datetime is not None:
        first_row = pd.DataFrame({'period': [start_datetime]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)
    else:
        first_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, 0)]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)

    # Add last row with "end_datetime"
    if start_datetime is not None:
        last_row = pd.DataFrame({'period': [end_datetime]})
        df_to_warp = pd.concat([df_to_warp, last_row], ignore_index=True)
    else:
        last_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, -1)]})
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

def get_downsample_summary(audio_path, summary_path, sr, start_datetime, 
                       end_datetime, n_frames=12, seed=0, audio_type="wav"):
    """
    Generates a downsampled audio summary and corresponding metadata from a given audio dataset.

    This function selects `n_frames` representative audio segments from the specified time range in the audio dataset,
    concatenates them with overlap, and saves the resulting summary audio file. It also creates a CSV file containing
    the datetimes and row indices of the selected segments.

    Parameters
    ----------
    audio_path : str
        Path to the source audio dataset.
    summary_path : str
        Path where the summary audio file will be saved.
    sr : int
        Sample rate for the output audio.
    start_datetime : datetime or str
        Start datetime for the selection window.
    end_datetime : datetime or str
        End datetime for the selection window.
    n_frames : int, optional
        Number of segments to include in the summary (default is 12).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    audio_type : str, optional
        Output audio format, either "wav" or "mp3" (default is "wav").

    Returns
    -------
    None
        The function saves the summary audio and metadata CSV to disk.

    Notes
    -----
    - The function assumes the existence of helper functions such as `datetime_generator`, `get_datetime_from_row`,
        `get_audio_from_row`, and `concatenate_audios_with_overlap`.
    - The summary audio is saved in the specified format, and the metadata CSV contains the datetimes and row indices
        of the selected segments.
    """

    random.seed(seed)

    #MT: WARINING le get_audio_len ne fait pas ce que je veux qu'il fasse
    all_data = np.array([x for i, x in enumerate(datetime_generator(audio_path, start_datetime=start_datetime, end_datetime=end_datetime))])
    audio_len = len(all_data)

    # Split indices into n_frames windows
    window_size = audio_len // n_frames
    windows = [all_data[i * window_size: (i + 1) * window_size] for i in range(n_frames)]

    # Randomly select one element from each window
    full_data = np.array([random.choice(window) for window in windows])

    # Remove elements of audio until its length is n_frames
    while len(full_data) > n_frames:
        print('The audio contains more elements than n_frames. Removing the last element.')
        full_data = full_data[:-1]

    indices = np.array([x[1] for x in full_data], dtype=object)
    sum_datetimes = [get_datetime_from_row(audio_path, row_index) for row_index in indices]
    audios = np.array([get_audio_from_row(audio_path, row_index) for row_index in indices])

    audio_files = concatenate_audios_with_overlap(audios, sr)

    # Create DataFrame
    df_to_warp = pd.DataFrame({
        "datetimes": sum_datetimes,
        "row": indices
    })

    # Add first row with "start_datetime"
    if start_datetime is not None:
        first_row = pd.DataFrame({'period': [start_datetime]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)
    else:
        first_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, 0)]})
        df_to_warp = pd.concat([first_row, df_to_warp], ignore_index=True)

    # Add last row with "end_datetime"
    if start_datetime is not None:
        last_row = pd.DataFrame({'period': [end_datetime]})
        df_to_warp = pd.concat([df_to_warp, last_row], ignore_index=True)
    else:
        last_row = pd.DataFrame({'period': [get_datetime_from_row(audio_path, -1)]})
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


def get_scenic_summary(logits_path, audio_path, summary_path, sr, start_datetime, end_datetime, n_frames=12, audio_type="wav"):
    """
    Generates a scenic audio summary by selecting the most representative audio segments based on model logits. This
    scenic summary is then used for calibration of the scenism metric.

    This function processes logits and audio data to identify and extract the top `n_frames` segments with the highest
    model confidence (logits) within a specified datetime range. The selected audio segments are concatenated with overlap
    and saved as a summary audio file (WAV or MP3). Additionally, a CSV file is generated containing the datetimes and row
    indices of the selected segments.

    Parameters
    ----------
    embeddings_path : str
        Path to the file containing audio embeddings.
    audio_path : str
        Path to the audio data file.
    summary_path : str
        Output path for the generated summary audio file.
    sr : int
        Sample rate for audio processing.
    start_datetime : datetime or str
        Start datetime for selecting embeddings.
    end_datetime : datetime or str
        End datetime for selecting embeddings.
    n_frames : int, optional
        Number of summary segments (windows) to generate (default is 12).
    audio_type : str, optional
        Output audio format, either "wav" or "mp3" (default is "wav").

    Returns
    -------
    None
        The function saves the summary audio file and a CSV file with segment metadata to disk.

    Notes
    -----
    - The function expects helper functions such as `embeddings_generator`, `get_datetime_from_row`,
        `get_audio_from_row`, and `concatenate_audios_with_overlap` to be defined elsewhere.
    - If the output format is "mp3", the audio is converted to int16 and exported using Pydub.
    - The CSV file contains the datetimes and row indices of the selected summary segments.
    """




    ################
    ## CALCULATE PANN KL DIVERGENCE
    full_out = np.array(list(logits_generator(logits_path, start_datetime=start_datetime, end_datetime=end_datetime)), dtype=object)
    logits = np.array([full_out[elem, 0] for elem in range(len(full_out))])
    dates = np.array([full_out[elem, 1] for elem in range(len(full_out))])
    indices = np.array([full_out[elem, 2] for elem in range(len(full_out))])
    # indices = np.arange(len(logits))

    max_logits = np.max(logits, axis=0)
    argmax_logits = np.argmax(logits, axis=0)

    # Get the indices of the top_pann_to_keep highest logits
    highest_logit_indices = np.argsort(max_logits)[-n_frames:]
    calib_indices = argmax_logits[highest_logit_indices]
    # Sort calib_indices from lowest to highest
    calib_indices = np.sort(calib_indices)
    calib_indices = indices[calib_indices]

    i_logit = 1
    while (np.unique(calib_indices).shape[0] < n_frames) & (i_logit < logits.shape[0] - n_frames):
        highest_logit_indices = np.argsort(max_logits)[-n_frames-i_logit:]
        calib_indices = argmax_logits[highest_logit_indices]
        # Sort calib_indices from lowest to highest
        calib_indices = np.sort(calib_indices)
        calib_indices = indices[calib_indices]
        i_logit += 1
    calib_indices = np.unique(calib_indices)

    if len(calib_indices) != n_frames:
        raise ValueError("Unable to find enough unique highest logit indices")

    audios = np.array([get_audio_from_row(audio_path, k) for k in calib_indices])
    sum_datetimes = np.array([get_datetime_from_row(audio_path, k) for k in calib_indices])

    # sum_datetimes = dates[calib_indices]
    audio_files = concatenate_audios_with_overlap(audios, sr)

    df = pd.DataFrame({
        "datetimes": sum_datetimes,
        "row": calib_indices
    })

    df.to_csv(os.path.splitext(summary_path)[0]+".csv", index=False)
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

def get_faithful_summary(embeddings_path, audio_path, summary_path, sr, 
                         start_datetime, end_datetime, n_frames=12, audio_type="wav"):

    """
    Generates a reference faithful audio summary by selecting representative audio segments based on embeddings.
    This summary is then used to calibrate the faithfulness metric.

    This function processes audio embeddings within a specified time range, divides them into windows,
    computes mean embeddings for each window, and selects the closest actual embeddings to these means.
    The corresponding audio segments are then concatenated (with overlap) to form a summary audio file.
    A CSV file containing the datetimes and row indices of the selected segments is also generated.

    Parameters
    ----------
    embeddings_path : str
        Path to the file containing audio embeddings.
    audio_path : str
        Path to the audio data file.
    summary_path : str
        Output path for the generated summary audio file.
    sr : int
        Sample rate for audio processing.
    start_datetime : datetime or str
        Start datetime for selecting embeddings.
    end_datetime : datetime or str
        End datetime for selecting embeddings.
    n_frames : int, optional
        Number of summary segments (windows) to generate (default is 12).
    audio_type : str, optional
        Output audio format, either "wav" or "mp3" (default is "wav").

    Returns
    -------
    None
        The function saves the summary audio file and a CSV file with segment metadata to disk.

    Notes
    -----
    - The function expects helper functions such as `embeddings_generator`, `get_datetime_from_row`,
        `get_audio_from_row`, and `concatenate_audios_with_overlap` to be defined elsewhere.
    - If the output format is "mp3", the audio is converted to int16 and exported using Pydub.
    - The CSV file contains the datetimes and row indices of the selected summary segments.
    """

    full_data = np.array(list(embeddings_generator(embeddings_path, start_datetime=start_datetime, end_datetime=end_datetime)), dtype=object)
    embeddings = np.array([full_data[elem, 0] for elem in range(len(full_data))])
    embeddings_len = len(embeddings)
    indices = np.array([full_data[elem, 2] for elem in range(len(full_data))])

    # Split indices into n_frames windows
    window_size = embeddings_len // n_frames
    windows = [embeddings[i * window_size: (i + 1) * window_size] for i in range(n_frames)]

    # Randomly select one element from each window
    mean_embeddings = np.array([np.mean(x, axis=0) for x in windows])

    # Remove elements of audio until its length is n_frames
    while len(mean_embeddings) > n_frames:
        print('The audio contains more elements than n_frames. Removing the last element.')
        mean_embeddings = mean_embeddings[:-1]

    # Calculate distances between mean_embeddings_interpolated and embeddings
    distances = np.linalg.norm(embeddings[:, np.newaxis] - mean_embeddings, axis=2)
    print("Distances shape:", distances.shape)

    # Find the indices of the embeddings with the lowest distance to each mean_embeddings_interpolated
    closest_indices = np.argmin(distances, axis=0)
    calib_indices = indices[closest_indices]
    sum_datetimes = [get_datetime_from_row(embeddings_path, k) for k in calib_indices]

    audios = np.array([get_audio_from_row(audio_path, k) for k in calib_indices])
    audio_files = concatenate_audios_with_overlap(audios, sr)

    # Create DataFrame
    df = pd.DataFrame({
        "datetimes": sum_datetimes,
        "row": calib_indices
    })

    df.to_csv(os.path.splitext(summary_path)[0]+".csv", index=False)
    if audio_type == "mp3":
        # Ensure NumPy array is float32 (Pydub expects int16 for PCM)
        if audio_files.dtype != np.int16:
            audio_files = (audio_files * 32767).astype(np.int16)  # Convert from float to int16

        # Convert to bytes and create an AudioSegment with correct parameters
        audio_segment = AudioSegment(
            audio_files.tobytes(), frame_rate=sr, sample_width=2, channels=1
        )
        audio_segment.export(summary_path, format="mp3", bitrate="320k")
    else:
        write(summary_path, sr, audio_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a greedy summary from clusters and embeddings.")
    parser.add_argument("audio_path", type=str, help="Path to the HDF5 file containing audio.")
    parser.add_argument("embeddings_path", type=str, help="Path to the HDF5 file containing embeddings.")
    parser.add_argument("clusters_path", type=str, help="Path to the CSV file containing cluster data.")
    parser.add_argument("summary_path", type=str, help="Path to the output WAV file for the summary.")
    parser.add_argument("scen", type=float, help="Level of scenism (between 0 and 1).")
    parser.add_argument("--start_datetime", type=str, default=None, help="Start datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--end_datetime", type=str, default=None, help="End datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--sr", type=int, nargs="?", default=32000, help="Output sample rate, set by default to 32000.")
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

    get_greedy_summary(args.clusters_path, args.audio_path, args.embeddings_path, args.summary_path, args.sr, args.scen, 
                       start_datetime, end_datetime, args.n_frames, args.gain_db, False, args.limit_count, args.n_iter, 6, 3, 5, 10, args.greedy_batch, args.verbose, args.audio_type)