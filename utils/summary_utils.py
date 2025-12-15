import numpy as np
from datetime import datetime, timedelta
import h5py
from scipy.integrate import trapezoid

def get_embedding_from_row(embeddings_path, row_index):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        embedding_dataset = hf["embeddings"]
        embedding = embedding_dataset[row_index]
        return embedding

def get_logit_from_row(logits_path, row_index):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(logits_path, "r") as hf:
        logits_dataset = hf["logits"]
        logit = logits_dataset[row_index]
        return logit

def get_dummy_embedding_from_row(embeddings_path):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        embedding_dataset = hf["embeddings"]
        embedding = embedding_dataset[0]
        embedding = np.full(embedding.shape, -100)
        return embedding

def get_audio_from_row(embeddings_path, row_index):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        audio_dataset = hf["audio"]
        audio = audio_dataset[row_index]
        return audio

def get_datetime_from_row(embeddings_path, row_index):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        datetime_dataset = hf["datetime"]
        my_datetime = datetime_dataset[row_index]
        timestamp_str = my_datetime.decode()
        current_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
        output = current_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
        return output

def get_embedding_len(embeddings_path):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        audio_dataset = hf["embeddings"]
        num_rows = audio_dataset.shape[0]  # Number of rows in the dataset
        return num_rows
    
def get_audio_len(embeddings_path):
    """Retrieve a single row from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        audio_dataset = hf["audio"]
        num_rows = audio_dataset.shape[0]  # Number of rows in the dataset
        return num_rows
    
def embeddings_generator(embeddings_path, start_datetime=None, end_datetime=None):
    """Generator to yield rows from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        embedding_dataset = hf["embeddings"]
        datetime_dataset = hf["datetime"]
        num_rows = embedding_dataset.shape[0]
        for i in range(0, num_rows):
            embedding_i = embedding_dataset[i]
            datetime_i = datetime.strptime(datetime_dataset[i].decode(), "%Y-%m-%d %H:%M:%S.%f")
            if start_datetime is not None:
                if datetime_i < start_datetime:
                    continue
            if end_datetime is not None:
                if datetime_i > end_datetime:
                    continue
            yield embedding_i, datetime_i, i

def logits_generator(embeddings_path, start_datetime=None, end_datetime=None):
    """Generator to yield rows from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        logits_dataset = hf["logits"]
        datetime_dataset = hf["datetime"]
        num_rows = logits_dataset.shape[0]
        for i in range(0, num_rows):
            logit_i = logits_dataset[i]
            datetime_i = datetime.strptime(datetime_dataset[i].decode(), "%Y-%m-%d %H:%M:%S.%f")
            if start_datetime is not None:
                if datetime_i < start_datetime:
                    continue
            if end_datetime is not None:
                if datetime_i > end_datetime:
                    continue
            yield logit_i, datetime_i, i

def audio_generator(embeddings_path, start_datetime=None, end_datetime=None):
    """Generator to yield rows from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        audio_dataset = hf["audio"]
        datetime_dataset = hf["datetime"]
        num_rows = audio_dataset.shape[0]
        for i in range(0, num_rows):
            audio_i = audio_dataset[i]
            datetime_i = datetime.strptime(datetime_dataset[i].decode(), "%Y-%m-%d %H:%M:%S.%f")
            if start_datetime is not None:
                if datetime_i < start_datetime:
                    continue
            if end_datetime is not None:
                if datetime_i > end_datetime:
                    continue
            yield audio_i, datetime_i, i

def datetime_generator(embeddings_path, start_datetime=None, end_datetime=None):
    """Generator to yield rows from the embeddings dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        datetime_dataset = hf["datetime"]
        num_rows = datetime_dataset.shape[0]
        for i in range(0, num_rows):
            datetime_i = datetime.strptime(datetime_dataset[i].decode(), "%Y-%m-%d %H:%M:%S.%f")
            if start_datetime is not None:
                if datetime_i < start_datetime:
                    continue
            if end_datetime is not None:
                if datetime_i > end_datetime:
                    continue
            yield datetime_i, i

def mean_batch_logits_generator(embeddings_path, batch_size=360, start_datetime=None, end_datetime=None):
    """Generator to yield batches of rows from the logits dataset."""
    with h5py.File(embeddings_path, "r") as hf:
        logit_dataset = hf["logits"]
        datetime_dataset = hf["datetime"]
        num_rows = logit_dataset.shape[0]
        for i in range(0, num_rows, batch_size):
            logit_i = logit_dataset[i:i+batch_size]
            datetime_i = datetime.strptime(datetime_dataset[i].decode(), "%Y-%m-%d %H:%M:%S.%f")
            
            if start_datetime is not None:
                if datetime_i < start_datetime:
                    continue
            if end_datetime is not None:
                if datetime_i > end_datetime:
                    continue
            yield np.mean(logit_i, axis=0)

def batch_embeddings_generator(embeddings_path, period=30, batch_lim=40, start_datetime=None, end_datetime=None):
    """Generator to yield batches of rows from the embeddings dataset based on fixed half-hour periods."""
    with h5py.File(embeddings_path, "r") as hf:
        embedding_dataset = hf["embeddings"]
        datetime_dataset = hf["datetime"]

        # logits_dataset = hf["logits"]
        num_rows = embedding_dataset.shape[0]

        current_batch_embeddings = []
        current_batch_datetimes = []
        current_batch_logits = []
        current_batch_rows = []

        period_timedelta = timedelta(minutes=period)
        current_period_start = None
        current_period_end = None

        for i in range(num_rows):
            timestamp_str = datetime_dataset[i].decode()
            if timestamp_str == '':
                continue
            current_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")

            if start_datetime is not None:
                if current_timestamp < start_datetime:
                    continue
            if end_datetime is not None:
                if current_timestamp > end_datetime:
                    continue

            # Initialize the start of the period if not already set
            if current_period_start is None:
                # Align the current timestamp to the nearest half-hour period
                minute = (current_timestamp.minute // period) * period
                current_period_start = current_timestamp.replace(minute=minute, second=0, microsecond=0)
                current_period_end = current_period_start + period_timedelta

            # Check if the current timestamp falls within the current period
            if current_period_start <= current_timestamp < current_period_end:
                current_batch_embeddings.append(embedding_dataset[i])
                current_batch_datetimes.append(current_timestamp)
                # current_batch_logits.append(logits_dataset[i])
                current_batch_rows.append(i)
            else:
                # Yield the current batch if it has more than n_clusters elements
                # Check if the current period ends before the specified date

                if len(current_batch_embeddings) > batch_lim:
                    yield np.array(current_batch_embeddings), np.array(current_batch_datetimes), np.array(current_batch_rows)
                    # Start a new period
                    current_batch_embeddings = [embedding_dataset[i]]
                    # current_batch_logits = [logits_dataset[i]]
                    current_batch_rows = [i]
                    current_batch_datetimes = [current_timestamp]
                    current_period_start = current_period_end
                    current_period_end = current_period_start + period_timedelta
                else:
                    # Reset current_period_start to the beginning of the current minute
                    current_period_start = current_timestamp.replace(minute=minute, second=0, microsecond=0)
                    current_period_end = current_period_start + period_timedelta

        # Yield the final batch if it has more than n_clusters elements
        if len(current_batch_embeddings) > batch_lim:
            # yield np.array(current_batch_embeddings), np.array(current_batch_datetimes), np.array(current_batch_logits), np.array(current_batch_rows)
            yield np.array(current_batch_embeddings), np.array(current_batch_datetimes), np.array(current_batch_rows)

# MATHIEU'S AUC formula proposal
def auc(points, num_samples=100):
    """
    Compute AUC by:
    - Interpolating original curve on uniform grid
    - Adding virtual starting point (0, y0) AFTER resampling
    - Sorting to integrate properly
    """
    if len(points) == 0:
        return 0.0

    # Sort input points by x
    points = points[np.argsort(points[:, 0])]
    x = points[:, 0]
    y = points[:, 1]

    # --- Resample only over actual data domain ---
    x_uniform = np.linspace(x.min(), x.max(), num_samples)
    y_uniform = np.interp(x_uniform, x, y)

    # --- Add virtual starting point AFTER resampling ---
    x_virtual = np.insert(x_uniform, 0, 0.0)
    y_virtual = np.insert(y_uniform, 0, y[0])

    # --- Ensure monotonic order for integration ---
    sort_idx = np.argsort(x_virtual)
    x_virtual = x_virtual[sort_idx]
    y_virtual = y_virtual[sort_idx]

    # --- Compute area ---
    area = trapezoid(y_virtual, x_virtual)

    # Return everything
    augmented_points = np.column_stack((x_virtual, y_virtual))
    return area

# OLD AUC before Mathieu's proposal
# def auc(points):
#     """
#     Compute area under a monotonic curve derived from points.
#     Enforces:
#       - Non-decreasing x (TC)
#       - Non-increasing y (SD)
#     while keeping the original z order.

#     Returns:
#         area: scalar float
#         filtered: cleaned curve points (including virtual starting point)
#         kept_idx: indices of kept points (in original order)
#     """
#     if len(points) == 0:
#         return 0.0, np.array([]), []

#     # Always start with the first original point
#     filtered = [points[0]]
#     kept_idx = [0]

#     for i, (x, y) in enumerate(points[1:], start=1):
#         # Skip if SD increases
#         if y > filtered[-1][1]:
#             continue
#         # Remove previous points if TC decreases
#         while any(fx > x for fx, _ in filtered):
#             filtered.pop()
#             kept_idx.pop()
#         filtered.append([x, y])
#         kept_idx.append(i)

#     filtered = np.array(filtered)

#     # Add virtual starting point at x=0 with same y as the first point
#     virtual_start = np.array([[0.0, filtered[0, 1]]])
#     filtered_with_virtual = np.vstack([virtual_start, filtered])

#     # Handle single-point case
#     if len(filtered_with_virtual) == 1:
#         area = 0.0
#     else:
#         dx = np.diff(filtered_with_virtual[:, 0])
#         area = np.sum(dx * filtered_with_virtual[:-1, 1])

#     return area

# def auc(x):
#     """
#     x: 2D numpy array --> axis 0: x-axis values, axis 1: y-axis values 
#     """

#     x_diff = np.diff(x[:, 0])
#     y_diff = np.diff(x[:, 1])

#     # Remove elements that are above the previous x or y value
#     is_counted = (x_diff >= 0) & (y_diff <= 0)
#     is_counted = np.insert(is_counted, 0, True)

#     x_counted = x[is_counted,:]
#     auc_x = np.array([x_counted[k,0] - x_counted[k-1, 0] if k != 0 else x_counted[k, 0]
#                            for k in range(x_counted.shape[0])])
#     auc_y = x_counted[:, 1]
#     auc_value = np.sum(auc_x * auc_y)

#     return(auc_value)