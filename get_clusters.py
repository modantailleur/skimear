import argparse
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
import csv
import random
from utils.summary_utils import batch_embeddings_generator, embeddings_generator, logits_generator


def get_kmeans_clusters(embeddings_path, cr, period, clusters_path, start_datetime, end_datetime, seed=0):
    """
    Performs per-period KMeans clustering on embeddings loaded from a specified HDF5 file, and writes cluster information to a CSV file.

    This function applies KMeans clustering on the embeddings of each time period independently, and extracts information 
    corresponding to each cluster medoid (point closest to the centroid).
    
    The number of clusters per period is determined by the cluster ratio (`cr`) and the batch size, with a minimum of 2 clusters
    per period. The results, including the datetime of each cluster medoid, the number of elements in each cluster,
    the batch period, and the corresponding row, are saved to a CSV file.
    Args:
        embeddings_path (str): Path to the file containing the embeddings.
        cr (float): Cluster ratio, used to determine the number of clusters per period.
        period (str): Time period for batching the embeddings.
        clusters_path (str): Path to the output CSV file for cluster information.
        start_datetime (datetime): Start datetime for filtering embeddings.
        end_datetime (datetime): End datetime for filtering embeddings.
        seed (int, optional): Random seed for KMeans initialization. Defaults to 0.
    Returns:
        None
    """

    all_datetime_clusters = []
    all_datetime_batches = []
    all_datetime_rows = []
    cluster_count = []

    for embeddings_batch, datetime_batch, row_batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=10, start_datetime=start_datetime, end_datetime=end_datetime):
        if np.all(embeddings_batch == 0):
            continue
        
        n_clusters = max(2, int((cr/100) * len(embeddings_batch))) # At least 2 clusters
        # Initialize KMeans with the specified number of clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed)

        # Fit KMeans to the data
        kmeans.fit(embeddings_batch)

        # Get the cluster labels for each embedding
        labels = kmeans.labels_

        # Get the indices of the embeddings that correspond to the centroids
        centroids = kmeans.cluster_centers_

        centroid_indices = [np.argmin(np.linalg.norm(embeddings_batch - centroid, axis=1)) for centroid in centroids]

        # Get the datetime values of the centroids
        datetime_clusters = datetime_batch[centroid_indices]
        all_datetime_clusters.extend(datetime_clusters)
        all_datetime_batches.extend([datetime_batch[0]]*len(datetime_clusters))

        # Populate the cluster_info_list with (num_elements, indices)
        for centroid_idx in centroid_indices:
            cluster_id = labels[centroid_idx]
            cluster_indices = np.where(labels == cluster_id)[0].tolist()
            num_elements = len(cluster_indices)
            cluster_count.append(num_elements)
            # Add the corresponding row to the all_datetime_rows list
            # ADD BATCH SIZE HERE
            all_datetime_rows.append(row_batch[centroid_idx])

    # Write datetime clusters to CSV file
    with open(clusters_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["clusters_datetimes", "cluster_count", "period", "row"])
        # Write datetime values, cluster count and datetime batches
        for dt, count, batch, row in zip(all_datetime_clusters, cluster_count, all_datetime_batches, all_datetime_rows):
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S.%f"), count, batch.strftime("%Y-%m-%d %H:%M:%S.%f"), row])



def get_double_kmeans_clusters(embeddings_path, cr, period, clusters_path, start_datetime, end_datetime, seed=0):
    """
    Performs a two-stage KMeans clustering on the embeddings of the different time periods,
     and writes cluster information to a CSV file.

    This function first applies KMeans clustering to batches of embeddings loaded from the specified path, 
    selecting cluster centroids and recording their associated datetimes and rows. It then performs a 
    global KMeans clustering on all embeddings within the specified datetime range to refine cluster counts. 
    The final cluster information, including centroid datetimes, cluster sizes, batch periods, and row indices, 
    is saved to a CSV file.

    Args:
        embeddings_path (str): Path to the file containing embeddings.
        cr (float): Cluster ratio, used to determine the number of clusters per batch.
        period (str): Time period for batching embeddings.
        clusters_path (str): Path to the output CSV file for cluster information.
        start_datetime (datetime): Start datetime for global clustering.
        end_datetime (datetime): End datetime for global clustering.
        seed (int, optional): Random seed for KMeans initialization. Defaults to 0.

    Returns:
        None
    """



    all_datetime_clusters = []
    all_datetime_batches = []
    all_datetime_rows = []
    all_embeddings = []
    cluster_count = []

    for embeddings_batch, datetime_batch, row_batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=10):

        n_clusters = max(2, int((cr/100) * len(embeddings_batch))) # At least 2 clusters
        # Initialize KMeans with the specified number of clusters
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=seed)

        # Fit KMeans to the data
        kmeans.fit(embeddings_batch)

        # Get the cluster labels for each embedding
        labels = kmeans.labels_

        # Get the indices of the embeddings that correspond to the centroids
        centroids = kmeans.cluster_centers_
        centroid_indices = [np.argmin(np.linalg.norm(embeddings_batch - centroid, axis=1)) for centroid in centroids]

        # Get the datetime values of the centroids
        datetime_clusters = datetime_batch[centroid_indices]
        all_datetime_clusters.extend(datetime_clusters)
        all_datetime_batches.extend([datetime_batch[0]]*len(datetime_clusters))
        all_embeddings.extend(embeddings_batch[centroid_indices])

        # Populate the cluster_info_list with (num_elements, indices)
        for centroid_idx in centroid_indices:
            cluster_id = labels[centroid_idx]
            cluster_indices = np.where(labels == cluster_id)[0].tolist()
            num_elements = len(cluster_indices)
            cluster_count.append(num_elements)
            # Add the corresponding row to the all_datetime_rows list
            # ADD BATCH SIZE HERE
            all_datetime_rows.append(row_batch[centroid_idx])

    #global count
    # Convert all_embeddings to a numpy array of type float64
    all_embeddings = np.array(all_embeddings)
    global_embeddings = np.array(list(embeddings_generator(embeddings_path, start_datetime=start_datetime, end_datetime=end_datetime)))
    global_kmeans = KMeans(n_clusters=len(all_datetime_batches), n_init='auto', random_state=seed)
    global_kmeans.fit(global_embeddings)
    global_labels = global_kmeans.labels_
    global_labels_embeddings = global_kmeans.predict(all_embeddings)
    global_cluster_count = [np.sum(global_labels == label) for label in global_labels_embeddings]
    cluster_count = [int((cluster_count[i]+global_cluster_count[i])/2) for i in range(len(cluster_count))]

    # Write datetime clusters to CSV file
    with open(clusters_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["clusters_datetimes", "cluster_count", "period", "row"])
        # Write datetime values, cluster count and datetime batches
        for dt, count, batch, row in zip(all_datetime_clusters, cluster_count, all_datetime_batches, all_datetime_rows):
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S.%f"), count, batch.strftime("%Y-%m-%d %H:%M:%S.%f"), row])

def get_random_clusters(embeddings_path, cr, period, clusters_path, start_datetime, end_datetime, seed=0):
    """
    Generates random clusters from batches of embeddings and writes cluster information to a CSV file.

    This function processes embeddings within a specified time period, randomly selects a number of clusters
    from each batch based on the cluster ratio (`cr`), and records the datetime and row information of each selected cluster.
    The cluster information, including the datetime, cluster count, period, and row, is saved to a CSV file.
    Args:
        embeddings_path (str): Path to the embeddings file or directory.
        cr (float): Cluster ratio (percentage) used to determine the number of clusters per batch.
        period (str): Time period for batching embeddings.
        clusters_path (str): Path to the output CSV file for cluster information.
        start_datetime (datetime): Start datetime for filtering embeddings.
        end_datetime (datetime): End datetime for filtering embeddings.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
    Returns:
        None
    """

    all_datetime_clusters = []
    all_datetime_batches = []
    all_datetime_rows = []
    cluster_count = []

    random.seed(seed)

    for embeddings_batch, datetime_batch, row_batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=10,
                                                                                     start_datetime=start_datetime, end_datetime=end_datetime):

        n_clusters = max(2, int((cr/100) * len(embeddings_batch))) # At least 2 clusters
        # Randomly select n_clusters different elements
        indices = list(range(len(embeddings_batch)))
        random.shuffle(indices)
        selected_indices = indices[:n_clusters]

        # Get the datetime values of the selected indices
        datetime_clusters = datetime_batch[selected_indices]
        all_datetime_clusters.extend(datetime_clusters)
        all_datetime_batches.extend([datetime_batch[0]]*len(datetime_clusters))

        # Populate the cluster_info_list with (num_elements, indices)
        for i, selected_idx in enumerate(selected_indices):
            cluster_indices = [selected_idx]
            num_elements = 1000 - i  # Since each selected element is its own cluster
            cluster_count.append(num_elements)
            # Add the corresponding row to the all_datetime_rows list
            all_datetime_rows.append(row_batch[selected_idx])

    # Write datetime clusters to CSV file
    with open(clusters_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["clusters_datetimes", "cluster_count", "period", "row"])
        # Write datetime values, cluster count and datetime batches
        for dt, count, batch, row in zip(all_datetime_clusters, cluster_count, all_datetime_batches, all_datetime_rows):
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S.%f"), count, batch.strftime("%Y-%m-%d %H:%M:%S.%f"), row])

def get_all_clusters(embeddings_path, cr, period, clusters_path, start_datetime, end_datetime):
    """
    Processes batches of embeddings within a specified time period, assigns each embedding to its own cluster,
    and writes cluster information to a CSV file. This function is not doing clustering at all, it is simply
    assigning each individual embedding to a different "cluster".

    Args:
        embeddings_path (str): Path to the file containing embeddings data.
        cr: Unused parameter (reserved for future use or compatibility).
        period (str): Time period for batching embeddings (e.g., 'hour', 'day').
        clusters_path (str): Path to the output CSV file where cluster information will be saved.
        start_datetime (datetime): Start of the time range for processing embeddings.
        end_datetime (datetime): End of the time range for processing embeddings.

    The function iterates over batches of embeddings, assigns each embedding to a separate cluster,
    and records the cluster's datetime, count, period, and associated row data. The results are saved
    in a CSV file with columns: clusters_datetimes, cluster_count, period, and row.
    """

    all_datetime_clusters = []
    all_datetime_batches = []
    all_datetime_rows = []
    cluster_count = []

    for embeddings_batch, datetime_batch, row_batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=10, 
                                                                                     start_datetime=start_datetime, end_datetime=end_datetime):
        # Randomly select n_clusters different elements
        indices = list(range(len(embeddings_batch)))
        selected_indices = indices

        # Get the datetime values of the selected indices
        datetime_clusters = datetime_batch[selected_indices]
        all_datetime_clusters.extend(datetime_clusters)
        all_datetime_batches.extend([datetime_batch[0]]*len(datetime_clusters))

        # Populate the cluster_info_list with (num_elements, indices)
        for i, selected_idx in enumerate(selected_indices):
            num_elements = 1000  # Since each selected element is its own cluster
            cluster_count.append(num_elements)
            # Add the corresponding row to the all_datetime_rows list
            all_datetime_rows.append(row_batch[selected_idx])

    # Write datetime clusters to CSV file
    with open(clusters_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["clusters_datetimes", "cluster_count", "period", "row"])
        # Write datetime values, cluster count and datetime batches
        for dt, count, batch, row in zip(all_datetime_clusters, cluster_count, all_datetime_batches, all_datetime_rows):
            writer.writerow([dt.strftime("%Y-%m-%d %H:%M:%S.%f"), count, batch.strftime("%Y-%m-%d %H:%M:%S.%f"), row])

def get_clusters_mean_emb(embeddings_path, clusters_path, period=30, start_datetime=None, end_datetime=None):
    """
    Computes the mean embeddings for batches of embeddings within a specified time period and saves them to a file.

    Args:
        embeddings_path (str): Path to the file containing the embeddings data.
        clusters_path (str): Path used to generate the output filename for the mean embeddings.
        period (int, optional): Time period (in minutes or other units) for batching embeddings. Defaults to 30.
        start_datetime (datetime, optional): Start datetime for filtering embeddings. Defaults to None.
        end_datetime (datetime, optional): End datetime for filtering embeddings. Defaults to None.

    Returns:
        None: The function saves the computed mean embeddings to a numpy file and does not return any value.

    Description:
        This function loads batches of embeddings from the specified path, computes the mean embedding for each batch,
        and saves the resulting array of mean embeddings to a new file. The output filename is generated by modifying
        the provided clusters_path, replacing its last part with 'meanembeddings.npy'.
    """



    embeddings = np.array(
        [batch[0] for batch in batch_embeddings_generator(embeddings_path, period=period, batch_lim=1, start_datetime=start_datetime, end_datetime=end_datetime)],
        dtype=object
    )

    mean_embeddings = np.array([np.mean(x, axis=0) for x in embeddings])

    # Split clusters_path and replace the last part with "meanembeddings.npy"
    path_parts = clusters_path.split("_")
    path_parts[-1] = "meanembeddings.npy"
    meanembeddings_path = "_".join(path_parts)

    # Save mean_embeddings to a numpy file
    np.save(meanembeddings_path, mean_embeddings)

def get_clusters_max_logits(logits_path, clusters_path, start_datetime=None, end_datetime=None):
    """
    Computes the maximum logits from a sequence of logits and saves them to a file.

    Args:
        logits_path (str): Path to the file or directory containing logits data.
        clusters_path (str): Path used as a template for saving the output file. The last part of the filename is replaced with 'maxlogits.npy'.
        start_datetime (optional): Start datetime for filtering logits data. Format and type depend on the implementation of `logits_generator`.
        end_datetime (optional): End datetime for filtering logits data. Format and type depend on the implementation of `logits_generator`.

    Returns:
        None

    Side Effects:
        Saves the computed maximum logits as a NumPy array to a file with the modified path based on `clusters_path`.
    """

    full_out = np.array(list(logits_generator(logits_path, start_datetime=start_datetime, end_datetime=end_datetime)), dtype=object)
    logits = np.array([full_out[elem, 0] for elem in range(len(full_out))])
    max_logits = np.max(logits, axis=0)

    # Split clusters_path and replace the last part with "maxlogits.npy"
    path_parts = clusters_path.split("_")
    path_parts[-1] = "maxlogits.npy"
    maxlogits_path = "_".join(path_parts)

    # Save max_logits to a numpy file
    np.save(maxlogits_path, max_logits)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster embeddings using KMeans.")
    parser.add_argument("embeddings_path", type=str, help="Path to the HDF5 file containing embeddings.")
    parser.add_argument("clusters_path", type=str, help="Path to the output CSV file for cluster data.")
    parser.add_argument("--start_datetime", type=str, default=None, help="Start datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--end_datetime", type=str, default=None, help="End datetime for clustering in format YYYY-MM-DD HH:MM:SS. Defaults to None.")
    parser.add_argument("--cr", type=float, help="Cluster ratio. By default, set to 15%.", default=15)
    parser.add_argument("--period", type=int, help="Time period for batching embeddings (in minutes). By default, set to 15 minutes.", default=15)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for KMeans.")

    args = parser.parse_args()

    start_datetime = datetime.strptime(args.start_datetime, "%Y-%m-%d %H:%M:%S") if args.start_datetime else None
    end_datetime = datetime.strptime(args.end_datetime, "%Y-%m-%d %H:%M:%S") if args.end_datetime else None

    get_kmeans_clusters(args.embeddings_path, args.cr, args.period, args.clusters_path, start_datetime, end_datetime, args.seed)


