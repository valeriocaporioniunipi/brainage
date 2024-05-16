import numpy as np
import argparse
import os
import pandas as pd
from loguru import logger

def csv_reader(csv_file, column_name=None, show_flag=False):
    """
    csv_reader allows to read the data from a CSV file and converts them into a NumPy array.
    It can also show the entire dataset as a Pandas dataframe on terminal
    or show a single column of the data table.
    The csv_reader function does not show the dataframe, unless specified by changing show_flag argument.

    :param csvfile: path to the CSV file
    :type csvfile: str
    :param column_name: optional (default = None): name of the column to select
    :type column_name: str
    :param show_flag: optional (default = False): if True, the entire dataframe is shown.
    :type show_flag: bool
    :return: the function returns a multidimensional numpy array if no column_name is passed as argument, otherwise it returns a unidimensional numpy array 
    :rtype: numpy.ndarray
    
    """
    df = pd.read_csv(csv_file, delimiter=';')
    if column_name is None:
        if show_flag:
            print(df)
        network_input = np.array(df.values)
        return network_input
    else:
        if show_flag:
            print(df[column_name])
        return np.array(df[column_name].values)

def get_data(filename, target_name, ex_cols = 0, **kwargs):
    """
    get_data obtains the features and target arrays

    Arguments:
    - filename (str): name of the file which data are read from
    - target_name (str): name of the column of the csv file that contains targets
    - ex_cols (int): optional, default = 0. Excluded columns

    Return:
    - features (ndarray): array of features
    - targets (ndarray): array of targets
    - group (ndarray): array that distinguishes between groups (experimental vs control)
    """
    group_name = kwargs.get('group_name', None)
    logger.info(f'Reading data from file {os.path.basename(filename)}, with {target_name} as target column ')
    features = csv_reader(filename)[:, ex_cols:]
    targets = csv_reader(filename, target_name)
    # Checking if the first dimension of features matches the length of targets
    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match")
    if group_name:
        group = csv_reader(filename, group_name)
        return features, targets, group
    else:
        return features, targets

def oversampling(features, targets, **kwargs):
    # Step 1: Extract optional parameters
    bins = kwargs.get('bins', 10)
    group = kwargs.get('group', None)
    
    # Step 2: Calculate target histogram
    hist, edges = np.histogram(targets, bins=bins)
    
    # Step 3: Find the bin with the maximum count
    max_bin_index = np.argmax(hist)
    max_count = hist[max_bin_index]

    # Step 4: Check if the bin with the maximum count has samples available
    if max_count == 0:
        raise ValueError("No samples available in the bin with the maximum count for oversampling. Adjust bin size or provide more data.")

    # Step 5: Oversample the minority classes to match the maximum count
    oversampled_features = []
    oversampled_targets = []
    oversampled_group = [] if group is not None else None

    for i in range(bins - 1):
        # Step 6: Find indices of samples within the current bin
        bin_indices = np.where((targets >= edges[i]) & (targets < edges[i + 1]))[0]
        
        # Step 8: Randomly sample with replacement from the indices to match max_count
        sampled_indices = np.random.choice(bin_indices, size=max_count, replace=True)
        
        # Step 9: Append the sampled features and targets to the oversampled lists
        oversampled_features.append(features[sampled_indices])
        oversampled_targets.append(targets[sampled_indices])
        if group is not None:
            oversampled_group.append(group[sampled_indices])

    # Step 10: Concatenate the oversampled features and targets
    new_features = np.concatenate(oversampled_features)
    new_targets = np.concatenate(oversampled_targets)
    new_group = np.concatenate(oversampled_group) if group is not None else None

    # Step 11: Return the oversampled data
    return new_features, new_targets, new_group if group is not None else (new_features, new_targets)

def main():
    parser = argparse.ArgumentParser(description="CSV Reader - A tool to read CSV files with Pandas.")

    parser.add_argument("command", choices=["show", "show_column"], help="Choose the command to execute")
    parser.add_argument("filename", help="Name of the CSV file")
    parser.add_argument("--column", help="Name of the column to display (required for 'show_column' command)")

    args = parser.parse_args()

    try:
        if args.command == "show":
            csv_reader(args.filename, show_flag=True)
        elif args.command == "show_column":
            if not args.column:
                parser.error("The '--column' argument is required for 'show_column' command.")
            else:
                csv_reader(args.filename, args.column, show_flag=True)
    except FileNotFoundError as e:
        logger.error("File not found", e)

if __name__ == "__main__":
    main()
