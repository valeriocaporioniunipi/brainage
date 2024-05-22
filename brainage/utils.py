import os
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from neuroHarmonize import harmonizationLearn, harmonizationApply

def abs_path(local_filename, data_folder):
    """
    Gets the absolute path of the file given the name of the folder containing the data
    and the name of the file inside that folder and
    assuming that the repository contains a data folder
    and a code folder.

    :param local_filename: name of the data file
    :type local_filename: str
    :param data_folder: name of the folder which contains the data
    :type data_folder: str
    :return: the function returns the absolute path of the selected file
    :rtype: str
    
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))     # path of the code
    
    # Construct the absolute path to the data directory relative to the code directory
    data_dir = os.path.join(script_dir, "..", data_folder)
    
    # Construct the absolute path to the data file
    data_file_path = os.path.join(data_dir, local_filename)
    
    return data_file_path


def csv_reader(filename, column_name=None, show_flag=False):
    """
    Reads data from a CSV file and converts it into a NumPy array.
    Optionally displays the entire dataset or a single column.

    :param filename: Path to the CSV file
    :type filename: str
    :param column_name: Name of the column to select (optional)
    :type column_name: str, optional
    :param show_flag: If True, displays the dataframe
    :type show_flag: bool, optional
    :return: A Pandas dataframe of the entire dataset or the specified column
    :rtype: pandas.df
    """
    df = pd.read_csv(filename, delimiter=';')
    if column_name is None:
        if show_flag:
            print(df)
        return df
    else:
        if show_flag:
            print(df[column_name])
        return df[column_name]

def handle_spurious(df):
    """
    Handles spurious zeroes and -9999 values in the DataFrame.
    
    :param df: Input DataFrame
    :type df: pd.DataFrame
    :return: Cleaned DataFrame with spurious values handled
    :rtype: pd.DataFrame
    """
    # Replace -9999 with NaN
    df.replace(-9999, np.nan, inplace=True)
    # Replace 0 with NaN
    df.replace(0, np.nan, inplace=True)
    # Fill NaN values with the mean of the respective columns
    df.fillna(df.mean(), inplace=True)
    return df


def get_data(filename, target_col, ex_cols=0, **kwargs):
    """
    Obtains the features and target arrays from a CSV file. Optionally harmonizes the data 
    using neuroHarmonize and includes additional columns for grouping.

    :param filename: Path to the CSV file.
    :type filename: str
    :param target_col: Name of the target column.
    :type target_col: str
    :param ex_cols: Number of initial columns to exclude from the features (default is 0).
    :type ex_cols: int, optional
    :param kwargs: Additional keyword arguments:
                   - group_col: Name of the group column (optional).
                   - site_col: Name of the site column for harmonization (optional).
    :return: NumPy arrays of features, targets, and optionally the group.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray or None)
    """
    group_col = kwargs.get('group_col', None)
    site_col = kwargs.get('site_col', None)
    logger.info(f'Reading {os.path.basename(filename)} with {target_col} as target column')
    # Importing data from csv file as data
    data = pd.read_csv(filename, delimiter = ';')
    # 
    #if group_col is not None:
    #    data = data[data[group_col] == -1]

    # Excluding the first ex_cols columns
    features_df = data.iloc[:, ex_cols:]
    # Removing spurious values from features and convertin to numpy matrix
    features = handle_spurious(features_df).values
    # Target array (numpy.ndarray)
    targets = data[target_col].values
    if site_col in data.columns:
        covars = data[[site_col]]
        covars.loc[:, site_col] = covars[site_col].str.rsplit('_', n=1).str[0]
        covars.rename(columns={site_col: 'SITE'}, inplace=True)  # Rename the column
        _ , features = harmonizationLearn(features, covars)
        logger.info('Harmonizing data with neuroHarmonize ')

    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match ")
        raise ValueError("Mismatch between number of features and targets samples")
    if group_col:
        logger.info(" Splitting into experimental "
                    f"& control group. Group column has name {group_col}")
        group = data[group_col].values
        return features, targets, group
    # implicit else
    return features, targets

def p_value_emp(arr1, arr2, permutations=100000):
    '''
    Calculate the empirical p-value for the difference in means
    between two groups using permutation testing.

    :param array-like arr1: Data for the first group.
    :param array-like arr2: Data for the second group.
    :param int permutations: Number of permutations to perform
                                 for the permutation test. Default is 100,000.

    :return: Empirically calculated p-value for the observed difference in means.
    :rtype: float

    This function performs a permutation test to
    estimate the empirical p-value for the difference in means between two groups.
    The observed test statistic is the difference in means between arr2 and arr1.

    The function generates permuted test statistics by randomly permuting
    the data between the two groups and calculates the difference in means for each permutation.
    The empirical p-value is then calculated as the proportion
    of permuted differences in means that are greater than
    or equal to the observed difference in means.
    '''

    # Observed test statistic (difference in means)
    observed_stat = np.mean(arr2) - np.mean(arr1)

    # Initialize array to store permuted test statistics
    permuted_stat = np.zeros(permutations)

    # Perform permutations and calculate permuted test statistics
    for i in range(permutations):
        # Concatenate and randomly permute the data
        combined_data = np.concatenate((arr1, arr2))
        np.random.shuffle(combined_data)

        # Split permuted data into two groups
        permuted_arr1 = combined_data[:len(arr1)]
        permuted_arr2 = combined_data[len(arr1):]

        # Calculate test statistic for permuted data
        permuted_statistic = np.mean(permuted_arr2) - np.mean(permuted_arr1)

        # Store permuted statistic
        permuted_stat[i] = permuted_statistic

    # Calculate p-value
    p_value = np.sum(np.abs(permuted_stat) >= np.abs(observed_stat)) / permutations

    print("Empirical p-value:", p_value)

    return p_value


def oversampling(features, targets, **kwargs):
    """
    Oversamples minority classes in the dataset to balance class distribution.

    :param features: Feature array
    :type features: numpy.ndarray
    :param targets: Target array
    :type targets: numpy.ndarray
    :return: Oversampled features and targets arrays
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    bins = kwargs.get('bins', 10)
    group = kwargs.get('group', None)
    
    hist, edges = np.histogram(targets, bins=bins)
    
    max_bin_index = np.argmax(hist)
    max_count = hist[max_bin_index]

    if max_count == 0:
        raise ValueError("No samples available in the bin with the maximum count for oversampling. "
                         "Adjust bin size or provide more data.")

    oversampled_features = []
    oversampled_targets = []
    oversampled_group = [] if group is not None else None

    for i in range(bins - 1):
        bin_indices = np.where((targets >= edges[i]) & (targets < edges[i + 1]))[0]
        size = max_count
        sampled_indices = np.random.choice(bin_indices, size=size, replace=True)
        
        oversampled_features.append(features[sampled_indices])
        oversampled_targets.append(targets[sampled_indices])
        if group is not None:
            oversampled_group.append(group[sampled_indices])

    new_features = np.concatenate(oversampled_features)
    new_targets = np.concatenate(oversampled_targets)
    new_group = np.concatenate(oversampled_group) if group is not None else None

    if group is not None:
        return new_features, new_targets, new_group
    else:
        return new_features, new_targets, None

def group_selection(array, group, value):
    indices = np.where(group == value)[0]
    selected = array[indices]
    return selected

def csv_reader_parsing():
    """
    Command-line interface for reading and displaying CSV content,
    or stripping everything after '_' in a specified column, and converting to canonical base.
    Example usage:
        python script.py show path/to/file.csv
        python script.py show_column path/to/file.csv --column column_name
    """
    parser = argparse.ArgumentParser(description="CSV Reader "
                                    "A tool to read CSV files with Pandas.")
    parser.add_argument("command", choices=
                        ["show", "show_column"],
                        help="Choose the command to execute")
    parser.add_argument("filename",
                        help="Name of the CSV file")
    parser.add_argument("--column",
                        help="Name of the column to display or process"
                        " (req. for 'show_column' commands)")

    args = parser.parse_args()

    try:
        if args.command == "show":
            csv_reader(args.filename, show_flag=True)
        elif args.command == "show_column":
            if not args.column:
                parser.error("The '--column' argument is required for 'show_column' command.")
            else:
                csv_reader(args.filename, args.column, show_flag=True)
        else:
            logger.error("No command was given ")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Column '{args.column}' not found in the CSV file: {e}")

if __name__ == "__main__":
    csv_reader_parsing()