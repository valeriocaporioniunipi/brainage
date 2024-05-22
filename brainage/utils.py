import os
import argparse
import numpy as np
import pandas as pd
from loguru import logger
from neuroHarmonize import harmonizationLearn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from smogn import smoter  # SmoteR for regression-oriented oversampling


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
    script_dir = os.path.dirname(os.path.abspath(__file__))  # path of the code

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

    # Run this snippet to see what are columns having the worst values
    for i in range(df.shape[1]):
        for j in range(df.shape[0]):
            value = df.iloc[j, i]
            if value == -9999 or value == 0:
                print(value, (i, j), df.columns[i])

    # As manually observed, column "5th-Ventricle_Volume_mm3"
    # seems to have quite dirty data, so we remove it
    df.drop("5th-Ventricle_Volume_mm3", axis="columns", inplace=True)

    # Replace -9999 with NaN
    df.replace(-9999, np.nan, inplace=True)
    # Replace 0 with NaN
    df.replace(0, np.nan, inplace=True)

    # Fill NaN values with the mean of the respective columns
    # Temporarily disabled, first we have to make sure all entries are numbers
    # df.fillna(df.mean(), inplace=True)

    # Remove all other rows containing dirty data
    df.dropna(inplace=True)
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
                   -overs: Boolean flag in order to perform SmoteR oversampling (optional).
    :return: NumPy arrays of features, targets, and optionally the group.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray or None)
    """
    group_col = kwargs.get('group_col', None)
    site_col = kwargs.get('site_col', None)
    overs = kwargs.get('overs', False)
    logger.info(f'Reading {os.path.basename(filename)} with {target_col} as target column')
    # Importing data from csv file as data
    data = pd.read_csv(filename, delimiter=';')
    if overs:
        data = smoter(data, target_col)
        logger.info('Oversampling performed with SmoteR')
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
        _, features = harmonizationLearn(features, covars)
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


def p_value_emp(arr1, arr2, permutations=10000):
    '''
    Calculate the empirical p-value for the difference in means
    between two groups using permutation testing. The empirical p-value is
    the proportion of permuted differences in means that are greater than
    or equal to the observed difference in means.

    :param arr1: Data for the first group.
    :type arr1: ndarray
    :param arr2: Data for the second group.
    :type arr2: ndarray
    :param permutations: Number of permutations to perform
                                 for the permutation test (default = 10 000)
    :type permutations: int

    :return: Empirically calculated p-value for the observed difference in means.
    :rtype: float
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


def group_selection(array, group, value):
    indices = np.where(group == value)[0]
    selected = array[indices]
    return selected


def new_prediction(features, targets, model):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    y_pred = model.predict(features)
    mae = mean_absolute_error(targets, y_pred)
    r2 = r2_score(targets, y_pred)
    print("Mean Absolute Error on exp:", mae)
    print("R-squared on exp:", r2)

    _, axa = plt.subplots(figsize=(10, 8))
    target_range = [targets.min(), targets.max()]
    # Plot the ideal line (y=x)
    axa.plot(target_range, target_range, 'k--', lw=2)
    axa.scatter(targets, y_pred, color='k', alpha=0.5,
                label=f'MAE : {mae:.2} y\n$R^2$ : {r2:.2}')

    # Set plot labels and title
    axa.set_xlabel('Actual age [y]', fontsize=20)
    axa.set_ylabel('Predicted age [y]', fontsize=20)
    axa.set_title('Actual vs. predicted age - ASD', fontsize=24)

    # Add legend and grid to the plot
    axa.legend(fontsize=16)
    axa.grid(False)
    # plt.savefig('linear_reg_exp.png', transparent = True)
    pad_new = y_pred.ravel() - targets
    return pad_new


def csv_reader_parsing():
    """
    Command-line interface for reading and displaying CSV content
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
    parser.add_argument("--delimiter",
                        help="Delimiter used in the CSV file")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.filename, delimiter=args.delimiter)
        if args.command == "show":
            print(df)
            return df
        elif args.command == "show_column":
            if not args.column:
                parser.error("The '--column' argument is required for 'show_column' command.")
            else:
                column_df = df[[args.column]]
                print(column_df)
                return column_df
        else:
            logger.error("No command was given ")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Column '{args.column}' not found in the CSV file: {e}")


if __name__ == "__main__":
    csv_reader_parsing()

    # Uncomment for a rapid test
    # df = csv_reader("../data/FS_features_ABIDE_males.csv")
    # handle_spurious(df)
