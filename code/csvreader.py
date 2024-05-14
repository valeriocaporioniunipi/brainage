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
    When importing this code as a module by writing

    from csvreader import csv_reader

    the csv_reader function does not show the dataframe, unless specified by changing show_flag argument. 

    Arguments:
    -csv_file (str): path to the CSV file
    -column_name (str): optional, default = None. Name of the column to select
    -show_flag (bool): optional, default = False. If True, the entire dataframe is shown.

    Return:
    numpy.ndarray: The function returns a NumPy array
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

def get_data(filename, target_name, ex_cols = 0):
    """
    get_data obtains the features and target arrays

    Arguments:
    - filename (str): name of the file which data are read from
    - target_name (str): name of the column of the csv file that contains targets
    - ex_cols (int): optional, default = 0. Excluded columns

    Return:
    - features (ndarray): array of features
    - targets (ndarray): array of targets
    """
    logger.info(f'Reading data from file {filename}, with {target_name} as target column ')
    features = csv_reader(filename)[:, ex_cols:]
    targets = csv_reader(filename, target_name)

    # Checking if the first dimension of features matches the length of targets
    if len(features) != len(targets):
        logger.error("Number of samples in features and targets do not match")
    
    return features, targets


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
