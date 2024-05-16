import numpy as np
import argparse
import os
import pandas as pd
from loguru import logger


def GetData(csv_file, column_name=None, show_flag=False):

    """
    GetData allows to read the data from a CSV file and converts them into a NumPy array.
    It can also show the entire dataset as a Pandas dataframe on terminal
    or show a single column of the data table.
    the GetData function does not show the dataframe, unless specified by changing show_flag argument. 

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


def main():
    parser = argparse.ArgumentParser(description="CSV Reader - A tool to read CSV files with Pandas.")

    parser.add_argument("command", choices=["show", "show_column"], help="Choose the command to execute")
    parser.add_argument("filename", help="Name of the CSV file")
    parser.add_argument("--column", help="Name of the column to display (required for 'show_column' command)")

    args = parser.parse_args()

    try:
        if args.command == "show":
            GetData(args.filename, show_flag=True)
        elif args.command == "show_column":
            if not args.column:
                parser.error("The '--column' argument is required for 'show_column' command.")
            else:
                GetData(args.filename, args.column, show_flag=True)
    except FileNotFoundError as e:
        logger.error("File not found", e)

if __name__ == "__main__":
    main()
