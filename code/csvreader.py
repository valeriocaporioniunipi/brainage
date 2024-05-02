import numpy as np
import argparse
import os
import pandas as pd
from loguru import logger

def GetData(csv_file, column_name = None, show_flag = False):
    """
    GetData allows to read the data from a CSV file and converts them into a NumPy array.
    It can also show the entire dataset as a Pandas dataframe on terminal
    or show a single column of the data table.
    When importing this code as a module by writing

    from csvreader import GetData

    the GetData function does not show the dataframe, unless specified by changing show_flag argument. 

    Arguments:
    csv_file (string): Path to the CSV file
    column_name (string): Optional, default = None. Name of the column to select
    show_flag (bool): Optional, default = False. If True, the entire dataframe is shown.

    Return:
    numpy.ndarray: The function returns a NumPy array
    """
    try:
        df = pd.read_csv(csv_file, delimiter=';')
        if column_name is None:
            if show_flag == True:
                print(df)
            network_input = np.array(df.values)[:, 2:] # stripping the first two columns (FILE_ID and AGE)
            return network_input
        else:
            if show_flag == True:
                print(df[column_name])
            return np.array(df[column_name].values)
    except FileNotFoundError:
        logger.error("File not found.")
        return None

def main():
    parser = argparse.ArgumentParser(description="CSV Reader - A tool to read CSV files with Pandas.")

    parser.add_argument("command", choices=["show", "show_column"], help="Choose the command to execute")
    parser.add_argument("filename", help="Name of the CSV file")
    parser.add_argument("--column", help="Name of the column to display (required for 'show_column' command)")

    args = parser.parse_args()

    if args.command == "show":
        GetData(args.filename, show_flag = True)
    elif args.command == "show_column":
        if not args.column:
            parser.error("The '--column' argument is required for 'show_column' command.")
        else:
            GetData(args.filename, args.column, show_flag = True)

if __name__ == "__main__":
    main()
