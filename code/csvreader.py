import numpy as np
import argparse
import os
import pandas as pd
from loguru import logger

def ShowCSV(csv_file, column_name=None):
    """
    ShowCSV allows to show a CSV file as a Pandas dataframe on terminal.
    It can also show a single column of the data.
    When importing csvreader as a module, in both cases ShowCSV returns the data as a NumPy array.

    Arguments:
    csv_file (string): Path to the CSV file
    column_name (string): Optional, default = None. Name of the column to select

    Return:
    numpy.ndarray: The function returns a NumPy array
    """
    try:
        df = pd.read_csv(csv_file, delimiter=';')
        if column_name is None:
            print(df)
            return np.array(df.values)
        else:
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
        ShowCSV(args.filename)
    elif args.command == "show_column":
        if not args.column:
            parser.error("The '--column' argument is required for 'show_column' command.")
        else:
            ShowCSV(args.filename, args.column)

if __name__ == "__main__":
    main()


