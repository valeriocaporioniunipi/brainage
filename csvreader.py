import numpy as np
import argparse
import os
import pandas as pd
from loguru import logger

def ShowCSV(csv_file, column_name=None):
    try:
        df = pd.read_csv(csv_file, delimiter=';')
        if column_name is None:
            print(df)
            return df.values
        else:
            print(df[column_name])
            return df[column_name].values
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


