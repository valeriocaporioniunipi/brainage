import numpy as np
import argparse
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from abspath import AbsolutePath
from csvreader import GetData

def LinRegression(filename, test_size=0.2, random_state=42, plot_flag = False):
    """
    LinRegression performs linear regression on the given dataset.

    Arguments:
    - filename (str): path to the CSV file containing the dataset
    - test_size (float): optional, default = 0.2. Proportion of the dataset to include in the test split
    - random_state (int): optional, default = 42. Random seed for reproducibility

    Returns:
    - None. Prints evaluation metrics of the linear regression model.
    """
    # Load data
    x = GetData(filename)
    y = GetData(filename, "AGE_AT_SCAN")

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Initialize and fit linear regression model
    model = LinearRegression()
    model.fit(x_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(x_test_scaled)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if plot_flag == True:
        # plot the actual vs. predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Brain age')
        plt.grid(False)
        plt.show()


    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

def main():
    parser = argparse.ArgumentParser(description='Linear regression predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename", help="Name of the file that has to be analized")
    parser.add_argument("--location", default='data', help="Location of the file, i.e. folder containing it")
    parser.add_argument("--plot", action='store_true', help="Show the plot of actual vs predicted brain age")


    args = parser.parse_args()

    try:
        if not args.location:
            LinRegression(args.filename, plot_flag=args.plot)
        else:
            args.filename = AbsolutePath(args.filename, args.location)
            LinRegression(args.filename, plot_flag=args.plot)
    except FileNotFoundError:
        logger.error("File not found.")
        return None
        

if __name__ == "__main__":
    main()

