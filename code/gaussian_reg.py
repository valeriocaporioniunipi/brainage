import numpy as np
import argparse
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from abspath import AbsolutePath
from csvreader import GetData

def GaussRegression(filename, n_splits, plot_flag=False):
    """
    GaussRegression performs a Gaussian regression with k-fold cross-validation on the given dataset.

    Arguments:
    - filename (str): path to the CSV file containing the dataset
    - n_splits (int): optional, default = 5. Number of folds for cross-validation
    - plot_flag (bool): optional, default = False. Whether to plot the actual vs. predicted values

    Returns:
    - None. Prints evaluation metrics of the gaussian regression model.
    """
    # Load data...
    #Importing features excluded first three columns: FILE_ID, AGE_AT_SCAN, SEX
    x = GetData(filename)[:, 3:] 
    y = GetData(filename, "AGE_AT_SCAN")

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits)

    # Initialize lists to store evaluation metrics
    mae_scores = []
    mse_scores = []
    r2_scores = []

    # Initialize figure for plotting
    plt.figure(figsize=(10, 8))

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(x_scaled), 1):
        # Split data into training and testing sets
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and fit linear regression model
        model = GaussianProcessRegressor()
        model.fit(x_train, y_train)

        # Predict on the test set
        y_pred = model.predict(x_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)

        # Plot actual vs. predicted values for current fold
        plt.scatter(y_test, y_pred, alpha=0.5, label=f'Fold {i} - MAE = {np.round(mae_scores[i-1], 2)}')

    # Plot the ideal line (y=x)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

    # Set plot labels and title
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted Brain Age')

    # Add legend and grid to the plot
    plt.legend()
    plt.grid(True)

    # Print average evaluation metrics over all folds
    print("Mean Absolute Error:", np.mean(mae_scores))
    print("Mean Squared Error:", np.mean(mse_scores))
    print("R-squared:", np.mean(r2_scores))

    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Gaussian regression with k-fold cross-validation predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename", help="Name of the file that has to be analyzed")
    parser.add_argument("--location", help="Location of the file, i.e. folder containing it")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds for k-folding cross-validation")
    parser.add_argument("--plot", action='store_true', help="Show the plot of actual vs predicted brain age")

    args = parser.parse_args()

    if args.n_splits > 0:
        try:
            if not args.location:
                GaussRegression(args.filename, n_splits=args.n_splits, plot_flag=args.plot)
            else:
                args.filename = AbsolutePath(args.filename, args.location)
                GaussRegression(args.filename, n_splits=args.n_splits, plot_flag=args.plot)
        except FileNotFoundError:
            logger.error("File not found.")
            return None
    else: 
        logger.error("Invalid number of folds: at least 5 folds required")

if __name__ == "__main__":
    main()
