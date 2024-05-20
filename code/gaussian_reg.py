import numpy as np
import argparse
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from code.abspath import abs_path
from code.csvreader import get_data

def gaussian_reg(features, target, n_splits,  plot_flag=False):
    """
    gaussian_reg performs gaussian regression with k-fold cross-validation on the
    given dataset and prints evaluation metrics of the gaussian regression model
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared.

    :param filename: path to the CSV file containing the dataset 
    :type filename: str
    :param n_splits: number of folds for cross-validation
    :type n_splits: int
    :param plot_flag: optional (default = False): Whether to plot the actual vs. predicted values
    :type plot_flag: bool
    :return: None

    """
    # Loading data...
    #Importing features excluded first three columns: FILE_ID, AGE_AT_SCAN, SEX
    x = features 
    y = target

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

    # Print average evaluation metrics over all folds
    meaned_mae = np.mean(mae_scores)
    meaned_mse = np.mean(mse_scores)
    meaned_r2 = np.mean(r2_scores)

    print("Mean Absolute Error:", meaned_mae)
    print("Mean Squared Error:", meaned_mse)
    print("R-squared:", meaned_r2)

    if plot_flag:

        # Plot the ideal line (y=x)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

        # Set plot labels and title
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Brain Age')

        # Add legend and grid to the plot
        plt.legend()
        plt.grid(True)


        # Show the plot
        plt.show()

    return meaned_mae, meaned_mse, meaned_r2

def gaussian_reg_parsing():
    """
    gaussian_reg function parsed that runs when the .py file is called.
    It performs a  gaussian regression with k-fold cross-validation
    predicting the age of patients from magnetic resonance imaging and
    prints evaluation metrics of the linear regression model 
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared.
    There are two ways to pass the csv file to this function. It's possible to
    pass the absolutepath of the dataset or you can store the dataset in a brother folder
    of the one containing code, and pass to the parsing function the filename and his container-folder.
    The parameters listed below are not parameters of the functions but are parsing arguments that have 
    to be passed to command line when executing the program as follow:

    .. code::

        $Your_PC>python gaussian_reg.py file.csv --target --location --folds --ex_cols --plot 

    where file.csv is the only mandatory argument, while others are optional and takes some default values,
    that if they have to be modified you can write for example:

    .. code::

        $Your_PC>python gaussian_reg.py file.csv --folds 10  

    :param filename: path to the CSV file containing the dataset or the name of the file if --location argument is passed 
    :type filename: str
    :param target: optional (default = AGE_AT_SCAN): Name of the column holding target values
    :type target: str
    :param location: optional: Location of the file, i.e. folder containing it 
    :type location: str
    :param folds: optional (>4, default 5):number of folds for cross-validation
    :type folds: int
    :param ex_cols: optional (default = 3): columns excluded when importing file
    :type ex_cols: int
    :param plot: optional (default = False): Show the plot of actual vs predicted brain age
    :type plot: bool
    :return: None

    """
    parser = argparse.ArgumentParser(description=
        'Gaussian regression predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename",
                         help="Name of the file that has to be analized")
    parser.add_argument("--target", default = "AGE_AT_SCAN",
                        help="Name of the colums holding target values")
    parser.add_argument("--location",
                         help="Location of the file, i.e. folder containing it")
    parser.add_argument("--folds", type = int, default = 5,
                         help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type = int, default = 3,
                         help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--plot", action="store_true",
                         help="Show the plot of actual vs predicted brain age")

    args = parser.parse_args()

    if args.folds > 4:
        try:
            args.filename = abs_path(args.filename,args.location) if args.location else args.filename
            logger.info(f"Opening file : {args.filename}")
            features, targets = get_data(args.filename, args.target, args.ex_cols)
            gaussian_reg(features, targets, args.folds, args.plot)
        except FileNotFoundError:
            logger.error("File not found.")
    else:
        logger.error("Invalid number of folds: at least 5 folds required.")


if __name__ == "__main__":
    gaussian_reg_parsing()

