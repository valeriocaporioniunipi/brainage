import numpy as np
import argparse
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

from utils import abs_path, get_data, p_value_emp

def group_selection(array, group, value):
    indices = np.where(group == value)[0]
    selected = array[indices]
    return selected

def linear_reg(features, targets, n_splits):

    """
    linear_reg performs linear regression with k-fold cross-validation on the
    given dataset and prints evaluation metrics of the linear regression model
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared.

    :param filename: path to the CSV file containing the dataset 
    :type filename: str
    :param n_splits: number of folds for cross-validation
    :type n_splits: int
    :param plot_flag: optional (default = False): Whether to plot the actual vs. predicted values
    :type plot_flag: bool
    :return: None
    """

    # Initialize data standardization (done after the k-folding split to avoid leakage)
    scaler = StandardScaler()

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle = True, random_state= 42)

    # Initialize lists to store evaluation metrics and prediction-actual-difference list
    mae_scores, r2_scores = [],[]
    pad_control = []

    # Initialization in order to find the best model parameters
    best_model = None
    mae_best = float('inf')

    figc, axc = plt.subplots(figsize=(10, 8))

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Split data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Initialize and fit linear regression model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Predict on the test set
        y_pred = model.predict(x_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        if mae < mae_best:
            mae_best = mae
            best_model = model

        mae_scores.append(mae)
        r2_scores.append(r2)
        pad_control.extend(y_pred.ravel()-y_test)

        # Plot actual vs. predicted values for current fold
        axc.scatter(y_test, y_pred, alpha=0.5, label =f'MAE : {mae:.2} y')

    # Print average evaluation metrics over all folds
    mae, r2 = np.mean(mae_scores), np.mean(r2_scores)
    print("Mean Absolute Error on control:", mae)
    print("R-squared on control:", r2)

    target_range = [targets.min(), targets.max()]
    # Plotting the ideal line (y=x)
    axc.plot(target_range, target_range, 'k--', lw=2)

    # Set plot labels and title
    axc.set_xlabel('Actual age [y]', fontsize = 20)
    axc.set_ylabel('Predicted age [y]', fontsize = 20)
    axc.set_title('Actual vs. predicted age - control', fontsize = 24)

    # Add legend and grid to the plot
    axc.legend(fontsize = 16)
    axc.grid(False)
    # plt.savefig('linear_reg_control.png', transparent = True)
    
    return best_model, mae, r2, pad_control

def lin_ads_prediction(features, targets, model):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    y_pred = model.predict(features)
    mae = mean_absolute_error(targets, y_pred)
    r2 = r2_score(targets, y_pred)
    print("Mean Absolute Error on exp:", mae)
    print("R-squared on exp:", r2)

    figa, axa = plt.subplots(figsize=(10, 8))
    target_range = [targets.min(), targets.max()]
    # Plot the ideal line (y=x)
    axa.plot(target_range, target_range, 'k--', lw=2)
    axa.scatter(targets, y_pred, color = 'k', alpha =0.5,
                label =f'MAE : {mae:.2} y\n$R^2$ : {r2:.2}')

    # Set plot labels and title
    axa.set_xlabel('Actual age [y]', fontsize = 20)
    axa.set_ylabel('Predicted age [y]', fontsize = 20)
    axa.set_title('Actual vs. predicted age - ASD', fontsize = 24)

    # Add legend and grid to the plot
    axa.legend(fontsize = 16)
    axa.grid(False)
    # plt.savefig('linear_reg_exp.png', transparent = True)
    pad_ads = y_pred.ravel()-targets
    return pad_ads


def linear_reg_parsing():
    """
    linear_reg function parsed that runs when the .py file is called.
    It performs a  linear regression with k-fold cross-validation
    predicting the age of patients from magnetic resonance imaging and
    prints evaluation metrics of the linear regression model 
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared.
    There are two ways to pass the csv file to this function. It's possible to
    pass the absolutepath of the dataset or you can store the dataset in a brother folder
    of the one containing code, and pass to the parsing
    function the filename and his container-folder.
    The parameters listed below are not parameters of the functions but
    are parsing arguments that have 
    to be passed to command line when executing the program as follow:

    .. code::

        $Your_PC>python linear_reg.py file.csv --target --location --folds --ex_cols --plot 

    where file.csv is the only mandatory argument,
    while others are optional and takes some default values,
    that if they have to be modified you can write for example:

    .. code::

        $Your_PC>python linear_reg.py file.csv --folds 10  

    :param filename: path to the CSV file containing
    the dataset or the name of the file if --location argument is passed 
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
        'Linear regression predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename",
                         help="Name of the file that has to be analized if --location argument is"
                        " passed. Otherwise pass to filename the absolutepath of the file")
    parser.add_argument("--target", default = "AGE_AT_SCAN",
                        help="Name of the column holding target values")
    parser.add_argument("--location",
                         help="Location of the file, i.e. folder containing it")
    parser.add_argument("--folds", type = int, default = 5,
                         help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type = int, default = 5,
                         help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--plot", action="store_true",
                         help="Show the plot of actual vs predicted brain age")
    parser.add_argument("--group", default = 'DX_GROUP',
                        help="Name of the column indicating the group (experimental vs control)")

    args = parser.parse_args()

    if args.folds > 4:
        try:
            args.filename = abs_path(args.filename,
                                    args.location) if args.location else args.filename
            logger.info(f"Opening file : {args.filename}")
            features, targets, group = get_data(args.filename,
                                                args.target,
                                                args.ex_cols,
                                                group_col = args.group)
            features_control = group_selection(features, group, -1)
            targets_control = group_selection(targets, group, -1)
            features_experimental = group_selection(features, group, 1)
            targets_experimental = group_selection(targets, group, 1)
            model, _, _, pad_control = linear_reg(features_control, targets_control, args.folds)
            pad_ads = lin_ads_prediction(features_experimental, targets_experimental, model)
            p_value_emp(pad_control, pad_ads)
            if args.plot:
                plt.show()
            else:
                logger.info('Skipping plots')
        except FileNotFoundError:
            logger.error("File not found.")
    else:
        logger.error("Invalid number of folds: at least 5 folds required.")


if __name__ == "__main__":
    linear_reg_parsing()
