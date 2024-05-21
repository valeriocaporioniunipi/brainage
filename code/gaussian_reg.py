import argparse
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from utils import abs_path, get_data


def gaussian_reg(features, targets, n_splits, **kwargs):
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
    # Definition of keyword arguments
    plot_flag = kwargs.get('plot_flag', False)
    group = kwargs.get('group', None)

    # Initialize data standardization (done after the k-folding split to avoid leakage)
    scaler = StandardScaler()

    # Initialize k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize lists to store evaluation metrics
    mae_scores, mse_scores, r2_scores = [], [], []

    if plot_flag:
        if group is not None:
            fig, (ax, ax_group) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax = plt.subplots(figsize=(10, 8))

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Split data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        if group is not None:
            group_test = group[test_index]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

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
        if plot_flag:
            ax.scatter(y_test, y_pred, alpha=0.5,
                       label=f'Fold {i} - MAE = {np.round(mae_scores[i - 1], 2)}')
            if group is not None:
                y_test_exp = y_test[group_test == 1]
                y_pred_exp = y_pred[group_test == 1]
                y_test_control = y_test[group_test == -1]
                y_pred_control = y_pred[group_test == -1]
                ax_group.scatter(y_test_exp, y_pred_exp, color='r', alpha=0.5)
                ax_group.scatter(y_test_control, y_pred_control, color='royalblue', alpha=0.5)

    # Print average evaluation metrics over all folds
    print("Mean Absolute Error:", np.mean(mae_scores))
    print("Mean Squared Error:", np.mean(mse_scores))
    print("R-squared:", np.mean(r2_scores))

    if plot_flag:
        target_range = [targets.min(), targets.max()]
        # Plot the ideal line (y=x)
        ax.plot(target_range, target_range, 'k--', lw=2)

        # Set plot labels and title
        ax.set_xlabel('Actual age [y]', fontsize=20)
        ax.set_ylabel('Predicted age [y]', fontsize=20)
        ax.set_title('Actual vs. predicted age', fontsize=24)

        # Add legend and grid to the plot
        ax.legend(fontsize=16)
        ax.grid(False)
        if group is not None:
            ax_group.plot(target_range, target_range, 'k--', lw=2)
            ax_group.set_xlabel('Actual age [y]', fontsize=20)
            ax_group.set_ylabel('Predicted age [y]', fontsize=20)
            ax_group.set_title('Actual vs. predicted age - exp. vs. control', fontsize=24)
            ax_group.grid(False)
            exp_legend = ax_group.scatter([], [], marker='o', color='r', label='exp.', alpha=0.5)
            control_legend = ax_group.scatter([], [], marker='o', color='royalblue', label='control', alpha=0.5)
            ax_group.legend(handles=[exp_legend, control_legend], loc='lower right', fontsize=16)
        # Show the plot
        # plt.savefig('/Users/valeriocaporioni/Downloads/gaussian_reg.png', transparent = True)
        plt.show()
    else:
        logger.info("Skipping the plot of actual vs predicted age ")


def gaussian_reg_parsing():
    """
    gaussian_reg function parsed that runs when the .py file is called.
    It performs a gaussian regression with k-fold cross-validation
    predicting the age of patients from magnetic resonance imaging and
    prints evaluation metrics of the linear regression model 
    such as MAE (mean absolute error), MSE (mean squared error) and R-squared.
    There are two ways to pass the csv file to this function. It's possible to
    pass the absolute path of the dataset, or you can store the dataset in a brother folder
    of the one containing code, and pass to the parsing function the filename and his container-folder.
    The parameters listed below are not parameters of the functions but are parsing arguments that have 
    to be passed to command line when executing the program as follows:

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
                                     'Linear regression predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename",
                        help="Name of the file that has to be analyzed if --location argument is"
                             " passed. Otherwise pass to filename the absolute path of the file")
    parser.add_argument("--target", default="AGE_AT_SCAN",
                        help="Name of the column holding target values")
    parser.add_argument("--location",
                        help="Location of the file, i.e. folder containing it")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type=int, default=5,
                        help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--plot", action="store_true",
                        help="Show the plot of actual vs predicted brain age")
    parser.add_argument("--group", default='DX_GROUP',
                        help="Name of the column indicating the group (experimental vs control)")

    args = parser.parse_args()

    if args.folds > 4:
        try:
            args.filename = abs_path(args.filename, args.location) if args.location else args.filename
            logger.info(f"Opening file : {args.filename}")
            features, targets, group = get_data(args.filename, args.target, args.ex_cols, group_col=args.group)
            gaussian_reg(features, targets, args.folds, plot_flag=args.plot, group=group)
        except FileNotFoundError:
            logger.error("File not found.")
    else:
        logger.error("Invalid number of folds: at least 5 folds required.")


if __name__ == "__main__":
    gaussian_reg_parsing()
