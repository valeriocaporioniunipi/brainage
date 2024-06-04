'''
Regression
'''
import argparse
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from utils import abs_path, get_data, p_value_emp, group_selection, new_prediction


@ignore_warnings(category=ConvergenceWarning)
def regression(type, features, targets, n_splits):

    """
    performs regression (using sklearn) with k-fold cross-validation with a
    specified number of splits on the given dataset and
    prints evaluation metrics of the linear regression model
    such as MAE (mean absolute error) and R-squared. Regressions models implemented are linear and
    gaussian. The function inizialize a plot that shows actual vs. predicted age for a control
    group of patients. 

    :param type: type of regression performed, only two arguments are possible
        'linear' or 'gaussian'  
    :type type: str
    :param features: features
    :type features: numpy.ndarray
    :param targets: array containing target feature
    :type targets: numpy.array 
    :param n_splits: number of folds for cross-validation
    :type n_splits: int

    :returns: A tuple containing:
    
        - **best_model** (*sequential*): the best model selected across k-folding.
        - **mae** (*float*): the mean absolute error mean across folds.
        - **r2** (*float*): the coefficient of determination mean across folds.
        - **pad_control** (*list*): the predicted actual difference.
    :rtype: tuple(sequential, float, float, list)

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

    _, axc = plt.subplots(figsize=(10, 8))

    # Perform k-fold cross-validation
    for _, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Split data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if type == "linear":
            # Initialize and fit linear regression model
            model = LinearRegression()
            model.fit(x_train, y_train)
        elif type == "gaussian":
            # Initialize and fit linear regression model
            kernel = C(1.0, (1, 1e2)) * Matern(length_scale=1.0, length_scale_bounds=(1, 1e2))
            model = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 5)
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


def reg_parsing():
    """
    regression function parsed that runs when the .py file is called.
    It performs regression (linear or gaussian) with k-fold cross-validation 
    predicting the age of patients from magnetic resonance imaging and
    prints evaluation metrics of the linear regression model 
    such as MAE (mean absolute error) and R-squared.
    There are two ways to pass the csv file to this function. It's possible to
    pass the absolutepath of the dataset or you can store the dataset in a brother folder
    of the one containing code, and pass to the parsing
    function the filename and his container-folder.
    The parameters listed below are not parameters of the functions but
    are parsing arguments that have 
    to be passed to command line when executing the program as follow:

    .. code::

        $Your_PC>python regression.py file.csv type --target --location --folds --ex_cols --plot 

    where file.csv and type are the only mandatory argument,
    while others are optional and takes some default values,
    that if they have to be modified you can write for example:

    .. code::

        $Your_PC>python linear_reg.py file.csv --folds 10  

    :param filename: path to the CSV file containing 
        the dataset or the name of the file if --location argument is passed 
    :type filename: str
    :param type: type of regression performed, only two arguments are possible
        'linear' or 'gaussian'  
    :type type: str
    :param target: optional (default = AGE_AT_SCAN): name of the column holding target values
    :type target: str
    :param location: optional: location of the file, i.e. folder containing it 
    :type location: str
    :param folds: optional (>4, default 5):number of folds for cross-validation
    :type folds: int
    :param ex_cols: optional (default = 5): columns excluded when importing file
    :type ex_cols: int
    :param overs: optional (default = False): if True oversampling on dataset is performed
    :type overs: bool
    :param plot: optional (default = False): show the plot of actual vs predicted brain age
    :type plot: bool
    :param group: optional (default = 'DX_GROUP'): name of the column indicating the group
        ('control' vs 'experimental')
    :return: None

    """

    parser = argparse.ArgumentParser(description=
                                     'Linear regression predicting the age of patients from'
                                       'magnetic resonance imaging')

    parser.add_argument("filename",
                         help="Name of the file that has to be analized if --location argument is"
                        " passed. Otherwise pass to filename the absolutepath of the file")
    parser.add_argument("type",
                         help="Type of regression model that could be implemented. Could be 'l'"
                         "for linear regression or 'g' for gaussian regression")
    parser.add_argument("--target", default = "AGE_AT_SCAN",
                        help="Name of the column holding target values")
    parser.add_argument("--location",
                         help="Location of the file, i.e. folder containing it")
    parser.add_argument("--folds", type = int, default = 5,
                         help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type = int, default = 5,
                         help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--overs", action = 'store_true', default = False,
                        help="Oversampling, done in order to have"
                        "a flat distribution of targets (default = False).")
    parser.add_argument("--plot", action="store_true",
                         help="Show the plot of actual vs predicted brain age")
    parser.add_argument("--group", default = 'DX_GROUP',
                        help="Name of the column indicating the group (experimental vs control)")

    args = parser.parse_args()

    if args.folds > 4:
        if args.type == "linear" or args.type == "gaussian":
            try:
                args.filename = abs_path(args.filename,
                                        args.location) if args.location else args.filename
                logger.info(f"Opening file : {args.filename}")
                features, targets, group = get_data(args.filename,
                                                    args.target,
                                                    args.ex_cols,
                                                    group_col = args.group,
                                                    overs = args.overs)
                features_control = group_selection(features, group, -1)
                targets_control = group_selection(targets, group, -1)
                features_experimental = group_selection(features, group, 1)
                targets_experimental = group_selection(targets, group, 1)
                model, _, _, pad_control = regression(args.type, features_control,
                                                    targets_control, args.folds)
                pad_asd, _, _ = new_prediction(features_experimental, targets_experimental, model)
                print(np.mean(pad_control), np.mean(pad_asd))
                p_value_emp(pad_control, pad_asd)
                if args.plot:
                    plt.show()
                else:
                    logger.info('Skipping plots')
            except FileNotFoundError:
                logger.error("File not found.")
        else:
            logger.error("Such regression model doesn't exist or it's not implemented. "
            "Please select a valid regression model")
    else:
        logger.error("Invalid number of folds: at least 5 folds required.")


if __name__ == "__main__":
    reg_parsing()
