"""
Module neural_net trains neural networks in order to guess age from brain features
"""
import argparse

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import colormaps as cmaps
from keras import Sequential
from keras import layers
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from utils import abs_path, get_data, oversampling

def create_reg_nn(input_shape,
                  hidden_layers = 1,
                  hidden_nodes = 48,
                  optimizer = 'rmsprop',
                  dropout = 0.05,
                  summary_flag = False):
    """
    Create a neural network model using Keras in order to solve a regression problem.

    :param input_shape: Shape of the data given to the input layer of the NN
    :type input_shape: tuple
    :param hidden_layers: Number of hidden layers in the network
    :type hidden_layers: int
    :param hidden_nodes: Number of nodes in each hidden layer
    :type hidden_nodes: int
    :param optimizer: Optimizer to use
    :type optimizer: str
    :param dropout: Dropout rate of dropout layer
    :type dropout: float
    :param summary_flag: Show the summary of the model
    :type summary_flag: bool
    :return: Neural Network model
    :rtype: Sequential
    """

    model = Sequential() # Defining the model
    model.add(layers.Input(shape=input_shape)) # Placing an input layer
    model.add(layers.Dropout(dropout))
    model.add(layers.BatchNormalization())

    # Adding variable number of hidden layers
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_nodes, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='linear'))  # Output layer of a regression problem

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    # Printing summary, if specified
    if summary_flag:
        logger.info("Model successfully compiled, showing detailed summary ")
        model.summary()
    else:
        logger.info(f"Model successfully compiled with {hidden_layers} hidden layers")
    return model

def training(features, targets, model, epochs, **kwargs):
    """"
    Train a neural network using k-fold cross-validation.
    The function can show actual vs predicted brain age scatter plot and training history plot.

    :param features: Matrix of features
    :type features: ndarray
    :param targets: Array of target values
    :type targets: array
    :param model: Neural network model
    :type model: Sequential
    :param epochs: Number of epochs for training
    :type epochs: int
    :param kwargs: Additional keyword arguments
        - n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
        - group (array, optional): Group information for stratified sampling. Defaults to None.
        - bins (int, optional): Number of bins for oversampling. Defaults to 10.
        - hist_flag (bool, optional): Plot training history. Defaults to False.
        - plot_flag (bool, optional): Plot actual vs predicted values. Defaults to False.
        - overs_flag (bool, optional): Perform oversampling. Defaults to False.

    :return: Array holding mean absolute error, mean squared error, and R-squared scores
    :rtype: ndarray
    :return: List holding the n-splits models obtained after training
    :rtype: list
    """

    # Optional kwargs
    n_splits = kwargs.get('n_splits', 5)
    group = kwargs.get('group', None)
    bins = kwargs.get('bins', 10)
    hist_flag = kwargs.get('hist_flag', False)
    plot_flag = kwargs.get('plot_flag', False)
    overs_flag = kwargs.get('overs_flag', False)

    # Standardization of features
    scaler = StandardScaler()
    # since k-folding is implemented, standardization occurs after data splitting
    # in order to avoid information leakage (information from the validation or test set
    # would inadvertently influence the preprocessing steps).

    # Initialization of k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle = True)

    # Initialization of lists to store evaluation metrics
    mae_scores, mse_scores, r2_scores = [], [], []

    # Initializing figures for plotting and creating rlated colours
    if hist_flag:
        figh, axh = plt.subplots(figsize=(10,8))

    if plot_flag:
        if group is not None:
            figp, (axp, axp_group) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            figp, axp = plt.subplots(figsize=(10, 8))

    colormap = cmaps.get_cmap('tab20')
    colors = [colormap(i) for i in range(n_splits + 1)]

    # Storing the initial weights in order to refresh them after every fold training
    initial_weights = model.get_weights()

    # Initializing the list that will hold the models once trained
    models =[]

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Splitting data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        if group is not None:
            group_test = group[test_index]

        # Standandization (after the split)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Oversampling
        if overs_flag:
            logger.info(f'Performing oversampling with {bins} bins')
            x_train, y_train, _ = oversampling(x_train, y_train, bins = bins)

        # Training the model (after having properly re-initialized the weights)
        model.set_weights(initial_weights)
        logger.info(f"Training the model with dataset {i}/{n_splits} for {epochs} epochs ")
        history = model.fit(x_train, y_train, epochs=epochs,
                            batch_size=32,
                            validation_split=0.1,
                            verbose = 0)
        logger.info('Training successfully ended ')

        # Predict on the test set
        y_pred = model.predict(x_test)

        # Appending the model to models list
        models.append(model)

        #Appending vectors with history data
        if hist_flag:
            validation_loss = history.history['val_loss']
            training_loss = history.history['loss']
            axh.plot(training_loss, label=f"Tr. {i}", color = colors[i])
            axh.plot(validation_loss, label=f"Val. {i}", color = colors[i], ls = 'dashed')

        # Evaluating the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)

        # Plotting actual vs. predicted values for current fold
        if plot_flag:
            axp.scatter(y_test, y_pred,
                        alpha=0.5,
                        color = colors[i],
                        label=f'Fold {i} - MAE = {np.round(mae_scores[i-1], 2)}')
            if group is not None:
                y_test_exp = y_test[group_test == 1]
                y_pred_exp = y_pred[group_test == 1]
                y_test_control = y_test[group_test == -1]
                y_pred_control = y_pred[group_test == -1]
                axp_group.scatter(y_test_exp, y_pred_exp,color = 'k')
                axp_group.scatter(y_test_control, y_pred_control, color = 'r')

    if hist_flag:
        axh.set_xlabel("epoch")
        axh.set_ylabel("loss")
        axh.set_title(f'History losses in {epochs} epochs')
        axh.set_yscale('log')
        figh.legend()
    else:
        logger.info("Skipping the plot of training history ")

    # Printing average evaluation metrics over all folds
    print("Mean Absolute Error:", np.mean(mae_scores))
    print("Mean Squared Error:", np.mean(mse_scores))
    print("R-squared:", np.mean(r2_scores))

    scores = np.array([np.mean(mae_scores), np.mean(mse_scores), np.mean(r2_scores)])

    if plot_flag:
        target_range = [targets.min(), targets.max()]
        # Plotting the ideal line (y=x)
        axp.plot(target_range, target_range, 'k--', lw=2)

        # Setting plot labels and title
        axp.set_xlabel('Actual age [y]')
        axp.set_ylabel('Predicted age [y]')
        axp.set_title(f'Actual vs. predicted age - {n_splits} folds')

        # Adding legend and grid to the plots
        figp.legend(loc = 'upper left')
        axp.grid(True)
        if group is not None:
            axp_group.plot(target_range, target_range, 'k--', lw=2)
            axp_group.set_xlabel('Actual age [y]')
            axp_group.set_ylabel('Predicted age [y]')
            axp_group.set_title('Actual vs. predicted age - exp. vs. control')
            axp_group.grid(True)
            exp_legend = axp_group.scatter([], [], marker = 'o', color = 'k', label = 'exp.')
            control_legend = axp_group.scatter([], [], marker = 'o', color = 'r', label = 'control')
            figp.legend(handles = [exp_legend, control_legend], loc='upper right')
    else:
        logger.info("Skipping the plot of actual vs predicted age ")

    plt.show()
    return scores, models

def neural_net_parsing():
    """
    neural_net_parsing executes the parsing from terminal

    The parameters listed below are not parameters of the functions, but are parsing arguments
    to be used in terminal, when executing the program as follows:

    .. code::

        $Your_PC>python neural_net.py file.csv --folds 7 --ex_cols 4 --plot 

    In order to read a description of every argument, execute:

    .. code::

        $Your_PC>python neural_net.py --help


    :param filename: Name of the file that has to be analized
    :type filename: str
    :param target: optional (default = AGE_AT_SCAN): Name of the colums holding target values
    :type target: str
    :param location: optional: Location of the file, i.e. folder containing it
    :type location: str
    :param hidden_layers: optional (default = 1): Number of hidden layers in the neural network
    :type hidden_layers: int
    :param hidden_nodes: optional (default = 32): Number of hidden layer nodes in the neural network
    :type hidden_nodes: int
    :param epochs: optional (default = 50): Number of epochs of training
    :type epochs: int
    :param folds: optional (>4, default = 5): Number of folds in the k-folding
    :type folds: int
    :param ex_cols: optional (default = 3): Number of columns excluded when importing
    :type ex_cols: int
    :param summary: optional: Show the summary of the neural network
    :type summary: bool
    :param history: optional: Show the history of the training
    :type history: bool
    :param plot: optional: Show the plot of actual vs predicted brain age
    :type plot: bool
    :param grid: optional: Grid search for hyperparameter optimization
    :type grid: bool

    """
    parser = argparse.ArgumentParser(description=
        'Neural network predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename",
                         help="Name of the file that has to be analized")
    parser.add_argument("--target", default = "AGE_AT_SCAN",
                        help="Name of the column holding target values")
    parser.add_argument("--location",
                         help="Location of the file, i.e. folder containing it")
    parser.add_argument("--hidden_layers", type = int, default = 1,
                         help="Number of hidden layers in the neural network")
    parser.add_argument("--hidden_nodes", type = int, default = 32,
                         help="Number of hidden layer nodes in the neural network")
    parser.add_argument("--epochs", type = int, default = 50,
                         help="Number of epochs of training (default 50)")
    parser.add_argument("--opt", default= "rmsprop",
                         help="Optimizer (default = 'rmsprop')")
    parser.add_argument("--folds", type = int, default = 5,
                         help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type = int, default = 3,
                         help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--summary", action="store_true",
                         help="Show the summary of the neural network")
    parser.add_argument("--history", action="store_true",
                         help="Show the history of the training")
    parser.add_argument("--plot", action="store_true",
                         help="Show the plot of actual vs predicted brain age")
    parser.add_argument("--group", default = 'DX_GROUP',
                        help="Name of the column indicating the group (experimental vs control)")
    parser.add_argument("--overs", action = 'store_true', default = False,
                        help="Oversampling, done in order to have"
                        "a flat distribution of targets (default = True).")
    parser.add_argument("--bins", type = int, default = 10,
                        help="Number of bins in resampling (default 0 20)")
    parser.add_argument("--grid", action = "store_true",
                        help="Grid search for hyperparameter optimization")

    args = parser.parse_args()

    try:
        args.filename = abs_path(args.filename,
                                        args.location) if args.location else args.filename
        logger.info(f"Opening file : {args.filename}")
        features, targets, group = get_data(args.filename,
                                            args.target,
                                            args.ex_cols,
                                            group_name = args.group)
        epochs = args.epochs
        input_shape = np.shape(features[0])
        if not args.grid:
            model = create_reg_nn(input_shape,
                                        hidden_layers = args.hidden_layers,
                                        hidden_nodes = args.hidden_nodes,
                                        optimizer = args.opt,
                                        summary_flag = args.summary)
            training(features,
                        targets,
                        model,
                        epochs,
                        n_splits = args.folds,
                        bins = args.bins,
                        group = group,
                        overs_flag = args.overs,
                        hist_flag = args.history,
                        plot_flag = args.plot)
        else:# args.grid
            param_grid = {
            'model__hidden_layers': [1, 2, 4],
            'model__hidden_nodes' : [32, 48],
            'model__optimizer': ['adam', 'adagrad', 'rmsprop'],
            'model__dropout': [0.0, 0.01, 0.05]
            }

            keras_regressor = KerasRegressor(model=lambda hidden_layers,
                                            hidden_nodes, dropout, optimizer:
                                            create_reg_nn( input_shape,
                                            hidden_layers=hidden_layers,
                                            hidden_nodes=hidden_nodes,
                                            dropout=dropout,
                                            optimizer=optimizer),
                                            epochs=epochs,
                                            batch_size=32,
                                            verbose=0)
            grid = GridSearchCV(estimator=keras_regressor,
                                param_grid=param_grid,
                                scoring='neg_mean_absolute_error',
                                refit = False,
                                cv = args.folds)
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(features)

            # Fitting grid search
            logger.info("Starting Grid Search for hyperparameter optimization")
            grid_result = grid.fit(x_scaled, targets)

            # Summarizing results
            logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                logger.info(f"{mean} ({std}) with: {param}")
            model = create_reg_nn(input_shape,
                                        hidden_layers =
                                        grid_result.best_params_["model__hidden_layers"],
                                        hidden_nodes
                                        = grid_result.best_params_["model__hidden_nodes"],
                                        optimizer =
                                        grid_result.best_params_["model__optimizer"],
                                        dropout = 
                                        grid_result.best_params_['model__dropout'],
                                        summary_flag = args.summary)
            training(features,
                        targets,
                        model,
                        epochs,
                        n_splits = args.folds,
                        bins = args.bins,
                        group = group,
                        overs_flag = args.overs,
                        hist_flag = args.history,
                        plot_flag = args.plot)
    except FileNotFoundError:
        logger.error("File not found.")


if __name__ == "__main__":
    neural_net_parsing()
