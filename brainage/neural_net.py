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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasRegressor

from utils import abs_path, get_data, group_selection, get_sites, sites_barplot, new_prediction
from utils import p_value_emp

def create_nn(input_shape,
                  hidden_layers = 1,
                  hidden_nodes = 48,
                  optimizer = 'rmsprop',
                  dropout = 0.05,
                  summary = False):
    """
    Create a neural network model using Keras API in order to solve a regression problem.

    :param input_shape: shape of the data given to the input layer of the NN
    :type input_shape: tuple
    :param hidden_layers: optional(default = 1): number of hidden layers in the network
    :type hidden_layers: int
    :param hidden_nodes: optional(default = 48) number of nodes in each hidden layer
    :type hidden_nodes: int
    :param optimizer: optional(default = 'rmsprop') optimizer to use
    :type optimizer: str
    :param dropout: optional (default = 0.05): dropout rate of dropout layers
    :type dropout: float
    :param summary: optional (default = False): show the summary of the model
    :type summary: bool
    :return: neural network model
    :rtype: Sequential
    """

    model = Sequential() # Defining the model
    model.add(layers.Input(shape=input_shape)) # Placing an input layer
    model.add(layers.Dropout(dropout)) # Placing dropout layre
    model.add(layers.BatchNormalization()) # BatchNormalization layer

    # Adding variable number of hidden layers (Dense+Dropout+BatchNormalization)
    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_nodes, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='linear'))  # Output layer of a regression problem

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    # Printing summary, if specified
    if summary:
        logger.info("Model successfully compiled, showing detailed summary ")
        model.summary()
    else:
        logger.info(f"Model successfully compiled with {hidden_layers} hidden layers")
    return model

def training(features, targets, model, epochs, **kwargs):
    """
    Train a neural network using k-fold cross-validation. 
    The function can show actual vs predicted brain age scatter plot and training history plot.

    :param features: matrix of features
    :type features: ndarray
    :param targets: array of target values
    :type targets: array
    :param model: neural network model
    :type model: sequential
    :param epochs: number of epochs for training
    :type epochs: int
    :param kwargs: additional keyword arguments
        - n_splits (int, optional, default to 5): number of folds for cross-validation

    :return: array holding mean absolute error, mean squared error, and R-squared scores
    :rtype: ndarray
    :return: list holding the n-splits models obtained after training
    :rtype: list
    """

    # Optional kwargs definition
    n_splits = kwargs.get('n_splits', 5)

    # Standardization of features
    scaler = StandardScaler()
    # since k-folding is implemented, standardization occurs after data splitting
    # in order to avoid information leakage (information from the validation or test set
    # would inadvertently influence the preprocessing steps).

    # Initialization of k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle = True, random_state=101)

    # Initialization of lists to store evaluation metrics
    mae_scores,r2_scores = [], []
    pad_control = []

    # Initializing figures for plotting and creating an array of colous
    figh, axh = plt.subplots(figsize=(10,8))

    figp, axp = plt.subplots(figsize=(10, 8))

    colormap = cmaps.get_cmap('tab20')
    colors = [colormap(i) for i in range(n_splits + 1)]

    # Storing the initial weights in order to refresh them after every fold training
    initial_weights = model.get_weights()

    best_model = None # Initialization of the variable associated with best model (least mae)
    mae_best = float('inf') 

    # Performing k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(features), 1):
        # Splitting data into training and testing sets
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        # Standandization (performed after the split)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

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

        #Appending vectors with history data
        validation_loss = history.history['val_loss']
        training_loss = history.history['loss']
        axh.plot(training_loss, label=f"Tr. {i}", color = colors[i])
        axh.plot(validation_loss, label=f"Val. {i}", color = colors[i], ls = 'dashed')

        # Evaluating the model
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        r2_scores.append(r2)
        pad_control.extend(y_pred.ravel()-y_test)

        if mae < mae_best:
            mae_best = mae
            best_model = model

        # Plotting actual vs. predicted values for current fold
        axp.scatter(y_test, y_pred,
                    alpha=0.5,
                    color = colors[i],
                    label=f'Fold {i} - MAE = {np.round(mae_scores[i-1], 2)}')

    axh.set_xlabel("epoch")
    axh.set_ylabel("loss [log]")
    axh.set_title(f'History losses in {epochs} epochs')
    axh.set_yscale('log')
    axh.legend()

    mae, r2 = np.mean(mae_scores), np.mean(r2_scores)
    # Printing average evaluation metrics over all folds
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)

    target_range = [targets.min(), targets.max()]
    # Plotting the ideal line (y=x)
    axp.plot(target_range, target_range, 'k--', lw=2)

    # Setting plot labels and title
    axp.set_xlabel('Actual age [y]', fontsize = 20)
    axp.set_ylabel('Predicted age [y]', fontsize = 20)
    axp.set_title(f'Actual vs. predicted age - control', fontsize = 24)

    # Adding legend and grid to the plots
    axp.legend(loc = 'upper left', fontsize = 16)
    axp.grid(False)
    return best_model, mae, r2, pad_control

def neural_net_parsing():
    """  
    The neural_net_parsing function is designed to parse command-line arguments and execute a neural
    network training process based on the provided parameters. It reads a dataset from a file,
    preprocesses the data, creates and trains a neural network model using k-fold cross-validation,
    and optionally performs a grid search for hyperparameter optimization. As output the function prints
    the model structure   

    The parameters listed below are not parameters of the functions, but are parsing arguments
    to be used in terminal, when executing the program as follows:

    .. code::

        $Your_PC>python neural_net.py file.csv --folds 7 --ex_cols 4 --plot 

    In order to read a description of every argument, execute:

    .. code::

        $Your_PC>python neural_net.py --help


    :param filename: name of the file that has to be analized
    :type filename: str
    :param target: optional (default = AGE_AT_SCAN): name of the colums holding target values
    :type target: str
    :param location: optional: location of the file, i.e. folder containing it
    :type location: str
    :param hidden_layers: optional (default = 1): number of hidden layers in the neural network
    :type hidden_layers: int
    :param hidden_nodes: optional (default = 32): number of hidden layer nodes in the neural network
    :type hidden_nodes: int
    :param epochs: optional (default = 300): number of epochs of training
    :type epochs: int
    :param opt: optional(default = "rmsprop"): optimizer 
    :type opt: str
    :param folds: optional (>4, default = 5): number of folds in the k-folding
    :type folds: int
    :param dropout: optional (default = 0.05): dropout rate in neural network
    :type dropout: int
    :param ex_cols: optional (default = 5): number of columns excluded when importing
    :type ex_cols: int
    :param summary: optional(default = False): show the summary of the neural network
    :type summary: bool
    :param plot: optional(default = False): show the plot of training history, actual vs 
        predicted brain (TD and ASD)
    :type plot: bool
    :param group: optional(default = 'DX_GROUP'): name of the column indicating the 
        group (experimental vs control)")
    :type group: str    
    :param overs: optional(default = False): oversampling, done in order to have
        a flat distribution of targets")
    :type overs: bool
    :param harm: optional: name of the column of sites, used for data harmonization
    :type harm: str             
    :param grid: optional(default = False): grid search for hyperparameter optimization
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
    parser.add_argument("--epochs", type = int, default = 300,
                         help="Number of epochs of training (default 50)")
    parser.add_argument("--opt", default= "rmsprop",
                         help="Optimizer (default = 'rmsprop')")
    parser.add_argument("--folds", type = int, default = 5,
                         help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--dropout", type = float, default = 0.05,
                         help="Dropout rate in the NN (default 0.05)")
    parser.add_argument("--ex_cols", type = int, default = 5,
                         help="Number of columns excluded when importing (default 3)")
    parser.add_argument("--summary", action="store_true",
                         help="Show the summary of the neural network")
    parser.add_argument("--plot", action="store_true",
                         help="Show the plot of training history and actual vs predicted brain age")
    parser.add_argument("--group", default = 'DX_GROUP',
                        help="Name of the column indicating the group (experimental vs control) (default DX_Group)")
    parser.add_argument("--overs", action = 'store_true', default = False,
                        help="Oversampling, done in order to have"
                        "a flat distribution of targets (default = False).")
    parser.add_argument("--harm",
                        help="Name of the column of sites, used for data harmonization")
    parser.add_argument("--grid", action = "store_true",
                        help="Grid search for hyperparameter optimization")
    parser.add_argument("--site_col", default = 'FILE_ID',
                        help = "Column with information about acquisition site")

    args = parser.parse_args()

    try:
        if args.location:
            args.filename = abs_path(args.filename, args.location)
        logger.info(f"Opening file : {args.filename}")
        features, targets, group = get_data(args.filename,
                                            args.target,
                                            args.ex_cols,
                                            group_col = args.group, 
                                            site_col = args.harm,
                                            overs = args.overs)
        features_control = group_selection(features, group, -1)
        targets_control = group_selection(targets, group, -1)
        features_exp = group_selection(features, group, 1)
        targets_exp = group_selection(targets, group, 1)
        epochs = args.epochs
        input_shape = np.shape(features[0])
        if not args.grid:
            model = create_nn(input_shape,
                                        hidden_layers = args.hidden_layers,
                                        hidden_nodes = args.hidden_nodes,
                                        dropout = args.dropout,
                                        optimizer = args.opt,
                                        summary = args.summary)
            model, _, _, pad_control = training(features_control,
                        targets_control,
                        model,
                        epochs,
                        n_splits = args.folds)
        else:# args.grid
            param_grid = {
            'model__hidden_layers': [1, 2, 4],
            'model__hidden_nodes' : [32, 48],
            'model__optimizer': ['adam', 'adagrad', 'rmsprop'],
            'model__dropout': [0.0, 0.01, 0.05]
            }

            keras_regressor = KerasRegressor(model=lambda hidden_layers,
                                            hidden_nodes, dropout, optimizer:
                                            create_nn( input_shape,
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
            x_scaled = scaler.fit_transform(features_control)

            # Fitting grid search
            logger.info("Starting Grid Search for hyperparameter optimization")
            grid_result = grid.fit(x_scaled, targets_control)

            # Summarizing results
            logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                logger.info(f"{mean} ({std}) with: {param}")
            model = create_nn(input_shape,
                                        hidden_layers =
                                        grid_result.best_params_["model__hidden_layers"],
                                        hidden_nodes
                                        = grid_result.best_params_["model__hidden_nodes"],
                                        optimizer =
                                        grid_result.best_params_["model__optimizer"],
                                        dropout = 
                                        grid_result.best_params_['model__dropout'],
                                        summary = args.summary)
            model, _, _, pad_control = training(features_control,
                        targets_control,
                        model,
                        epochs,
                        n_splits = args.folds,
                        plot_flag = args.plot)
        pad_asd, _, _ = new_prediction(features_exp, targets_exp, model)
        p_value_emp(pad_control, pad_asd)
        sites_asd = get_sites(args.filename, site_col = args.site_col, group_col = args.group,
                          group_value = 1) # 1 is for ASD group
        print(np.mean(pad_control), np.mean(pad_asd))
        sites_barplot(pad_asd, sites_asd)
        if args.plot:
            plt.show()
        else:
            logger.info("Skipping the plot of training history and of prediction result")
    except FileNotFoundError:
        logger.error("File not found.")


if __name__ == "__main__":
    neural_net_parsing()
