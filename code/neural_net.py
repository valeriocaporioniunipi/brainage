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

from abspath import abs_path
from csvreader import get_data, oversampling

def create_neural_net(input_shape,
                        num_hidden_layers = 1,
                        num_hidden_layer_nodes = 32,
                        optimizer = 'adam',
                        metrics = ['mae'],
                        summary_flag = False):
    """
    create_neural_net creates an instance of the Sequential class of Keras,
    creating a Neural Network with variable hidden layers, each with 32 nodes,
    and setting the initial weights at random values.

    Arguments:
    - input_shape (tuple): shape of the data given to the input layer of the NN
    - num_hidden_layers (int). Number of hidden layers in the network
    - optimizer (str): optional, default = 'adam'. Optimizer to use
    - metrics (list): optional, default = ['mae']. List of metrics to use
    - summary_flag (bool): optional, default = False. Show the summary of the NN

    Return: the instance of the Sequential class, i.e. the model object
    """
    
    # Defining the model
    model = Sequential()
    model.add(layers.Input(shape=input_shape))

    # Adding variable number of hidden layers
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(num_hidden_layer_nodes, activation='relu'))

    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=metrics)

    # Printing the summary, if specified
    if summary_flag:
        logger.info("Model successfully compiled, showing detailed summary ")
        model.summary()
    else:
        logger.info(f"Model successfully compiled with {num_hidden_layers} hidden layers")
    return model

def build_model(input_shape,
                num_hidden_layers=1,
                num_hidden_layer_nodes = 32,
                optimizer='adam',
                **kwargs):
    """
    Wrapper function to create a Keras model with specified hyperparameters
    """
    return create_neural_net(input_shape, num_hidden_layers, optimizer)


def training(features, targets, model, epochs, **kwargs):
    """"
    training trains a neural network with k-folding

    Arguments:
    - features (ndarray): matrix of features
    - targets (ndarray): array of targets
    - model (SequentialType): NN model, instance of Sequential class
    - epochs (int): number of epochs during neural network training
    - **kwargs: additional keyword arguments for configuring the function behavior
        - n_splits (int): number of folds for cross-validation
        - hist_flag (bool): optional, default = False. Plot a graph showing val_loss
            (labeled as validation) vs loss (labeled as training) during epochs.
        - plot_flag (bool): optional, default = False.
            Show the plot of actual vs predicted brain age.

    Return:
    - scores (ndarray): array holding MAE, MSE and R-squared, averaged among the folds

    Printing:
    - MAE (mean absolute error)
    - MSE (mean squared error)
    - R-squared
    Optionally showing:
    
    - Actual vs Predicted brain age scatter plot
    - Training history plot

    :param filename: path to the CSV file containing the dataset 
    :type filename: str
    :param epochs: number of epochs during neural network training
    :type epochs: int
    :param n_splits: number of folds for cross-validation
    :type n_splits: int
    :param ex_cols: optional (default = 0): number of folds for cross-validation
    :type ex_cols: int
    :param summary_flag: optional (default = False): print the structure of neural network
    :type summary_flag: bool
    :param hist_flag: optional (default = False): plot a graph showing val_loss(labeled as valuation) vs loss(labeled as training) during epochs
    :type hist_flag: bool
    :param plot_flag: optional (default = False): show the plot of actual vs predic
    :type plot_flag: bool

    return: None

    """

    # Optional kwargs
    n_splits = kwargs.get('n_splits', 5)
    group = kwargs.get('group', None)
    hist_flag = kwargs.get('hist_flag', False)
    plot_flag = kwargs.get('plot_flag', False)
    # Renaming data
    x = features
    y = targets

    # Defining a boolean value if experimental and control groups are separated
    isgroup = np.any(group)

    # Standardization of features
    scaler = StandardScaler()
    # since k-folding is implemented, standardization occurs after data splitting
    # in order to avoid information leakage (information from the validation or test set
    # would inadvertently influence the preprocessing steps).

    # Initialization of k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle = True)

    # Initialization of lists to store evaluation metrics
    mae_scores = []
    mse_scores = []
    r2_scores = []

    # Initializing figures for plotting and creating rlated colours
    if hist_flag:
        figh, axh = plt.subplots(figsize=(10,8))

    if plot_flag:
        if isgroup:
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
    for i, (train_index, test_index) in enumerate(kf.split(x), 1):
        # Split data into training and testing sets
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if isgroup:
            group_test = group[test_index]

        # Standandization (after the split)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Training the model (after having re-initialized the weights)
        model.set_weights(initial_weights)
        logger.info(f"Training the model with dataset {i}/{n_splits} for {epochs} epochs ")
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

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
            axh.set_yscale('log')

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
            if isgroup:
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
        figh.legend()

    else:
        logger.info("Skipping the plot of training history ")

    # Printing average evaluation metrics over all folds
    print("Mean Absolute Error:", np.mean(mae_scores))
    print("Mean Squared Error:", np.mean(mse_scores))
    print("R-squared:", np.mean(r2_scores))

    scores = np.array([np.mean(mae_scores), np.mean(mse_scores), np.mean(r2_scores)])

    if plot_flag:
        # Plotting the ideal line (y=x)
        axp.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)

        # Setting plot labels and title
        axp.set_xlabel('Actual')
        axp.set_ylabel('Predicted')
        axp.set_title(f'Actual vs. predicted age - {n_splits} folds')

        # Adding legend and grid to the plots
        figp.legend(loc = 'upper left')
        axp.grid(True)
        if isgroup:
            axp_group.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            axp_group.set_xlabel('Actual')
            axp_group.set_ylabel('Predicted')
            axp_group.set_title(f'Actual vs. predicted age - exp. vs. control')
            axp_group.grid(True)
            exp_legend = axp_group.scatter([], [], marker = 'o', color = 'k', label = 'exp.')
            control_legend = axp_group.scatter([], [], marker = 'o', color = 'r', label = 'control')
            figp.legend(handles = [exp_legend, control_legend], loc='upper right')
    else:
        logger.info("Skipping the plot of actual vs predicted age ")

    plt.show()
    return scores

def neural_net_parsing():
    """
    Parsing from terminal
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
    parser.add_argument("--overs", action = 'store_true', default = True,
                        help="Oversampling, done in order to have a flat distribution of targets (default = True).")
    parser.add_argument("--bins", type = int, default = 10,
                        help="Number of bins in resampling (default 0 20)")
    parser.add_argument("--grid", action = "store_true",
                        help="Grid search for hyperparameter optimization")

    args = parser.parse_args()

    try:
        args.filename = abs_path(args.filename,
                                        args.location) if args.location else args.filename
        logger.info(f"Opening file : {args.filename}")
        features, targets, group = get_data(args.filename, args.target, args.ex_cols, group_name = args.group)
        if args.overs:
            features, targets, group = oversampling(features, targets, group=group)
        epochs = args.epochs
        input_shape = np.shape(features[0])
        if not args.grid:
            model = create_neural_net(input_shape,
                                        num_hidden_layers = args.hidden_layers,
                                        num_hidden_layer_nodes = args.hidden_nodes,
                                        summary_flag = args.summary)
            training(features,
                        targets,
                        model,
                        epochs,
                        n_splits = args.folds,
                        group = group,
                        hist_flag = args.history,
                        plot_flag = args.plot)
        else: # args.grid 
            param_grid = {
            'model__num_hidden_layers': [1, 4, 6],
            'model__num_hidden_nodes' : [32, 48],
            'model__optimizer': ['adam', 'sgd', 'rmsprop']
            }

            keras_regressor = KerasRegressor(model=lambda **kwargs: build_model(input_shape, **kwargs),
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

            # Summarize results
            logger.info(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, std, param in zip(means, stds, params):
                logger.info(f"{mean} ({std}) with: {param}")
            model = create_neural_net(input_shape,
                                        num_hidden_layers =
                                        grid_result.best_params_["model__num_hidden_layers"],
                                        num_hidden_layer_nodes 
                                        = grid_result.best_params_["model__num_hidden_nodes"],
                                        optimizer =
                                        grid_result.best_params_["model__optimizer"],
                                        summary_flag = args.summary)
            training(features,
                        targets,
                        model,
                        epochs,
                        n_splits = args.folds,
                        group = group,
                        hist_flag = args.history,
                        plot_flag = args.plot)
    except FileNotFoundError:
        logger.error("File not found.")


if __name__ == "__main__":
    neural_net_parsing()
