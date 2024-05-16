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
from csvreader import get_data

def create_neural_net(input_shape,
                        num_hidden_layers,
                        num_hidden_layer_nodes = 32,
                        optimizer='adam',
                        metrics=['mae'],
                        summary_flag=False):
    """
    create_neural_net creates an instance of the Sequential class of Keras,
    creating a Neural Network with variable hidden layers, each with 32 nodes,
    and setting the initial weights at random values.

    :param input_shape: shape of the data given to the input layer of the NN
    :type input_shape: tuple
    :param num_hidden_layers: number of hidden layers in the network
    :type num_hidden_layers: int
    :param optimizer: optional (default = 'adam'): Optimizer to use
    :type optimizer: str
    :param metrics: optional (default = ['mae']): List of metrics to use
    :type metrics: list
    :param summary_flag: optional (default = False): Show the summary of the NN
    :type summary_flag: Bool
    :return: Neural Network model
    :rntype: SequentialType
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
    Non so bene che scrivere qui
    """
    return create_neural_net(input_shape, num_hidden_layers, optimizer)


def training(features, targets, model, epochs, **kwargs):
    """"
    training function trains a neural network with k-folding. As input the function needs a matrix 
    containing features and a target array with the feature under exam. Is necessary a neural network model 
    and to specify number of epochs. Other arguments are possible and are listed below. The basic
    output is the printing of MAE, MSE, R. It can shows also Actual vs Predicted brain age scatter plot and 
    Training history plot

    :param features: matrix of features
    :type features: ndarray
    :param targets: array of targets
    :type targets: array
    :param model: NN model, instance of Sequential class
    :type model: SequentialType
    :param epochs: number of epochs during neural network training
    :type epochs: int
    :param n_splits: optional (default = 5): number of folds for cross-validation
    :type n_splits: int
    :param hist_flag: optional (default = False). Plot a graph showing val_loss
        (labeled as validation) vs loss (labeled as training) during epochs.
    :type hist_flag: bool
    :param plot_flag: optional (default = False). 
        Show the plot of actual vs predicted brain age.
    :type plot_flag: bool
    :return: a matrix whose columns are array holding MAE, MSE and R-squared, averaged among the folds
    :rntype: ndarray



    """

    # Optional kwargs
    n_splits = kwargs.get('n_splits', 5)
    hist_flag = kwargs.get('hist_flag', False)
    plot_flag = kwargs.get('plot_flag', False)
    # Renaming data
    x = features
    y = targets

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
            axp.scatter(y_test, y_pred, alpha=0.5, color = colors[i],
                         label=f'Fold {i} - MAE = {np.round(mae_scores[i-1], 2)}')

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
        axp.set_title('Actual vs. predicted age')

        # Adding legend and grid to the plot
        figp.legend()
        axp.grid(True)

    else:
        logger.info("Skipping the plot of actual vs predicted age ")

    plt.show()
    return scores

def neural_net_parsing():
    """
    Parsing from terminal

    The parameters listed below are not parameters of the functions but are parsing arguments that have 
    to be passed to command line when executing the program as follow:

    .. code::

        $Your_PC>python neural_net.py file.csv --folds 7 --ex_cols 4 --plot  


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
                        help="Name of the colums holding target values")
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
    parser.add_argument("--grid", action = "store_true",
                        help="Grid search for hyperparameter optimization")

    args = parser.parse_args()

    try:
        args.filename = abs_path(args.filename,
                                        args.location) if args.location else args.filename
        logger.info(f"Opening file : {args.filename}")
        features, targets = get_data(args.filename, args.target, args.ex_cols)
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
                                        num_hidden_layers = grid_result.best_params_["model__num_hidden_layers"],
                                        num_hidden_layer_nodes = grid_result.best_params_["model__num_hidden_nodes"],
                                        optimizer= grid_result.best_params_["model__optimizer"],
                                        summary_flag = args.summary)
            training(features,
                        targets,
                        model,
                        epochs,
                        n_splits = args.folds,
                        hist_flag = args.history,
                        plot_flag = args.plot)
    except FileNotFoundError:
        logger.error("File not found.")


if __name__ == "__main__":
    neural_net_parsing()
