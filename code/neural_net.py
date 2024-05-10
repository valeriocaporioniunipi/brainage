import numpy as np
import argparse
from loguru import logger
from matplotlib import pyplot as plt

from keras import Sequential
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from abspath import AbsolutePath
from csvreader import GetData

def NeuralNetwork(filename, epochs, n_splits, ex_cols = 0, summary_flag=False, hist_flag=False, plot_flag=False):

    """
    NeuralNetwork creates a neural network. Inputs data are splitted in two parts: 'train' and
    'test'; both inputs are normalized in order to have zero as mean and one as variance.

    Arguments:
    -filename (str): path to the CSV file
    -epochs (int): number of epochs during neural network training
    -n_splits (int): number of folds for cross-validation
    -ex_cols (int): optional, default = 0. Number of columns excluded from dataset
    -summary_flag (bool): optional, default = False. Print the structure of neural network
    -hist_flag (bool): optional, default = False. Plot a graph showing
     val_loss(labeled as valuation) vs loss(labeled as training) during epochs
    -plot_flag (bool): optional, default = False. Show the plot of actual vs predic

    Return:
    None. In the simpliest form just print the MAE (mean absolute error), the MSE (mean squared error) and the R-squared
    """
    # Load data...
    #Importing features excluded first three columns: FILE_ID, AGE_AT_SCAN, SEX
    x = GetData(filename)[:, ex_cols:]
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


    # Defining the model
    model = Sequential()
    model.add(layers.Input(shape = np.shape(x[0])))
    # Defining the model outside is better from a computational-resources point of view. The shape of
    # x[0] is the same of x_train[0], which will be defined later
    # [0] is needed in order to pass the shape of a single feature array (the first, for instance)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    logger.info("Model successfully compiled.")

    # Printing the summary, if specified
    if summary_flag:
        model.summary()
    else: 
        logger.info("Skipping model summary.")

    # Initialize figure for plotting
    if plot_flag:
        plt.figure(figsize=(10, 8))

    # Perform k-fold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(x_scaled), 1):
        # Split data into training and testing sets
        x_train, x_test = x_scaled[train_index], x_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Training the model
        logger.info(f"Training the model with dataset {i}/{n_splits}")
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=32)
        
        # Showing the history plot
        if hist_flag:
            plt.plot(history.history["val_loss"], label="validation")
            plt.plot(history.history["loss"], label="training")
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f'History of split {i}')
            plt.legend()
            plt.show()

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
            plt.scatter(y_test, y_pred, alpha=0.5, label=f'Fold {i} - MAE = {np.round(mae_scores[i-1], 2)}')

    # Print average evaluation metrics over all folds
    print("Mean Absolute Error:", np.mean(mae_scores))
    print("Mean Squared Error:", np.mean(mse_scores))
    print("R-squared:", np.mean(r2_scores))

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
    else: 
        logger.info("Skipping the plot of Actual vs Predicted Brain Age.")

def main():
    parser = argparse.ArgumentParser(description='Neural network predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename", help="Name of the file that has to be analized")
    parser.add_argument("--location", help="Location of the file, i.e. folder containing it")
    parser.add_argument("--epochs", type = int, default = 50, help="Number of epochs of training")
    parser.add_argument("--folds", type = int, default = 5, help="Number of folds in the k-folding (>4, default 5)")
    parser.add_argument("--ex_cols", type = int, default = 3, help="Number of columns excluded when importing data")
    parser.add_argument("--summary", action="store_true", help="Show the summary of the neural network")
    parser.add_argument("--history", action="store_true", help="Show the history of the training")
    parser.add_argument("--plot", action="store_true", help="Show the plot of actual vs predicted brain age")
    
    args = parser.parse_args()

    if args.folds > 4:
        try:
            args.filename = AbsolutePath(args.filename, args.location) if args.location else args.filename
            logger.info(f"Opening file : {args.filename}")
            NeuralNetwork(args.filename, args.epochs, args.folds, args.ex_cols, args.summary, args.history, args.plot)
        except FileNotFoundError:
            logger.error("File not found.")
            return None
    else: 
        logger.error("Invalid number of folds: at least 5 folds required.")


if __name__ == "__main__":
    main()