
import numpy as np
import os
import argparse
from loguru import logger
from matplotlib import pyplot as plt
from keras import Sequential
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from abspath import AbsolutePath
from csvreader import GetData

def NeuralNetwork(filename, epochs, summary_flag=False, hist_flag=False, plot_flag=False):
    """
    NeuralNetwork creates a neural network. Inputs data are splitted in two parts: 'train' and
    'test'; both inputs are normalized in order to have zero as mean and one as variance.

    Arguments:
    -filename (str): path to the CSV file
    -epochs (int): optional, dafault = 50. Number of iterations on whole dataset
    -summary_flag (bool): optional, default = False. Print the structure of neural network
    -hist_flag (bool): optional, default = False. Plot a graph showing
     val_loss(labeled as valuation) vs loss(labeled as training) during epochs
    -plot_flag (bool): optional, default = False. Show the plot of actual vs predic

    Return:
    None. In the simpliest form just print the MAE (mean absolute error)

    """

    x = GetData(filename)
    y = GetData(filename, "AGE_AT_SCAN")

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Normalizing features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print(np.shape(x_train_scaled))
 
    # Defining the model
    model = Sequential()
    model.add(layers.Input(shape = np.shape(x_train_scaled[0])))
    # [0] is needed in order to pass the shape of a single feature array (the first, for instance)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Printing the summary, if specified
    if summary_flag:
        model.summary()

    # Training the model
    history = model.fit(x_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    # Showing the history plot
    if hist_flag:
        plt.plot(history.history["val_loss"], label="validation")
        plt.plot(history.history["loss"], label="training")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title('History')
        plt.legend()
        plt.show()

    # Predicting on the test set
    y_pred = model.predict(x_test_scaled)

    # Evaluating the model
    mse, mae = model.evaluate(x_test_scaled, y_test)

    if plot_flag == True:
        # plot the actual vs. predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted Brain age')
        plt.grid(False)
        plt.show()


    print("Mean Absolute Error on Test Set:", mae)
    print("Mean Squared Error on Test Set:", mse)



def main():
    parser = argparse.ArgumentParser(description='Neural network predicting the age of patients from magnetic resonance imaging')

    parser.add_argument("filename", help="Name of the file that has to be analized")
    parser.add_argument("--location", help="Location of the file, i.e. folder containing it")
    parser.add_argument("--epochs", type = int, default = 50, help="Number of epochs of training")
    parser.add_argument("--summary", action="store_true", help="Show the summary of the neural network")
    parser.add_argument("--history", action="store_true", help="Show the history of the training")
    parser.add_argument("--plot", action="store_true", help="Show the plot of actual vs predicted brain age")
    

    args = parser.parse_args()

    try:
        args.filename = AbsolutePath(args.filename, args.location) if args.location else args.filename
        logger.info("Opening file:", args.filename)
        NeuralNetwork(args.filename, args.epochs, args.summary, args.history, args.plot)
    except FileNotFoundError:
        logger.error("File not found.")
        return None

if __name__ == "__main__":
    main()
