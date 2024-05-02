import numpy as np
from csvreader import GetData
import argparse
from loguru import logger
from keras import Sequential
from keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def neural_network(filename, epochs = 50, summary_flag = False, hist_flag = False):
    """Documentation"""
    X = GetData(filename)
    y = GetData(filename,"AGE_AT_SCAN")

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizing features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Defining the model
    model = Sequential()
    model.add(layers.Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compiling the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Printing the summary, if specified
    if summary_flag == True:
        model.summary()

    # Training the model
    history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    # Showing the history plot
    if hist_flag == True:
        from matplotlib import pyplot as plt
        plt.plot(history.history["val_loss"], label = "validation")
        plt.plot(history.history["loss"], label = "training")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

    # Evaluating the model
    loss, mae = model.evaluate(X_test_scaled, y_test)
    print("Mean Absolute Error on Test Set:", mae)

def main():

    parser = argparse.ArgumentParser(description='This program does stuff')

    parser.add_argument("filename", help = "Name of the file that has to be analized")
    parser.add_argument("epochs", help = "Number of epochs of training")
    parser.add_argument("--summary", action="store_true", help = "Show the summary of the neural network")
    parser.add_argument("--history",action="store_true", help = "Show the history of the training")

    args = parser.parse_args()

    neural_network(args.filename, int(args.epochs), args.summary, args.history)


if __name__ == "__main__":

    main()