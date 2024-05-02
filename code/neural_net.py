import numpy as np
from csvreader import ShowCSV
import argparse
from loguru import logger
from keras import Sequential
from keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def neural_network(filename):
    """Documentation"""
    X = np.column_stack((ShowCSV(filename,"TotalGrayVol"), ShowCSV(filename, "SEX")))
    y = np.array(ShowCSV(filename,"AGE_AT_SCAN"))

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    model = Sequential()
    model.add(layers.Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))  # Output layer

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss, mae = model.evaluate(X_test_scaled, y_test)
    print("Mean Absolute Error on Test Set:", mae)

def main():

    parser = argparse.ArgumentParser(description='This program does stuff')

    parser.add_argument("filename", help = "Name of the file that have to be analized")

    args = parser.parse_args()

    neural_network(args.filename)


if __name__ == "__main__":

    main()