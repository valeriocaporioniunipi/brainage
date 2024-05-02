import csv
import numpy as np
import pandas as pd
from loguru import logger
from keras import Sequential
from keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def ShowCSV(csv_file, column_name=None):
    try:
        df = pd.read_csv(csv_file, delimiter=';')
        if column_name is None:
            print(df)
            return df.values
        else:
            print(df[column_name])
            return df[column_name].values
    except FileNotFoundError:
        logger.error("File not found.")
        return None

dataset = r"C:\Users\Jacopo\the_cartella\Magistrale\CMFEP\DATASETS\FEATURES\Brain_MRI_FS_ABIDE\FS_features_ABIDE_males_someGlobals.csv"

X = np.column_stack((ShowCSV(dataset,"TotalGrayVol"), ShowCSV(dataset, "SEX")))
y = np.array(ShowCSV(dataset,"AGE_AT_SCAN"))

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

#cosa fa con git