import numpy as np
from csvreader import ShowCSV
from loguru import logger
from keras import Sequential
from keras import layers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


dataset = r"C:\Users\Jacopo\the_cartella\Magistrale\CMFEP\DATASETS\FEATURES\Brain_MRI_FS_ABIDE\FS_features_ABIDE_males_someGlobals.csv"

X = np.column_stack((ShowCSV(dataset)))
y = ShowCSV(dataset,"AGE_AT_SCAN")

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

# Printing a summary of the model
model.summary()

# Training the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluating the model
loss, mae = model.evaluate(X_test_scaled, y_test)
print("Mean Absolute Error on Test Set:", mae)
