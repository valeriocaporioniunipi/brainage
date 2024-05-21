# Required Libraries
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras import layers

# Importing custom functions
from utils import csv_reader, get_data

# Load and preprocess the data
filename = '/Users/valeriocaporioni/Documents/cmepda/brainage/data/FS_features_ABIDE_males.csv'
target_column = 'AGE_AT_SCAN'
site_column = 'FILE_ID'
group_column = 'DX_GROUP'
data, targets, group = get_data(filename, target_column, ex_cols=3, group_name=group_column)

data = data.astype(np.float32)

data = data[:150]
targets = targets[:150]
group = group[:150]

# Get the stripped column
site_names = pd.Series(csv_reader(filename, site_column))
str_list = site_names.apply(lambda x: x.split('_')[0])

str_list = str_list[:150]

# Extract unique sites
sites = str_list.unique()


# Function to create a binary classifier model
def create_class_nn(input_shape,
                    num_hidden_layers=1,
                    num_hidden_layer_nodes=32,
                    optimizer='adam',
                    metrics=['accuracy'],
                    summary_flag=False
                    ):
    model = Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for _ in range(num_hidden_layers - 1):
        model.add(layers.Dense(num_hidden_layer_nodes, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    if summary_flag:
        model.summary()
    return model


# Prepare AUC matrix
auc_matrix = np.zeros((len(sites), len(sites)))

# Training and evaluating models for each site pair
for i, site_i in enumerate(sites):
    for j, site_j in enumerate(sites):
        if i >= j:  # Only compute for upper triangle and diagonal
            continue
        # Get data for site_i and site_j
        site_i_indices = str_list == site_i
        site_j_indices = str_list == site_j
        indices = site_i_indices | site_j_indices

        site_data = data[indices]
        site_targets = targets[indices]
        # Binary labels: 1 for site_i, 0 for site_j
        site_labels = site_i_indices[indices].astype(int)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(site_data,
                                                            site_labels,
                                                            test_size=0.3,
                                                            random_state=42)

        # Create and train model
        logger.info(f"Model regarding {sites[i]} vs {sites[j]} classification ")
        model = create_class_nn(input_shape=X_train.shape[1])
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Predict and compute AUC
        y_pred = model.predict(X_test).ravel()
        auc = roc_auc_score(y_test, y_pred)

        # Fill the AUC matrix
        auc_matrix[i, j] = auc
        auc_matrix[j, i] = auc

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(auc_matrix, xticklabels=sites,
            yticklabels=sites,
            annot=True,
            fmt=".2f",
            cmap='coolwarm')

plt.title('AUC Scores Heatmap')
plt.show()
