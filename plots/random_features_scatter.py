import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r'../data/FS_features_ABIDE_males.csv', delimiter=';')
# selects 4 random features excluding patient info like age itself or FIQ
random_features = data.iloc[:, 5:].sample(n=4, axis=1)

# sets up four subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# plots each random feature vs brain age, with distinction based on DX_GROUP = \pm 1
for i, col in enumerate(random_features.columns):
    ax = axes[i // 2, i % 2]
    ax.scatter(data['AGE_AT_SCAN'], data[col], c=data['DX_GROUP'].map({1: 'red', -1: 'blue'}), alpha=0.5)
    ax.set_xlabel('Age [y]')
    ax.set_ylabel(col)
    ax.set_title(col)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('random_features.png', transparent = True)
plt.show()