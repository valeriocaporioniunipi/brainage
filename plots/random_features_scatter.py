import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'../data/abide.csv', delimiter=';')
ages = data['AGE_AT_SCAN'].values
# dx_group = data['DX_GROUP'].values
# control, experimental = np.count_nonzero(dx_group == -1), np.count_nonzero(dx_group == 1)
# print(control, experimental)
# print(np.mean(ages))
# print(np.quantile(ages, 0.95))
# print('max and min age : ', np.min(ages), np.max(ages))
# selects 4 random features excluding patient info like age itself or FIQ
random_features = data.iloc[:, 5:].sample(n=4, axis=1, random_state=12)

# sets up four subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# plots each random feature vs brain age, with distinction based on DX_GROUP = \pm 1
for i, col in enumerate(random_features.columns):
    ax = axes[i // 2, i % 2]
    ax.scatter(ages, data[col], c=data['DX_GROUP'].map({1: 'red', -1: 'blue'}), alpha=0.5)
    ax.set_xlabel('Age [y]')
    ax.set_ylabel(col)
    ax.set_title(col)
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('random_features.png', transparent = True)
plt.show()