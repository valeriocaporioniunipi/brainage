import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from brainage.utils import get_data

_, targets = get_data(r'../data/abide_globals.csv',
                         'AGE_AT_SCAN', 5, overs = False)
_, targets_os = get_data(r'../data/abide_globals.csv',
                         'AGE_AT_SCAN', 5, overs = True)

plt.figure(figsize=(10, 6))

plt.hist(targets, bins=30, color='blue',
         alpha=0.5, edgecolor='black', label='Original targets')

plt.hist(targets_os, bins=30, color='green',
         alpha=0.5, edgecolor='black', label='Oversampled targets')

plt.title('Oversampling on ages')
plt.xlabel('Age')
plt.ylabel('Occurrences')

plt.legend()

plt.show()


