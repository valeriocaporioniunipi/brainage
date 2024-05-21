import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data = pd.read_csv(r'/Users/valeriocaporioni/Documents/cmepda/brainage/data/FS_features_ABIDE_males.csv', delimiter=';')
data['SITE'] = data['FILE_ID'].str.split('_').str[0]

age_by_site = {}
for site, age_group in data.groupby('SITE')['AGE_AT_SCAN']:
    age_by_site[site] = age_group.values

# Median age for each site and order the site arrays by median age
median_ages = {site: np.median(age_group) for site, age_group in age_by_site.items()}
sorted_sites = sorted(age_by_site.keys(), key=lambda x: median_ages[x])

# Figure with a single subplot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the box plot
boxprops = dict(linewidth=1.5, color='black')
medianprops = dict(color='black', linewidth=1.5)
bp = ax.boxplot([age_by_site[site] for site in sorted_sites], patch_artist=True, boxprops=boxprops, medianprops=medianprops)

# Set all box colors to pastel blue
for patch in bp['boxes']:
    patch.set_facecolor(mcolors.to_rgba('royalblue', alpha=0.7))

ax.set_xticklabels(sorted_sites, fontsize = 14)
ax.set_xlabel('Site', fontsize = 20)
ax.set_ylabel('Age [years]', fontsize = 20)
ax.grid(False)

# Create a second x-axis for the histogram
ax_hist = ax.twiny()
ax_hist.set_xticks([])

# Plot the histogram
all_ages = data['AGE_AT_SCAN'].values
ax_hist.hist(all_ages, bins=30, orientation='horizontal', alpha=0.3, color='gray', edgecolor='none')
ax_hist.set_ylabel('Age [years]')
ax_hist.grid(False)

plt.tight_layout()
plt.show()
"""



# Select random features (columns with index > 3)
random_features = data.iloc[:, 4:].sample(n=4, axis=1)

# Set up the subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot each random feature against brain age with distinction based on DX_GROUP
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

"""


