import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Extraction of data as pandas dataframe
data = pd.read_csv(r'../data/abide.csv', delimiter=';')
data['SITE'] = data['FILE_ID'].str.rsplit('_', n = 1).str[0]

age_by_site = {}
for site, age_group in data.groupby('SITE')['AGE_AT_SCAN']:
    age_by_site[site] = age_group.values

# median age for each site and ordering the site arrays by median age
median_ages = {site: np.median(age_group) for site, age_group in age_by_site.items()}
sorted_sites = sorted(age_by_site.keys(), key=lambda x: median_ages[x])

fig, ax = plt.subplots(figsize=(12, 8))

# box plot
boxprops = dict(linewidth=1.5, color='black')
medianprops = dict(color='black', linewidth=1.5)
bp = ax.boxplot([age_by_site[site] for site in sorted_sites],
                patch_artist=True,
                boxprops=boxprops, medianprops=medianprops, )

for patch in bp['boxes']:
    patch.set_facecolor(mcolors.to_rgba('royalblue', alpha=0.7))

ax.set_xticklabels(sorted_sites, fontsize = 14, rotation = 45)
ax.set_xlabel('Site', fontsize = 24)
ax.set_ylabel('Age [years]', fontsize = 24)
ax.grid(False)

ax_hist = ax.twiny() # for the histogram
ax_hist.set_xticks([])

# plotting the hist
all_ages = data['AGE_AT_SCAN'].values
ax_hist.hist(all_ages, bins=30, orientation='horizontal',
             alpha=0.3, color='gray', edgecolor='none')
ax_hist.set_ylabel('Age [years]')
ax_hist.grid(False)

plt.tight_layout()
plt.savefig('histogram_age.png', transparent = True)
plt.show()