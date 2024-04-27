# -*- coding: utf-8 -*-
"""
CPS844 - A2

Clustering Analysis

Ben Costas 501025364
"""
import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from scipy.cluster.hierarchy import linkage, dendrogram

import warnings
warnings.filterwarnings('ignore')

# Load the dataset with correct delimiter and encoding type
data = pd.read_csv("2023-2024 NBA Player Stats.csv", encoding='ISO-8859-1', delimiter=';')

# Remove duplicate players (only keep their TOT games)
data = data[data.duplicated('Player', keep=False) & data['Tm'].eq('TOT') | ~data.duplicated('Player', keep='first')]

# Only keep first position
data['Pos'] = data['Pos'].apply(lambda x: x.split('-')[0])

# Replace all 0s with 0.01 (for processing)
data = data.replace(0, 0.01)

# Keep players who played more than 20 games (assume players who played less games had little to no impact)
data = data[data['G'] >= 30]

# Combine Steals and BLocks to "STOCKS" --> measures defensive impact
data['STOCKS'] = data['STL'] + data['BLK']

# Combine Turnovers and Personal Fouls to "MISTAKES" --> measures the amount of mistakes the player makes per game 
data['MISTAKES'] = data['TOV'] + data['PF']

# Drop unused columns
data = data.drop(columns=['Rk', 'Tm', 'G', 'GS', 'FG', 'FT', 'FTA', 'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF', '3P', '2P', 'FG%', '3PA', '2PA'])

# Numerize the Pos
position = {'PG':1, 'SG':2, 'SF':3, 'PF':4,'C':5}
data['Pos'] = data['Pos'].replace(position)


# Now that all the data is preprocessed, we can perform clustering analysis

# First lets find the optimal K-value by using elbow method

# Let's perform the analysis on a random sample size of the initial data, this is to compensate for runtime

sse = []
sample_data = data.sample(frac=0.1, random_state=42)
player_data = sample_data['Player']

sample_data = sample_data.drop(columns=['Player'])
max_k = 10

for k in range(1, max_k + 1):  # Trying k values from 1 to 10
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10).fit(sample_data)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), sse, marker='o')  # Corrected plotting call
plt.title('Elbow Method for Determining the Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.xticks(range(1, max_k + 1))
plt.show()

k = 3

kmeans_optimized = cluster.KMeans(n_clusters=k, random_state=42).fit(sample_data)
labels = kmeans_optimized.labels_

clusters = pd.DataFrame(labels, index=player_data, columns=['Cluster ID'])
sorted_clusters = clusters.sort_values(by='Cluster ID').to_string()

with open("cluster_output.txt", "w") as file:
    file.write(sorted_clusters)

# Now that we found the optimal k value and each player is assigned to a cluster,
# We can represent this information as a dendrogram

Z = linkage(sample_data, method='average')

# Plotting the dendrogram
plt.figure(figsize=(10, 12))
dendrogram(Z, labels=player_data.values, leaf_font_size=10, orientation='right')
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Players")
plt.ylabel("Distance")
plt.show()
