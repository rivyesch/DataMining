# Importing Python libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc

########################################################################################################################
# importing raw data and assigning column header names
features = ["Min Temp", "Max Temp", "Mean Temp", "Min RH", "Max RH", "Mean RH", "Min Pressure", "Max Pressure",
            "Mean Pressure", "Precipitation", "Snowfall Amount", "Sunshine Duration", "Min Wind Gust", "Max Wind Gust",
            "Mean Wind Gust", "Min Wind Speed", "Max Wind Speed", "Mean Wind Speed"]
df = pd.read_csv("ClimateDataBasel.csv", header=None, names=features)

# Getting the summary of the data
df.describe().T
print("Size of raw dataframe: ", df.shape)

########################################################################################################################
# Pre-processing Stage

# Count missing values
df.isnull().sum().sort_values(ascending=False)

# Since there are no missing values, no further action is taken

# Normalisation of the data
names = df.columns
indexes = df.index
sc1 = MinMaxScaler((0, 1))
df1 = sc1.fit_transform(df)
df_normalised = pd.DataFrame(df1, columns=names, index=indexes)
print(df_normalised.var())

# Removing features with extremely low variance
# It is assumed that features with a higher variance may contain more useful information
variance = VarianceThreshold(threshold=0.01)
variance.fit(df_normalised)
mask = variance.get_support()
# Removing the features Snowfall Amount and precipitation which has the lowest variances from the un-normalised
# dataframe
df = df.loc[:, mask]

# Removing outliers based on z-score (3 std dev)
# Based on visual inspection of the boxplot the following features contain anomalies:
# Max/Mean RH, Min/Max/Mean Pressure, Precipitation, Snowfall Amount, Min/Max/Mean Wind Gust, Min/Max/Mean Wind Speed

# Calculating the z-score for all elements in the dataframe
z = np.abs(stats.zscore(df))

# Finding the index of the rows which contains a z-score that is more or less than 3 standard deviations of the
# features' mean
for var in df.columns:
    index = z[(z[var] > 3) | (z[var] < -3)].index
    z.drop(index, inplace=True)
    df.drop(index, inplace=True)

# Getting the summary of the data
print("Size of dataframe after removing outliers and dropping low variance features: ", df.shape)
# Dropped/Removed a total of 113 rows of instances (out of 1763 instances) due to outliers

# Standardisation of the Data
sc2 = StandardScaler()
df2 = sc2.fit_transform(df)
df_standardised = pd.DataFrame(df2, columns=df.columns, index=df.index)

# Creating a Pearson's Correlation heatmap to illustrate the correlation between the features in the dataset
# plt.figure("Figure 1")
plt.figure(figsize=(12, 12))
cor = df_standardised.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig('Climate Correlation Heatmap.png', bbox_inches = 'tight')

df_standardised = df_standardised.drop(["Min Pressure", "Min Temp", "Max Pressure", "Max Temp", "Min RH",
                                        "Max Wind Speed", "Max Wind Gust"], axis=1)

########################################################################################################################
# Feature Extraction - PCA

# Plotting dependence of explained variance on number of PCA components
plt.figure()
pca = PCA().fit(df_standardised)
# print(pca.explained_variance_ratio_)
# print(np.cumsum(pca.explained_variance_ratio_))
plt.plot(range(1, 10), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('Explained Variance Ratio.png', dpi=600)

# estimate only 3 PCs - gives an explained variance of 79%
num_comp = 3
pca = PCA(n_components=num_comp)

# project the original data into the PCA space
principalComponents = pca.fit_transform(df_standardised)
# print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_)*100)
PCA_components = pd.DataFrame(principalComponents)

# Plot the first two PCA components
plt.figure("Figure 2")
plt.scatter(PCA_components[1], PCA_components[0], alpha=.1, color='black')
plt.xlabel('PCA 2')
plt.ylabel('PCA 1')

# Checking the importance of the original features on the PCA components
PCA_importance = pd.DataFrame(pca.components_, columns=list(df_standardised.columns))
pd.set_option('display.max_columns', None)
# print(PCA_importance)

# For PC1 the features that are most influential/important are Mean Wind Speed, Mean Wind Gust and Max Wind Speed
# For PC2 the features that are most influential/important are Mean Pressure, Max Pressure and Min Pressure

########################################################################################################################
# Modeling the first clustering algorithm K-Means

# Elbow method to determine optimal number of clusters
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(PCA_components)
                for k in range(1, 10)]

inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure("Figure 3")
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
# plt.title('The Elbow Diagram')
plt.savefig('Elbow Method.png', dpi=600)


# Silhouette Score method to determine optimal number of clusters
silhouette_scores = [silhouette_score(PCA_components, model.labels_)
                     for model in kmeans_per_k[1:]]

plt.figure("Figure 4")
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.xlabel("$k$")
plt.ylabel("Silhouette Score")
# plt.title('The Silhouette score')
plt.savefig('Silhouette Score.png', dpi=600)

# Figure 3 and Figure 4 both shows that 3 clusters is the optimal number of clusters. The second best choice would be 4
# clusters.

k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(PCA_components)
labels_kmeans = kmeans.labels_

df_kmeans = pd.concat([df_standardised.reset_index(drop=True), PCA_components], axis=1)

df_kmeans.columns.values[-3:] = ["Component 1", "Component 2", "Component 3"]
# df_segm_pca_kmeans.columns.values[-4:] = ["Component 1", "Component 2", "Component 3", "Component 4"]

df_kmeans["Label"] = labels_kmeans

df_kmeans["Segment"] = df_kmeans["Label"].map({0: "First", 1: "Second", 2: "Third"})
colours = ["g", "r", "c"]
# df_segm_pca_kmeans["Segment"] = df_segm_pca_kmeans["Label"].map({0: "First", 1: "Second", 2: "Third", 3: "Fourth"})
# colours = ["g", "r", "c", "m"]

# Plot data by PCA components. The x-axis is the second component (PCA 2) and the y-axis is the first component (PCA 1)
x_axis = df_kmeans["Component 2"]
y_axis = df_kmeans["Component 1"]
plt.figure("Figure 5")
plt.xlabel("PCA 2")
plt.ylabel("PCA 1")
# sns.scatterplot(x=x_axis, y=y_axis, hue=df_kmeans["Segment"], palette=colours)
plt.scatter(df_kmeans["Component 2"], df_kmeans["Component 1"], c=labels_kmeans, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.title("K-means Clustering")
plt.savefig('K-means Clustering.png', dpi=600)

########################################################################################################################
# Modeling the second clustering algorithm - BIRCH
plt.figure("Figure 6")
# plt.title("Dendrograms")
plt.xlabel("Customers")  # or sample index
plt.ylabel("Euclidean Distance")
dend = shc.dendrogram(shc.linkage(PCA_components, method='ward'))
plt.axhline(y=75, color='m', linestyle='--')
plt.savefig('Dendrogram.png', dpi=600)


# brc = Birch(branching_factor=50, n_clusters=3, threshold=0.7)
brc = Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=2, threshold=0.005)

brc.fit(PCA_components)
labels_birch = brc.predict(PCA_components)

df_birch = pd.concat([df_standardised.reset_index(drop=True), PCA_components], axis=1)
df_birch.columns.values[-3:] = ["Component 1", "Component 2", "Component 3"]
df_birch["Label"] = labels_birch
plt.figure("Figure 7")
plt.xlabel("PCA 2")
plt.ylabel("PCA 1")
plt.scatter(df_birch["Component 2"], df_birch["Component 1"], c=labels_birch, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.title("Birch Clustering")
plt.savefig('BIRCH Clustering.png', dpi=600)
plt.show()


