import pandas as pd
import numpy as np

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

# Compute inertia for Elbow Method
def compute_elbow_inertia(X, k_min=1, k_max=10, random_state=42, n_init=10):
    X_vals = X.to_numpy()

    k_values = list(range(k_min, k_max + 1))
    inertias = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        model.fit(X_vals)
        inertias.append(model.inertia_)

    return k_values, inertias

# K-Means Clustering
def run_kmeans(X, k=4, random_state=42, n_init=10):
    X_vals = X.to_numpy()
    model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = model.fit_predict(X_vals)

    return labels

# Agglomerative Clustering
def run_agglomerative(X, k=4):
    X_vals = X.to_numpy()
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_vals)

    return labels

# Compute Linkage Matrix for Dendrogram Plot
def compute_linkage_matrix(X, method="ward"):
    X_vals = X.to_numpy()

    return linkage(X_vals, method=method)
