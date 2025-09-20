import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from minisom import MiniSom
from kneed import KneeLocator

# Load dataset
def load_data(file_path="Mall_Customers_2.csv"):
    df = pd.read_csv(file_path)
    df = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    return df

# k-Means clustering
def kmeans_clustering(data, n_clusters=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    data_kmeans = data.copy()
    data_kmeans['Cluster'] = labels

    score = silhouette_score(X_scaled, labels)
    return data_kmeans, kmeans, score

# SOM clustering
def som_clustering(data, x=5, y=5, sigma=1.0, learning_rate=0.5, iters=1000):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    som = MiniSom(x, y, X_scaled.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.train_random(X_scaled, iters)

    winner_coordinates = np.array([som.winner(x) for x in X_scaled])
    cluster_index = np.ravel_multi_index(winner_coordinates.T, (x, y))

    data_som = data.copy()
    data_som['Cluster'] = cluster_index

    if len(set(cluster_index)) > 1:
        score = silhouette_score(X_scaled, cluster_index)
    else:
        score = -1

    return data_som, som, score

# Elbow Method
def elbow_method(data, k_range=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    inertias = []
    K = range(2, k_range + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    kl = KneeLocator(K, inertias, curve="convex", direction="decreasing")
    optimal_k = kl.elbow

    return list(K), inertias, optimal_k
