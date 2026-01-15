from sklearn.cluster import KMeans

def run_kmeans(X, k=4):
    return KMeans(n_clusters=k, random_state=42).fit_predict(X)