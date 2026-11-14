import numpy as np


class KMeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None

    def _init_centers(self, X: np.ndarray) -> np.ndarray:
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return np.array(X[indices], dtype=float, copy=True)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not 1 <= self.n_clusters <= len(X):
            raise ValueError("X must be 2D and 1 <= n_clusters <= n_samples.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")

        self.cluster_centers_ = self._init_centers(X)
        previous_labels = None

        for _ in range(self.max_iter):
            distances = np.sum(
                (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
                axis=2,
            )
            labels = np.argmin(distances, axis=1)
            counts = np.bincount(labels, minlength=self.n_clusters)
            for cluster in np.flatnonzero(counts == 0):
                eligible = np.flatnonzero(counts[labels] > 1)
                errors = distances[eligible, labels[eligible]]
                point = eligible[np.argmax(errors)]
                counts[labels[point]] -= 1
                labels[point] = cluster
                counts[cluster] += 1
                self.cluster_centers_[cluster] = X[point]

            self.cluster_centers_ = np.array([
                X[labels == cluster].mean(axis=0)
                for cluster in range(self.n_clusters)
            ])
            self.labels_ = labels

            if previous_labels is not None and np.array_equal(labels, previous_labels):
                break
            previous_labels = labels.copy()

        return self.cluster_centers_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each query to its nearest center, breaking ties by index."""
        X = np.asarray(X, dtype=float)
        distances = np.sum(
            (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances, axis=1)
