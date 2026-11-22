from __future__ import annotations
from typing import Optional
import numpy as np


class FluxEquilibriumClustering:
    """Flux Equilibrium Clustering (FEC) with improved defaults.

    Parameters
    ----------
    k : int, default=10
        Number of nearest neighbours when building the k-NN graph.
        For better clustering, consider k in range [8, 15].
    sigma : Optional[float], default=None
        Gaussian kernel bandwidth for the initial edge weights. If ``None``,
        infer from data. Recommended: use a fraction of median pairwise distance
        (e.g., 0.2 * median_distance) for more discriminative weights.
    alpha : float, default=0.5
        Fraction of stored flux that can be redistributed at each iteration.
        Higher values (0.5-0.8) promote more aggressive clustering.
    T : int, default=25
        Number of transport iterations. Consider 30-50 for better convergence.
    epsilon : float, default=1e-4
        Threshold on the *outgoing* flux below which a node is considered a sink.
        Lower values (1e-4 to 1e-6) help create fewer, larger clusters.
    beta : Optional[float], default=None
        Controls the non-linear reinforcement when re-weighting edges.

        * ``beta is None``  –  use ``phi(f) = log(1+f)`` (recommended).
        * ``beta = 0``      –  recovers the *original* FEC behaviour.
        * any other value   –  use ``phi(f) = f**beta``.

    Notes
    -----
    Implementation hints for better clustering:
    - Handle small datasets: ensure k <= n-1 to avoid boundary errors
    - Consider relaxed downhill flow: allow flux to similar-distance neighbors
    - Add cycle detection in path-following to prevent infinite loops
    - Post-process: merge very small clusters with nearest larger ones
    """

    def __init__(
        self,
        k: int = 10,
        sigma: Optional[float] = None,
        alpha: float = 0.5,
        T: int = 25,
        epsilon: float = 1e-4,
        beta: Optional[float] = None,
    ) -> None:

        self.k = k
        self.sigma = sigma
        self.alpha = alpha
        self.T = T
        self.epsilon = epsilon
        self.beta = beta

    def fit(self, X: np.ndarray) -> "FluxEquilibriumClustering":
        """Compute cluster labels for *X*.

        Implementation steps:
        1. Compute global centroid and distances to centroid
        2. Build symmetric k-NN graph with Gaussian edge weights
           - Handle small datasets: use effective_k = min(k, n-1)
           - If sigma is None, infer from data (e.g., fraction of median distance)
        3. Initialize flux: f_i = 1 for all nodes
        4. Run T flux-transport iterations:
           - Apply flux-aware reweighting (beta parameter)
           - Use relaxed downhill criterion (optional improvement)
           - Update flux: f_i(t+1) = f_i(t) - outgoing + incoming
        5. Identify sinks: nodes with outgoing flux < epsilon
        6. Assign clusters via steepest downhill paths
           - Add cycle detection to prevent infinite loops
           - Consider post-processing to merge small clusters

        The final labels must be stored in ``self.labels``.
        """
        # Step 1: Compute centroid and pairwise distances
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        centroid = np.mean(X, axis=0)
        distances = np.linalg.norm(X - centroid, axis=1)

        pairwise_distances = np.linalg.norm(
            X[:, None, :] - X[None, :, :],
            axis=2
        )

        # Calculate sigma
        if self.sigma is None:
            upper = pairwise_distances[np.triu_indices(n, k=1)]
            sigma = 0.2 * np.median(upper) if len(upper) else 1.0
        else:
            sigma = self.sigma

        sigma = max(float(sigma), 1e-10)
        effective_k = min(self.k, n - 1)

        # Find nearest neighbors
        np.fill_diagonal(pairwise_distances, np.inf)
        neighbors = np.argsort(pairwise_distances, axis=1)[:, :effective_k]

        neighbor_distances = np.take_along_axis(
            pairwise_distances, neighbors, axis=1
        )

        # Gaussian edge weights
        w_ij = np.exp(-(neighbor_distances ** 2) / (sigma ** 2))

        # Step 2: Initialize flux
        flux = np.ones(n)

        # Step 3: Simulate flux transport
        tau = 0.1 * sigma
        delta = np.zeros((n, effective_k))

        for t in range(self.T):
            delta = np.zeros((n, effective_k))

            for i in range(n):
                nbrs = neighbors[i]

                # Determine downhill neighbors
                downhill = distances[nbrs] <= distances[i] + tau

                if not np.any(downhill):
                    continue

                weights = w_ij[i].copy()

                # Flux-aware reinforcement
                if t > 0:
                    if self.beta is None:
                        phi = np.log1p(flux[nbrs])
                        weights *= phi
                    elif self.beta != 0:
                        phi = flux[nbrs] ** self.beta
                        weights *= phi

                weights[~downhill] = 0

                total_weight = np.sum(weights)

                if total_weight > 0:
                    delta[i] = (
                            self.alpha * flux[i] * weights / total_weight
                    )

            # Update flux simultaneously
            new_flux = flux.copy()

            for i in range(n):
                new_flux[i] -= np.sum(delta[i])

                for j, neighbor in enumerate(neighbors[i]):
                    new_flux[neighbor] += delta[i, j]

            flux = new_flux

        # Step 4: Detect sinks
        outgoing = np.sum(delta, axis=1)
        sinks = outgoing < self.epsilon

        if not np.any(sinks):
            sinks[np.argmin(distances)] = True

        # Step 5: Assign clusters by following steepest paths
        labels = np.full(n, -1, dtype=int)

        sink_indices = np.where(sinks)[0]

        for label, idx in enumerate(sink_indices):
            labels[idx] = label

        next_label = len(sink_indices)

        # Find steepest neighbor for each point
        steepest = np.full(n, -1, dtype=int)

        for i in range(n):
            if not sinks[i] and np.any(delta[i] > 0):
                j = np.argmax(delta[i])
                steepest[i] = neighbors[i, j]

        # Follow paths with cycle detection
        for i in range(n):
            if labels[i] != -1:
                continue

            path = []
            visited = set()
            current = i

            while labels[current] == -1:
                if current in visited:
                    # Cycle detected: create a new sink
                    sinks[current] = True
                    labels[current] = next_label
                    next_label += 1
                    break

                visited.add(current)
                path.append(current)

                next_node = steepest[current]

                if next_node == -1:
                    sinks[current] = True
                    labels[current] = next_label
                    next_label += 1
                    break

                current = next_node

            # Assign path nodes to the terminal cluster
            for node in path:
                labels[node] = labels[current]

        # Step 6: Merge small clusters
        min_size = max(2, n // 20)

        unique_labels, counts = np.unique(labels, return_counts=True)

        large_clusters = [
            label for label, count in zip(unique_labels, counts)
            if count >= min_size
        ]

        small_clusters = [
            label for label, count in zip(unique_labels, counts)
            if count < min_size
        ]

        for small_label in small_clusters:
            if not large_clusters:
                break

            small_points = X[labels == small_label]
            small_centroid = np.mean(small_points, axis=0)

            best_label = None
            best_distance = np.inf

            for large_label in large_clusters:
                large_points = X[labels == large_label]
                large_centroid = np.mean(large_points, axis=0)

                dist = np.linalg.norm(
                    small_centroid - large_centroid
                )

                if dist < best_distance:
                    best_distance = dist
                    best_label = large_label

            if best_label is not None:
                labels[labels == small_label] = best_label

        # ---- DO NOT CHANGE THE REST OF THIS METHOD ---- #
        # We require you maintain these attributes for testing purpose.
        self.w_ij = w_ij
        self.flux = flux
        self.sinks = sinks
        self.labels = labels

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Convenience method that returns the cluster labels."""
        self.fit(X)
        return self.labels

    def __repr__(self) -> str:
        params = (
            f"k={getattr(self, 'k', '?')}",
            f"sigma={getattr(self, 'sigma', '?')}",
            f"alpha={getattr(self, 'alpha', '?')}",
            f"T={getattr(self, 'T', '?')}",
            f"epsilon={getattr(self, 'epsilon', '?')}",
            f"beta={getattr(self, 'beta', '?')}",
        )
        return f"{self.__class__.__name__}({', '.join(params)})"
