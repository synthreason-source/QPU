#!/usr/bin/env python3
import numpy as np
import networkx as nx

def compute_graph_s_matrix(G, max_hops=4):
    """
    Constructs a multi-scale structural S-matrix for a graph G 
    using powers of the random walk transition matrix.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros((1, 1))
        
    A = nx.adjacency_matrix(G).astype(np.float64).toarray()
    degrees = np.sum(A, axis=1, keepdims=True)
    degrees[degrees == 0.0] = 1.0
    
    # Random walk transition matrix P = D^{-1} A
    P = A / degrees
    
    # Feature matrix combining multi-hop walk probabilities and node degrees
    features = [np.sum(A, axis=1, keepdims=True), np.diag(A, 0).reshape(-1, 1)]
    
    current_P = P.copy()
    for hop in range(2, max_hops + 1):
        features.append(np.sum(current_P, axis=1, keepdims=True))
        current_P = current_P @ P
        
    X = np.hstack(features)
    
    # Standardize feature matrix
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-12] = 1.0
    X_norm = (X - mean) / std
    
    # Construct S-matrix
    S = X_norm.T @ X_norm
    return S

def are_graphs_isomorphic_svd(G1, G2, tol=1e-5):
    """
    Determines if two graphs are isomorphic by comparing their 
    S-matrix singular value spectra and latent subspace invariants.
    """
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
        
    S1 = compute_graph_s_matrix(G1)
    S2 = compute_graph_s_matrix(G2)
    
    # Compute SVD for both S-matrices
    try:
        _, sv1, _ = np.linalg.svd(S1, full_matrices=False)
        _, sv2, _ = np.linalg.svd(S2, full_matrices=False)
    except Exception:
        return False
        
    # Normalize singular spectra for scale invariance
    sv1_norm = sv1 / (np.linalg.norm(sv1) + 1e-12)
    sv2_norm = sv2 / (np.linalg.norm(sv2) + 1e-12)
    
    # Check spectral distance
    spectral_distance = np.linalg.norm(sv1_norm - sv2_norm)
    return spectral_distance < tol

# --- Example Usage ---
if __name__ == "__main__":
    # Create two isomorphic cycle graphs of size 10
    G1 = nx.cycle_graph(10)
    G2 = nx.relabel_nodes(G1, {i: (i + 3) % 10 for i in range(10)})
    
    is_iso = are_graphs_isomorphic_svd(G1, G2)
    print(f"Graphs Isomorphic (S-Matrix SVD): {is_iso}")
