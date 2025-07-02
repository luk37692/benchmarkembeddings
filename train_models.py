#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from utils import read_data, mask_edges
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from models import train_rle, train_geld, train_tadw, prepare_data
import networkx as nx
import pickle
import os

def train_all_embeddings(dataset="cora2", d=160, drop_rate=0.5, methods=None):
    """
    Train all embedding methods and save results to disk
    
    Args:
        dataset: Dataset name ("cora2", "dblp", "nyt")
        d: Embedding dimension
        drop_rate: Edge drop rate for masking
        methods: List of methods to train
    
    Returns:
        Dictionary of trained embeddings
    """
    if methods is None:
        methods = ["BASELINE_TFIDF", "RLE", "GELD", "SINE", "MUSAE", "TADW"]
    
    print(f"\nLoading {dataset}...")
    # Load data
    tf_idf, groups, A, graph, voc, raw, tf = read_data(dataset)
    A = mask_edges(A, drop_rate)
    n = A.shape[0]
    graph = nx.from_scipy_sparse_matrix(A)
    graph.remove_edges_from(nx.selfloop_edges(graph))  
    print(f"{n} nodes, {tf_idf.shape[1]} features, {len(graph.edges())} edges")

    # Prepare and learn embeddings
    data_graph, data_text, sigma_init, D_init, U = prepare_data(d, tf, A, voc, raw)
    embeddings = {}

    if "BASELINE_TFIDF" in methods:
        print("\n" + "="*50)
        print("BASELINE: TF-IDF (sans liens)")
        print("="*50)
        # Réduire les dimensions pour comparaison équitable
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=d, random_state=42)
        tfidf_reduced = svd.fit_transform(tf_idf)
        embeddings["BASELINE_TFIDF"] = normalize(tfidf_reduced, norm='l2', axis=1)
        print(f"Baseline TF-IDF embeddings shape: {embeddings['BASELINE_TFIDF'].shape}")

    if "GELD" in methods:
        print("\n" + "="*50)
        print("Training GELD...")
        print("="*50)
        alpha_map = {"cora2": 0.99, "dblp": 0.8, "nyt": 0.95}
        D, D_norm, sigma = train_geld(
            d, data_graph, data_text, U, D_init, sigma_init,
            n_epoch=40, lamb=None, alpha=alpha_map[dataset]
        )
        #embeddings["GELD"] = normalize(D_norm, norm='l2', axis=1)
        embeddings["GELD"] = D_norm
        print(f"GELD embeddings shape: {embeddings['GELD'].shape}")
        print(f"GELD embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['GELD'][0]):.4f}")

    if "RLE" in methods:
        print("\n" + "="*50)
        print("Training RLE...")
        print("="*50)
        window_map = {"cora2": 15, "dblp": 5, "nyt": 10}
        rle_embeddings, _ = train_rle(A, tf, U, d, 0.7, verbose=True)
        #embeddings["RLE"] = normalize(rle_embeddings, norm='l2', axis=1)
        embeddings["RLE"] = rle_embeddings
        print(f"RLE embeddings shape: {embeddings['RLE'].shape}")
        print(f"RLE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['RLE'][0]):.4f}")

    if "MUSAE" in methods:
        print("\n" + "="*50)
        print("Training MUSAE embeddings...")
        print("="*50)
        from karateclub import MUSAE

        # Convert TF-IDF features to the format expected by Karate Club
        if hasattr(tf_idf, 'tocoo'):
            feature_matrix = tf_idf.tocoo()
        else:
            from scipy.sparse import coo_matrix
            feature_matrix = coo_matrix(tf_idf)

        musae_model = MUSAE(
            dimensions=d,
            epochs=50,
            learning_rate=0.01,
            seed=42
        )
        musae_model.fit(graph, feature_matrix)
        musae_embeddings = musae_model.get_embedding()
        embeddings["MUSAE"] = normalize(musae_embeddings, norm='l2', axis=1)
        print(f"MUSAE embeddings shape: {embeddings['MUSAE'].shape}")
        print(f"MUSAE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['MUSAE'][0]):.4f}")

    if "TADW" in methods:
        print("\n" + "="*50)
        print("Training TADW embeddings...")
        print("="*50)
        
        # TADW hyperparameters based on dataset
        tadw_params = {
            "cora2": {"order": 2, "iter_max": 5, "lamb": 0.2},
            "dblp": {"order": 2, "iter_max": 20, "lamb": 0.1}, 
            "nyt": {"order": 2, "iter_max": 15, "lamb": 0.15}
        }
        
        params = tadw_params.get(dataset, {"order": 2, "iter_max": 15, "lamb": 0.2})
        
        tadw_embeddings, training_time = train_tadw(
            A, tf_idf, d=d, 
            order=params["order"],
            iter_max=params["iter_max"], 
            lamb=params["lamb"],
            verbose=True
        )
        
        embeddings["TADW"] = normalize(tadw_embeddings, norm='l2', axis=1)
        print(f"TADW embeddings shape: {embeddings['TADW'].shape}")
        print(f"TADW embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['TADW'][0]):.4f}")

    if "SINE" in methods:
        print("\n" + "="*50)
        print("Training SINE embeddings...")
        print("="*50)
        
        from karateclub import SINE
        
        if hasattr(tf_idf, 'tocoo'):
            feature_matrix = tf_idf.tocoo()
        else:
            from scipy.sparse import coo_matrix
            feature_matrix = coo_matrix(tf_idf)
        
        sine_model = SINE(
            walk_number   = 40,    # Original paper default
            walk_length   = 80,    # Compromise between paper (100) and efficiency
            dimensions    = d,     # Keep your 160 for fair comparison
            window_size   = 10,    # Keep as is (matches paper)
            epochs        = 50,    # Increase for better convergence
            learning_rate = 0.025, # Use paper's default
            workers       = 8,     # Keep as is
            seed          = 42     # Keep for reproducibility
        )

        sine_model.fit(graph, feature_matrix)
        sine_embeddings = sine_model.get_embedding()
        embeddings["SINE"] = normalize(sine_embeddings, norm='l2', axis=1)
        
        print(f"SINE embeddings shape: {embeddings['SINE'].shape}")
        print(f"SINE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['SINE'][0]):.4f}")

    # Print normalization summary
    print("\n" + "="*50)
    print("EMBEDDING NORMALIZATION SUMMARY")
    print("="*50)
    for method, emb in embeddings.items():
        # Check a few random vectors to verify normalization
        sample_norms = [np.linalg.norm(emb[i]) for i in range(min(5, len(emb)))]
        avg_norm = np.mean(sample_norms)
        print(f"{method}: Average norm = {avg_norm:.6f} (should be ~1.0)")
        print(f"  Sample norms: {[f'{norm:.4f}' for norm in sample_norms]}")

    # Save embeddings to disk
    save_embeddings(embeddings, dataset, drop_rate)
    
    data_info = {
        'A': A,
        'tf_idf': tf_idf,
        'raw': raw,
        'groups': groups,
        'graph': graph,
        'dataset': dataset,
        'drop_rate': drop_rate,
        'd': d
    }
    
    with open(f"{dataset}_data_dr{drop_rate}.pkl", 'wb') as f:
        pickle.dump(data_info, f)
    
    return embeddings, data_info

def save_embeddings(embeddings, dataset, drop_rate):
    """Save trained embeddings to disk"""
    os.makedirs('embeddings', exist_ok=True)
    
    for method, embedding in embeddings.items():
        filename = f"embeddings/{dataset}_{method}_dr{drop_rate}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(embedding, f)
        print(f"Saved {method} embeddings to {filename}")

def load_embeddings(dataset, drop_rate, methods=None):
    """Load pre-trained embeddings from disk"""
    if methods is None:
        methods = ["BASELINE_TFIDF", "RLE", "GELD", "SINE", "MUSAE", "TADW"]
    
    embeddings = {}
    for method in methods:
        filename = f"embeddings/{dataset}_{method}_dr{drop_rate}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                embeddings[method] = pickle.load(f)
            print(f"Loaded {method} embeddings from {filename}")
        else:
            print(f"Warning: {filename} not found")
    
    return embeddings

if __name__ == '__main__':
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import find
    # Configuration
    dataset = "cora2"
    d = 160
    drop_rate = 0.7
    methods = ["TADW", "GELD", "RLE", "MUSAE", "SINE", "BASELINE_TFIDF"]
    
    print("="*70)
    print("TRAINING ALL EMBEDDING METHODS")
    print("="*70)
    
    # Train all methods
    embeddings, data_info = train_all_embeddings(dataset, d, drop_rate, methods)
    # emb = embeddings["TADW", "GELD", "RLE", "MUSAE", "SINE", "BASELINE_TFIDF"]
    # print(np.mean(emb), np.std(emb))
    # print(emb[:5])
    # A = data_info['A']
    # i, j, _ = find(A)
    # sims = [cosine_similarity([emb[a]], [emb[b]])[0,0] for a,b in zip(i,j) if a != b]
    # print("min:", np.min(sims), "max:", np.max(sims), "mean:", np.mean(sims), "std:", np.std(sims))
    # print("nb sous 0.3:", np.sum(np.array(sims) < 0.3), "sur", len(sims))
        
    print(f"\nTraining complete! All embeddings saved for dataset: {dataset}")
    print(f"Drop rate: {drop_rate}")
    print(f"Embedding dimension: {d}")
    print(f"Methods trained: {list(embeddings.keys())}") 