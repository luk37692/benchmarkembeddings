#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from utils import read_data
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from models import train_rle, train_geld, train_tadw, prepare_data
import networkx as nx
from semantic_fracture import SemanticFractureEvaluator

# Configuration
methods = ["BASELINE_TFIDF", "RLE", "GELD", "SINE", "MUSAE", "TADW"]  # Ajout baseline
dataset = "cora2"
d = 160
k = 5  # Number of similar documents to retrieve

print(f"\nLoading {dataset}...")
# Load data
tf_idf, groups, A, graph, voc, raw, tf = read_data(dataset)
n = A.shape[0]
print(f"{n} nodes, {tf_idf.shape[1]} features, {len(graph.edges())} edges")

# Prepare and learn embeddings
data_graph, data_text, sigma_init, D_init, U = prepare_data(d, tf, A, voc, raw)
embeddings = {}

# 1. BASELINE: TF-IDF pur (sans structure de graphe)
print("\n" + "="*50)
print("BASELINE: TF-IDF (sans liens)")
print("="*50)
# Réduire les dimensions pour comparaison équitable
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=d, random_state=42)
tfidf_reduced = svd.fit_transform(tf_idf)
embeddings["BASELINE_TFIDF"] = normalize(tfidf_reduced, norm='l2', axis=1)
print(f"Baseline TF-IDF embeddings shape: {embeddings['BASELINE_TFIDF'].shape}")

# 2. Méthodes avec liens (votre code existant)
if "GELD" in methods:
    alpha_map = {"cora2": 0.99, "dblp": 0.8, "nyt": 0.95}
    D, D_norm, sigma = train_geld(
        d, data_graph, data_text, U, D_init, sigma_init,
        n_epoch=40, lamb=None, alpha=alpha_map[dataset]
    )
    embeddings["GELD"] = normalize(D_norm, norm='l2', axis=1)

    # Normalize GELD embeddings
    embeddings["GELD"] = normalize(D_norm, norm='l2', axis=1)
    print("\nGELD embeddings (normalized):")
    print(embeddings["GELD"])
    print(f"GELD embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['GELD'][0]):.4f}")

if "RLE" in methods:
    window_map = {"cora2": 15, "dblp": 5, "nyt": 10}
    rle_embeddings, _ = train_rle(A, tf, U, d, 0.7, verbose=True)
    # Normalize RLE embeddings
    embeddings["RLE"] = normalize(rle_embeddings, norm='l2', axis=1)
    print("\nRLE embeddings (normalized):")
    print(embeddings["RLE"])
    print(f"RLE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['RLE'][0]):.4f}")

if "MUSAE" in methods:
    print("\nTraining MUSAE embeddings...")
    from karateclub import MUSAE

    # Convert TF-IDF features to the format expected by Karate Club
    # Karate Club SINE expects a scipy sparse COO matrix for features
    if hasattr(tf_idf, 'tocoo'):
        feature_matrix = tf_idf.tocoo()
    else:
        from scipy.sparse import coo_matrix
        feature_matrix = coo_matrix(tf_idf)

    musae_model = MUSAE(
        dimensions=d,
        epochs=10,
        learning_rate=0.01,
        seed=42
    )
    musae_model.fit(graph, feature_matrix)
    musae_embeddings = musae_model.get_embedding()
    # Normalize MUSAE embeddings
    embeddings["MUSAE"] = normalize(musae_embeddings, norm='l2', axis=1)
    print("\nMUSAE embeddings (normalized):")
    print(embeddings["MUSAE"])
    print(f"MUSAE embeddings shape: {embeddings['MUSAE'].shape}")
    print(f"MUSAE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['MUSAE'][0]):.4f}")


if "TADW" in methods:
    print("\nTraining TADW embeddings...")
    
    # TADW hyperparameters based on dataset
    tadw_params = {
        "cora2": {"order": 2, "iter_max": 5, "lamb": 0.2},
        "dblp": {"order": 2, "iter_max": 20, "lamb": 0.1}, 
        "nyt": {"order": 2, "iter_max": 15, "lamb": 0.15}
    }
    
    params = tadw_params.get(dataset, {"order": 2, "iter_max": 15, "lamb": 0.2})
    
    # Train TADW model
    tadw_embeddings, training_time = train_tadw(
        A, tf_idf, d=d, 
        order=params["order"],
        iter_max=params["iter_max"], 
        lamb=params["lamb"],
        verbose=True
    )
    
    # Normalize TADW embeddings
    embeddings["TADW"] = normalize(tadw_embeddings, norm='l2', axis=1)
    print("\nTADW embeddings (normalized):")
    print(embeddings["TADW"])
    print(f"TADW embeddings shape: {embeddings['TADW'].shape}")
    print(f"TADW embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['TADW'][0]):.4f}")


if "SINE" in methods:
    print("\nTraining SINE embeddings...")
    
    # Use Karate Club SINE implementation
    try:
        from karateclub import SINE
        
        print("Using Karate Club SINE implementation...")
        
        # Convert TF-IDF features to the format expected by Karate Club
        # Karate Club SINE expects a scipy sparse COO matrix for features
        if hasattr(tf_idf, 'tocoo'):
            feature_matrix = tf_idf.tocoo()
        else:
            from scipy.sparse import coo_matrix
            feature_matrix = coo_matrix(tf_idf)
        
        # Initialize SINE model with correct parameters
        sine_model = SINE(
            walk_number=10,        # Number of random walks
            walk_length=80,        # Length of random walks  
            dimensions=d,          # Embedding dimensions
            workers=4,             # Number of cores
            window_size=5,         # Matrix power order
            epochs=20,              # Number of epochs
            learning_rate=0.01,    # HogWild! learning rate
            min_count=1,           # Minimal count of node occurrences
            seed=42                # Random seed value
        )
        
        # Fit the model - SINE expects graph and feature matrix
        sine_model.fit(graph, feature_matrix)
        
        # Get embeddings and normalize
        sine_embeddings = sine_model.get_embedding()
        embeddings["SINE"] = normalize(sine_embeddings, norm='l2', axis=1)
        
        print(f"SINE embeddings shape: {embeddings['SINE'].shape}")
        print(f"SINE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['SINE'][0]):.4f}")
        
    except ImportError as e:
        print(f"Karate Club not available: {e}")
        print("Installing Karate Club...")
        
        try:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "karateclub"], check=True)
            print("Karate Club installed successfully!")
            print("Please restart the script to use SINE")
            raise RuntimeError("Please restart after Karate Club installation")
            
        except Exception as install_error:
            print(f"Failed to install Karate Club: {install_error}")
            print("Please install manually: pip install karateclub")
            raise
    
    except Exception as e:
        print(f"Error with Karate Club SINE: {e}")
        print("Falling back to simplified implementation...")
        
        # Simple fallback implementation
        from sklearn.decomposition import TruncatedSVD
        
        # Combine structural and textual features
        if hasattr(tf_idf, 'toarray'):
            text_features = tf_idf.toarray()
        else:
            text_features = tf_idf
            
        # Use adjacency matrix as structural features
        if hasattr(A, 'toarray'):
            adj_features = A.toarray()
        else:
            adj_features = A
        
        # Apply dimensionality reduction
        svd = TruncatedSVD(n_components=d, random_state=42)
        
        # Reduce text features
        text_reduced = svd.fit_transform(text_features)
        
        # Use normalized adjacency as simplified structural features
        adj_normalized = normalize(adj_features, norm='l2', axis=1)
        struct_reduced = svd.fit_transform(adj_normalized)
        
        # Combine with equal weighting
        alpha = 0.5
        min_dims = min(text_reduced.shape[1], struct_reduced.shape[1])
        combined = alpha * struct_reduced[:, :min_dims] + (1 - alpha) * text_reduced[:, :min_dims]
        
        # Normalize the combined embeddings
        embeddings["SINE"] = normalize(combined, norm='l2', axis=1)
        print(f"Fallback SINE embeddings shape: {embeddings['SINE'].shape}")
        print(f"Fallback SINE embeddings norm check (should be ~1.0): {np.linalg.norm(embeddings['SINE'][0]):.4f}")
    
    print("\nSINE embeddings (normalized):")
    print(embeddings["SINE"])
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


# Build NearestNeighbors index for retrieval
neighbors_index = {}
for method, emb in embeddings.items():
    nbrs = NearestNeighbors(metric='cosine').fit(emb)
    neighbors_index[method] = nbrs

def comprehensive_evaluation(embeddings, A, raw_texts, k=5):
    """
    Évaluation complète incluant plusieurs documents de test
    """
    # 1. Sélectionner plusieurs documents test (différents degrés de connectivité)
    graph_nx = nx.from_scipy_sparse_array(A)
    degrees = dict(graph_nx.degree())
    
    # Documents avec différents niveaux de connectivité
    high_degree_nodes = [node for node, deg in degrees.items() if deg >= 10]
    medium_degree_nodes = [node for node, deg in degrees.items() if 3 <= deg < 10]
    low_degree_nodes = [node for node, deg in degrees.items() if deg == 1]
    
    test_nodes = []
    if high_degree_nodes: test_nodes.extend(np.random.choice(high_degree_nodes, min(3, len(high_degree_nodes)), replace=False))
    if medium_degree_nodes: test_nodes.extend(np.random.choice(medium_degree_nodes, min(5, len(medium_degree_nodes)), replace=False))
    if low_degree_nodes: test_nodes.extend(np.random.choice(low_degree_nodes, min(2, len(low_degree_nodes)), replace=False))
    
    print(f"Documents de test sélectionnés: {test_nodes}")
    
    # 2. Évaluation de similarité pour chaque document test
    all_results = []
    
    for test_doc in test_nodes:
        print(f"\n--- Évaluation pour le document {test_doc} ---")
        doc_degree = degrees[test_doc]
        print(f"Degré de connectivité: {doc_degree}")
        
        for method in embeddings:
            result = retrieve_similar(test_doc, method, k)
            result['test_doc'] = test_doc
            result['test_doc_degree'] = doc_degree
            all_results.append(result)
    
    # 3. Évaluation de la fracture sémantique
    print("\n" + "="*60)
    print("ÉVALUATION DE LA FRACTURE SÉMANTIQUE")
    print("="*60)
    
    fracture_evaluator = SemanticFractureEvaluator(embeddings, A, raw_texts)
    fracture_results = fracture_evaluator.comprehensive_evaluation()
    
    # 4. Analyse comparative
    print("\n" + "="*60)
    print("ANALYSE COMPARATIVE: BASELINE vs MÉTHODES AVEC LIENS")
    print("="*60)
    
    # Comparer BASELINE vs autres méthodes
    baseline_fracture = fracture_results['fracture_severity']['BASELINE_TFIDF']
    
    print(f"Baseline (TF-IDF) - Sévérité de fracture: {baseline_fracture['fracture_severity_index']:.4f}")
    print("Amélioration apportée par les liens:")
    
    for method in embeddings:
        if method != 'BASELINE_TFIDF':
            method_fracture = fracture_results['fracture_severity'][method]
            improvement = baseline_fracture['fracture_severity_index'] - method_fracture['fracture_severity_index']
            print(f"  {method}: {improvement:.4f} ({improvement/baseline_fracture['fracture_severity_index']*100:.1f}% d'amélioration)")
    
    return {
        'similarity_results': pd.concat(all_results, ignore_index=True),
        'fracture_evaluation': fracture_results,
        'test_nodes': test_nodes
    }

def retrieve_similar(doc_idx, method='RLE', k=3):
    """Version améliorée avec plus d'informations"""
    emb = embeddings[method]
    nbrs = neighbors_index[method]
    distances, indices = nbrs.kneighbors(emb[doc_idx].reshape(1, -1), n_neighbors=k + 1)
    
    # Drop the query itself
    sim_idxs = indices[0][1:]
    sim_dists = distances[0][1:]
    
    # Vérifier si les documents similaires sont liés dans le graphe
    linked_status = []
    for sim_idx in sim_idxs:
        is_linked = A[doc_idx, sim_idx] > 0 or A[sim_idx, doc_idx] > 0
        linked_status.append(is_linked)
    
    results = pd.DataFrame({
        'method': [method] * k,
        'query_idx': [doc_idx] * k,
        'neighbor_idx': sim_idxs,
        'cosine_distance': sim_dists,
        'cosine_similarity': 1 - sim_dists,
        'is_actually_linked': linked_status,  # NOUVEAU: information sur les liens réels
        'text_snippet': [raw[i][:200] + "..." if len(raw[i]) > 200 else raw[i] for i in sim_idxs]
    })
    return results

def analyze_link_preservation():
    """
    Analyse spécifique: combien de liens réels sont préservés dans les top-k ?
    """
    print("\n" + "="*60)
    print("ANALYSE DE PRÉSERVATION DES LIENS")
    print("="*60)
    
    graph_nx = nx.from_scipy_sparse_array(A)
    
    # Prendre tous les nœuds avec au moins 2 liens
    connected_nodes = [node for node in graph_nx.nodes() if graph_nx.degree(node) >= 2]
    sample_nodes = np.random.choice(connected_nodes, min(20, len(connected_nodes)), replace=False)
    
    link_preservation_results = {}
    
    for method in embeddings:
        preserved_links = 0
        total_possible_links = 0
        
        for node in sample_nodes:
            # Vrais voisins
            true_neighbors = set(graph_nx.neighbors(node))
            total_possible_links += len(true_neighbors)
            
            # Top-k voisins selon embedding
            result = retrieve_similar(node, method, k=min(10, len(true_neighbors)*2))
            found_neighbors = set(result['neighbor_idx'].tolist())
            
            # Intersection
            preserved = len(true_neighbors.intersection(found_neighbors))
            preserved_links += preserved
        
        preservation_rate = preserved_links / total_possible_links if total_possible_links > 0 else 0
        link_preservation_results[method] = preservation_rate
        
        print(f"{method}: {preservation_rate:.4f} ({preservation_rate*100:.1f}% des liens préservés)")
    
    return link_preservation_results

# Exécution de l'évaluation complète
if __name__ == '__main__':
    print("\n" + "="*70)
    print("BENCHMARK COMPLET POUR L'ÉTUDE DE FRACTURE SÉMANTIQUE")
    print("="*70)
    
    # 1. Évaluation complète
    comprehensive_results = comprehensive_evaluation(embeddings, A, raw, k=5)
    
    # 2. Analyse de préservation des liens
    link_preservation = analyze_link_preservation()
    
    # 3. Sauvegarde des résultats
    comprehensive_results['similarity_results'].to_csv('comprehensive_similarity_results.csv', index=False)
    
    # Sauvegarder les résultats de fracture
    fracture_summary = pd.DataFrame({
        method: {
            'fracture_severity': results['fracture_severity_index'],
            'mean_linked_similarity': comprehensive_results['fracture_evaluation']['linked_similarity'][method]['mean_linked_similarity'],
            'link_preservation_rate': link_preservation[method]
        }
        for method, results in comprehensive_results['fracture_evaluation']['fracture_severity'].items()
    }).T
    
    fracture_summary.to_csv('fracture_analysis_summary.csv')
    
    print(f"\nRésultats sauvegardés:")
    print("- comprehensive_similarity_results.csv")
    print("- fracture_analysis_summary.csv")
    
    # 4. Conclusion
    print("\n" + "="*60)
    print("CONCLUSION DE L'ÉTUDE")
    print("="*60)
    
    baseline_fracture = comprehensive_results['fracture_evaluation']['fracture_severity']['BASELINE_TFIDF']['fracture_severity_index']
    best_method = min(comprehensive_results['fracture_evaluation']['fracture_severity'].items(), 
                     key=lambda x: x[1]['fracture_severity_index'])
    
    improvement = (baseline_fracture - best_method[1]['fracture_severity_index']) / baseline_fracture * 100
    
    print(f"La méthode {best_method[0]} réduit la fracture sémantique de {improvement:.1f}% par rapport au baseline TF-IDF")
    print(f"Fracture baseline: {baseline_fracture:.4f}")
    print(f"Fracture {best_method[0]}: {best_method[1]['fracture_severity_index']:.4f}")