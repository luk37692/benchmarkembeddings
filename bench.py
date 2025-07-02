#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pickle
import os
from semantic_fracture import KNNEmbeddingEvaluator
from train_models import load_embeddings

def load_data_info(dataset, drop_rate):
    """Load the saved data information"""
    filename = f"{dataset}_data_dr{drop_rate}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data_info = pickle.load(f)
        print(f"Loaded data info from {filename}")
        return data_info
    else:
        raise FileNotFoundError(f"Data file {filename} not found. Please run train_models.py first.")

def build_knn_indices(embeddings):
    """Build NearestNeighbors index for retrieval"""
    neighbors_index = {}
    for method, emb in embeddings.items():
        nbrs = NearestNeighbors(metric='cosine').fit(emb)
        neighbors_index[method] = nbrs
    return neighbors_index

def retrieve_similar(doc_idx, method, embeddings, neighbors_index, A, raw, k=3):
    """Version améliorée avec plus d'informations"""
    emb = embeddings[method]
    nbrs = neighbors_index[method]
    distances, indices = nbrs.kneighbors(emb[doc_idx].reshape(1, -1), n_neighbors=k + 1)
    
    # Drop the query itself
    sim_idxs = indices[0][1:]
    sim_dists = distances[0][1:]
    
    # Verify if the similar documents are linked in the graph
    linked_status = []
    for sim_idx in sim_idxs:
        is_linked = A[doc_idx, sim_idx] > 0 or A[sim_idx, doc_idx] > 0
        linked_status.append(is_linked)
    
    # Create text snippets by joining tokens and then truncating
    text_snippets = []
    for i in sim_idxs:
        # Convert list of tokens to string
        text = " ".join(raw[i])
        # Truncate if too long
        if len(text) > 200:
            snippet = text[:200] + "..."
        else:
            snippet = text
        text_snippets.append(snippet)
    
    results = pd.DataFrame({
        'method': [method] * k,
        'query_idx': [doc_idx] * k,
        'neighbor_idx': sim_idxs,
        'cosine_distance': sim_dists,
        'cosine_similarity': 1 - sim_dists,
        'is_actually_linked': linked_status,  # NEW: information on actual links
        'text_snippet': text_snippets
    })
    return results

def comprehensive_evaluation(embeddings, neighbors_index, A, raw_texts, k=5):
    """
    Comprehensive evaluation including multiple test documents
    """
    # 1. Select multiple test documents (different degrees of connectivity)
    graph_nx = nx.from_scipy_sparse_matrix(A)
    degrees = dict(graph_nx.degree())
    
    # Documents with different levels of connectivity
    high_degree_nodes = [node for node, deg in degrees.items() if deg >= 10]
    medium_degree_nodes = [node for node, deg in degrees.items() if 3 <= deg < 10]
    low_degree_nodes = [node for node, deg in degrees.items() if deg == 1]
    
    test_nodes = []
    if high_degree_nodes: test_nodes.extend(np.random.choice(high_degree_nodes, min(3, len(high_degree_nodes)), replace=False))
    if medium_degree_nodes: test_nodes.extend(np.random.choice(medium_degree_nodes, min(5, len(medium_degree_nodes)), replace=False))
    if low_degree_nodes: test_nodes.extend(np.random.choice(low_degree_nodes, min(2, len(low_degree_nodes)), replace=False))
    
    print(f"Test documents selected: {test_nodes}")
    
    # 2. Evaluation of similarity for each test document
    all_results = []
    
    for test_doc in test_nodes:
        print(f"\n--- Evaluation for document {test_doc} ---")
        doc_degree = degrees[test_doc]
        print(f"Degree of connectivity: {doc_degree}")
        
        for method in embeddings:
            result = retrieve_similar(test_doc, method, embeddings, neighbors_index, A, raw_texts, k)
            result['test_doc'] = test_doc
            result['test_doc_degree'] = doc_degree
            all_results.append(result)
    
    # 3. Evaluation of k-NN effectiveness
    print("\n" + "="*60)
    print("EVALUATION OF k-NN EFFECTIVENESS")
    print("="*60)
    
    knn_evaluator = KNNEmbeddingEvaluator(embeddings, A, raw_texts)
    knn_results = knn_evaluator.comprehensive_knn_evaluation(k=k)
    
    # 4. Analyse comparative
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS: BASELINE vs METHODS WITH LINKS")
    print("="*60)
    
    # Get global scores
    if 'overall_ranking' in knn_results:
        print("Ranking by k-NN effectiveness:")
        for i, (method, metrics) in enumerate(knn_results['ranked_methods'], 1):
            print(f"{i}. {method}: {metrics['overall_knn_effectiveness']:.4f}")
        
        # Compare BASELINE vs other methods
        if 'BASELINE_TFIDF' in knn_results['overall_ranking']:
            baseline_score = knn_results['overall_ranking']['BASELINE_TFIDF']['overall_knn_effectiveness']
            print(f"\nBaseline (TF-IDF) - Effectiveness score: {baseline_score:.4f}")
            print("Improvement provided by links:")
            
            for method in embeddings:
                if method != 'BASELINE_TFIDF' and method in knn_results['overall_ranking']:
                    method_score = knn_results['overall_ranking'][method]['overall_knn_effectiveness']
                    improvement = method_score - baseline_score
                    if baseline_score > 0:
                        improvement_pct = improvement / baseline_score * 100
                        print(f"  {method}: +{improvement:.4f} ({improvement_pct:.1f}% improvement)")
                    else:
                        print(f"  {method}: {method_score:.4f}")
    else:
        print("No ranking available.")
    
    return {
        'similarity_results': pd.concat(all_results, ignore_index=True),
        'knn_evaluation': knn_results,
        'test_nodes': test_nodes
    }

def analyze_link_preservation(embeddings, neighbors_index, A, k=10):
    """
    Specific analysis: how many real links are preserved in the top-k?
    """
    print("\n" + "="*60)
    print("ANALYSIS OF LINK PRESERVATION")
    print("="*60)
    
    graph_nx = nx.from_scipy_sparse_matrix(A)
    
    # Take all nodes with at least 2 links
    connected_nodes = [node for node in graph_nx.nodes() if graph_nx.degree(node) >= 2]
    sample_nodes = np.random.choice(connected_nodes, min(20, len(connected_nodes)), replace=False)
    
    link_preservation_results = {}
    
    for method in embeddings:
        preserved_links = 0
        total_possible_links = 0
        
        for node in sample_nodes:
            # True neighbors
            true_neighbors = set(graph_nx.neighbors(node))
            total_possible_links += len(true_neighbors)
            
            # Top-k neighbors according to embedding
            emb = embeddings[method]
            nbrs = neighbors_index[method]
            distances, indices = nbrs.kneighbors(emb[node].reshape(1, -1), n_neighbors=min(k, len(true_neighbors)*2) + 1)
            found_neighbors = set(indices[0][1:])  # Exclude self
            
            # Intersection
            preserved = len(true_neighbors.intersection(found_neighbors))
            preserved_links += preserved
        
        preservation_rate = preserved_links / total_possible_links if total_possible_links > 0 else 0
        link_preservation_results[method] = preservation_rate
        
        print(f"{method}: {preservation_rate:.4f} ({preservation_rate*100:.1f}% of links preserved)")
    
    return link_preservation_results

def run_knn_benchmark(dataset="cora2", drop_rate=0.5, k=5):
    """
    Main function to run KNN benchmark using pre-trained embeddings
    """
    print("="*70)
    print("KNN BENCHMARK FOR EMBEDDING EFFECTIVENESS EVALUATION")
    print("="*70)
    
    # 1. Load pre-trained embeddings
    print(f"\nLoading pre-trained embeddings for {dataset} (drop_rate={drop_rate})...")
    embeddings = load_embeddings(dataset, drop_rate)
    
    if not embeddings:
        print("No embeddings found! Please run train_models.py first.")
        return
    
    # 2. Load data information
    data_info = load_data_info(dataset, drop_rate)
    A = data_info['A']
    raw = data_info['raw']
    
    print(f"Loaded embeddings for methods: {list(embeddings.keys())}")
    print(f"Graph: {A.shape[0]} nodes, {A.nnz} edges")
    
    # 3. Build KNN indices
    print("\nBuilding KNN indices...")
    neighbors_index = build_knn_indices(embeddings)
    
    # 4. Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    comprehensive_results = comprehensive_evaluation(embeddings, neighbors_index, A, raw, k=k)
    
    # 5. Link preservation analysis
    link_preservation = analyze_link_preservation(embeddings, neighbors_index, A, k=10)
    
    # 6. Save results
    print("\nSaving results...")
    comprehensive_results['similarity_results'].to_csv(f'comprehensive_similarity_results_{drop_rate}.csv', index=False)
    
    # Save k-NN effectiveness results
    if 'overall_ranking' in comprehensive_results['knn_evaluation']:
        knn_summary = pd.DataFrame({
            method: {
                'overall_knn_effectiveness': scores['overall_knn_effectiveness'],
                'f1_score': scores['component_scores'].get('f1@k', 0),
                'structure_preservation': scores['component_scores'].get('structure', 0),
                'robustness': scores['component_scores'].get('robustness', 0),
                'link_preservation_rate': link_preservation.get(method, 0)
            }
            for method, scores in comprehensive_results['knn_evaluation']['overall_ranking'].items()
        }).T
        
        knn_summary.to_csv(f'knn_analysis_summary_{drop_rate}.csv')
    else:
        print("No ranking results available for saving.")
    
    print(f"\nSaved results:")
    print(f"- comprehensive_similarity_results_{drop_rate}.csv")
    print(f"- knn_analysis_summary_{drop_rate}.csv")
    
    # 7. Conclusion
    print("\n" + "="*60)
    print("CONCLUSION OF THE STUDY")
    print("="*60)
    
    if 'ranked_methods' in comprehensive_results['knn_evaluation'] and comprehensive_results['knn_evaluation']['ranked_methods']:
        best_method = comprehensive_results['knn_evaluation']['ranked_methods'][0]
        best_method_name, best_method_scores = best_method
        
        baseline_score = 0
        if 'BASELINE_TFIDF' in comprehensive_results['knn_evaluation']['overall_ranking']:
            baseline_score = comprehensive_results['knn_evaluation']['overall_ranking']['BASELINE_TFIDF']['overall_knn_effectiveness']
        
        if baseline_score > 0:
            improvement = (best_method_scores['overall_knn_effectiveness'] - baseline_score) / baseline_score * 100
            print(f"The method {best_method_name} improves k-NN effectiveness by {improvement:.1f}% compared to the baseline TF-IDF")
        else:
            print(f"The best method is {best_method_name} with a score of {best_method_scores['overall_knn_effectiveness']:.4f}")
        
        print(f"Score baseline: {baseline_score:.4f}")
        print(f"Score {best_method_name}: {best_method_scores['overall_knn_effectiveness']:.4f}")
    else:
        print("Impossible to determine the best method.")
    
    return comprehensive_results, link_preservation

if __name__ == '__main__':
    # Configuration
    dataset = "cora2"
    drop_rate = 0.5
    k = 5  # Number of similar documents to retrieve
    
    # Run benchmark
    results, link_preservation = run_knn_benchmark(dataset, drop_rate, k)