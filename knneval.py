#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict

class KNNEmbeddingEvaluator:
    """
    Class to evaluate the effectiveness of embeddings for k-NN semantic search
    Neutral approach focused on retrieval performance
    """
    
    def __init__(self, embeddings, adjacency_matrix, raw_texts=None):
        """
        Args:
            embeddings: dict with embeddings by method
            adjacency_matrix: adjacency matrix of the document graph
            raw_texts: raw texts of the documents (optional)
        """
        self.embeddings = embeddings
        self.adjacency_matrix = adjacency_matrix
        self.raw_texts = raw_texts
        self.graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
        
    def evaluate_knn_retrieval_quality(self, k=5):
        """
        Directly evaluates the quality of k-NN search
        Uses the graph neighbors as ground truth
        """
        results = {}
        
        # Retrieve nodes with at least one neighbor
        nodes_with_neighbors = [node for node in self.graph.nodes() if self.graph.degree(node) > 0]
        
        if not nodes_with_neighbors:
            print("WARNING: No nodes with neighbors found!")
            return {method: {'precision@k': 0, 'recall@k': 0, 'f1@k': 0} for method in self.embeddings.keys()}
        
        print(f"k-NN evaluation on {len(nodes_with_neighbors)} nodes with neighbors")
        
        for method_name, embeddings in self.embeddings.items():
            print(f"  Evaluating {method_name}...")
            
            precision_scores = []
            recall_scores = []
            
            for doc_id in nodes_with_neighbors:
                # "True" neighbors according to the original graph
                true_neighbors = set(self.graph.neighbors(doc_id))
                
                if len(true_neighbors) == 0:
                    continue
                
                try:
                    # k-NN in the embedding space
                    doc_embedding = embeddings[doc_id].reshape(1, -1)
                    similarities = cosine_similarity(doc_embedding, embeddings)[0]
                    similarities[doc_id] = -1  # Exclude the document itself
                    
                    # Adapt k to the number of available documents
                    effective_k = min(k, len(embeddings) - 1)
                    top_k_indices = np.argsort(similarities)[-effective_k:][::-1]
                    retrieved_neighbors = set(top_k_indices)
                    
                    # Quality metrics
                    intersection = true_neighbors.intersection(retrieved_neighbors)
                    precision = len(intersection) / effective_k if effective_k > 0 else 0
                    recall = len(intersection) / len(true_neighbors) if len(true_neighbors) > 0 else 0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    
                except Exception as e:
                    print(f"    Error for document {doc_id}: {e}")
                    continue
            
            # Compute average metrics
            if precision_scores and recall_scores:
                mean_precision = np.mean(precision_scores)
                mean_recall = np.mean(recall_scores)
                f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
            else:
                mean_precision = mean_recall = f1_score = 0.0
            
            results[method_name] = {
                'precision@k': mean_precision,
                'recall@k': mean_recall,
                'f1@k': f1_score,
                'evaluated_documents': len(precision_scores),
                'k_used': k
            }
            
            print(f"    F1@{k}: {f1_score:.4f}, Precision: {mean_precision:.4f}, Recall: {mean_recall:.4f}")
        
        return results
    
    def evaluate_structure_preservation(self, max_distance=3):
        """
        Measures how well the embedding preserves the distances of the original graph
        """
        results = {}
        
        print("Evaluating structure preservation...")
        
        # Check graph connectivity
        if not nx.is_connected(self.graph):
            print("  Graph is not connected - analysis on the largest component")
            largest_cc = max(nx.connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
        else:
            subgraph = self.graph
        
        # Compute distances in the graph (sample to avoid O(n²))
        sample_size = min(100, len(subgraph.nodes()))
        sample_nodes = list(subgraph.nodes())[:sample_size]
        
        graph_distances = {}
        for node in sample_nodes:
            graph_distances[node] = nx.single_source_shortest_path_length(
                subgraph, node, cutoff=max_distance
            )
        
        for method_name, embeddings in self.embeddings.items():
            print(f"  Analyzing {method_name}...")
            
            distance_correlations = []
            
            # Analyze for each distance
            for target_distance in range(1, max_distance + 1):
                pairs_at_distance = []
                embedding_similarities = []
                
                for node1 in sample_nodes:
                    for node2, graph_dist in graph_distances.get(node1, {}).items():
                        if graph_dist == target_distance and node1 < node2:  # Avoid duplicates
                            try:
                                # Similarity in the embedding
                                sim = cosine_similarity([embeddings[node1]], [embeddings[node2]])[0][0]
                                embedding_similarities.append(sim)
                                pairs_at_distance.append((node1, node2))
                            except Exception as e:
                                continue
                
                # Analyze correlation for this distance
                if len(embedding_similarities) > 10:
                    # The greater the distance, the lower the similarity should be
                    expected_decay = [1.0 / target_distance] * len(embedding_similarities)
                    
                    try:
                        corr = np.corrcoef(expected_decay, embedding_similarities)[0, 1]
                        if not np.isnan(corr):
                            distance_correlations.append(abs(corr))  # Absolute value because we want the strength of the relationship
                    except:
                        pass
            
            # Structure preservation score
            structure_score = np.mean(distance_correlations) if distance_correlations else 0.0
            
            results[method_name] = {
                'structure_preservation_score': structure_score,
                'distance_correlations': distance_correlations,
                'analyzed_distances': list(range(1, len(distance_correlations) + 1)),
                'sample_size': len(sample_nodes)
            }
            
            print(f"    Structure preservation score: {structure_score:.4f}")
        
        return results
    
    def evaluate_query_types_performance(self, k=5):
        """
        Evaluates effectiveness according to different query profiles
        """
        results = {}
        
        print("Evaluation by query types...")
        
        # Categorize nodes by their degree (connectivity)
        node_degrees = dict(self.graph.degree())
        degree_values = list(node_degrees.values())
        
        if not degree_values:
            return {method: {} for method in self.embeddings.keys()}
        
        # Define percentiles
        p25, p75 = np.percentile(degree_values, [25, 75])
        
        categories = {
            'high_connectivity': [n for n, d in node_degrees.items() if d >= p75],
            'medium_connectivity': [n for n, d in node_degrees.items() if p25 <= d < p75],
            'low_connectivity': [n for n, d in node_degrees.items() if d > 0 and d < p25]
        }
        
        print(f"  Categories: High({len(categories['high_connectivity'])}), "
              f"Medium({len(categories['medium_connectivity'])}), "
              f"Low({len(categories['low_connectivity'])})")
        
        for method_name, embeddings in self.embeddings.items():
            print(f"  Analyzing {method_name}...")
            
            category_performance = {}
            
            for category_name, nodes in categories.items():
                if len(nodes) == 0:
                    category_performance[category_name] = {
                        'precision@k': 0.0,
                        'sample_size': 0
                    }
                    continue
                
                # Sample for this category
                sample_size = min(20, len(nodes))
                category_sample = nodes[:sample_size]
                
                precision_scores = []
                
                for node in category_sample:
                    true_neighbors = set(self.graph.neighbors(node))
                    if not true_neighbors:
                        continue
                    
                    try:
                        # k-NN
                        similarities = cosine_similarity([embeddings[node]], embeddings)[0]
                        similarities[node] = -1
                        
                        effective_k = min(k, len(embeddings) - 1)
                        top_k = np.argsort(similarities)[-effective_k:][::-1]
                        
                        # Precision
                        intersection = true_neighbors.intersection(set(top_k))
                        precision = len(intersection) / effective_k if effective_k > 0 else 0
                        precision_scores.append(precision)
                        
                    except Exception as e:
                        continue
                
                category_performance[category_name] = {
                    'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
                    'sample_size': len(precision_scores)
                }
                
                print(f"    {category_name}: {category_performance[category_name]['precision@k']:.4f}")
            
            results[method_name] = category_performance
        
        return results
    
    def evaluate_robustness(self, k=5, n_trials=5):
        """
        Evaluates the robustness and consistency of k-NN results
        """
        results = {}
        
        print("Evaluating robustness...")
        
        nodes_with_neighbors = [n for n in self.graph.nodes() if self.graph.degree(n) > 0]
        
        if len(nodes_with_neighbors) < 10:
            print("  Not enough nodes for robustness evaluation")
            return {method: {'robustness_score': 0, 'mean_performance': 0} for method in self.embeddings.keys()}
        
        for method_name, embeddings in self.embeddings.items():
            print(f"  Robustness test {method_name}...")
            
            trial_scores = []
            
            for trial in range(n_trials):
                # Different random sampling for each trial
                np.random.seed(trial + 42)  # Reproducibility
                sample_size = min(30, len(nodes_with_neighbors))
                sample_nodes = np.random.choice(
                    nodes_with_neighbors, 
                    size=sample_size, 
                    replace=False
                )
                
                precision_scores = []
                for node in sample_nodes:
                    true_neighbors = set(self.graph.neighbors(node))
                    if not true_neighbors:
                        continue
                    
                    try:
                        similarities = cosine_similarity([embeddings[node]], embeddings)[0]
                        similarities[node] = -1
                        
                        effective_k = min(k, len(embeddings) - 1)
                        top_k = np.argsort(similarities)[-effective_k:][::-1]
                        
                        intersection = true_neighbors.intersection(set(top_k))
                        precision = len(intersection) / effective_k if effective_k > 0 else 0
                        precision_scores.append(precision)
                        
                    except Exception as e:
                        continue
                
                if precision_scores:
                    trial_scores.append(np.mean(precision_scores))
            
            # Robustness calculation
            if trial_scores and np.mean(trial_scores) > 0:
                mean_perf = np.mean(trial_scores)
                std_perf = np.std(trial_scores)
                robustness_score = 1 - (std_perf / mean_perf)  # More stable = more robust
                robustness_score = max(0, robustness_score)  # Bound to [0,1]
            else:
                mean_perf = 0
                std_perf = 0
                robustness_score = 0
            
            results[method_name] = {
                'mean_performance': mean_perf,
                'std_performance': std_perf,
                'robustness_score': robustness_score,
                'trial_scores': trial_scores
            }
            
            print(f"    Mean performance: {mean_perf:.4f}, Robustness: {robustness_score:.4f}")
        
        return results
    
    def comprehensive_knn_evaluation(self, k=5):
        """
        Comprehensive evaluation oriented towards k-NN effectiveness
        """
        print("=" * 70)
        print("K-NN SEARCH EFFECTIVENESS EVALUATION")
        print("=" * 70)
        print(f"Graph: {len(self.graph.nodes())} nodes, {len(self.graph.edges())} edges")
        print(f"Embedding methods: {list(self.embeddings.keys())}")
        print(f"k used: {k}")
        print()
        
        results = {}
        
        try:
            print("1. K-NN RETRIEVAL QUALITY")
            print("-" * 50)
            results['knn_retrieval_quality'] = self.evaluate_knn_retrieval_quality(k=k)
            df_1 = pd.DataFrame(results['knn_retrieval_quality']).T
            print(df_1.round(4))
            print()
        except Exception as e:
            print(f"ERROR k-NN evaluation: {e}")
            results['knn_retrieval_quality'] = {}
        
        try:
            print("2. STRUCTURE PRESERVATION")
            print("-" * 50)
            results['structure_preservation'] = self.evaluate_structure_preservation()
            df_2 = pd.DataFrame(results['structure_preservation']).T[['structure_preservation_score', 'sample_size']]
            print(df_2.round(4))
            print()
        except Exception as e:
            print(f"ERROR Structure preservation: {e}")
            results['structure_preservation'] = {}
        
        try:
            print("3. PERFORMANCE BY QUERY TYPE")
            print("-" * 50)
            results['query_types_performance'] = self.evaluate_query_types_performance(k=k)
            
            # Formatted display
            for method, categories in results['query_types_performance'].items():
                print(f"{method}:")
                for cat, metrics in categories.items():
                    print(f"  {cat}: {metrics['precision@k']:.4f} (n={metrics['sample_size']})")
            print()
        except Exception as e:
            print(f"ERROR Performance by type: {e}")
            results['query_types_performance'] = {}
        
        try:
            print("4. ROBUSTNESS")
            print("-" * 50)
            results['robustness'] = self.evaluate_robustness(k=k)
            df_4 = pd.DataFrame(results['robustness']).T[['mean_performance', 'robustness_score']]
            print(df_4.round(4))
            print()
        except Exception as e:
            print(f"ERROR Robustness: {e}")
            results['robustness'] = {}
        
        # Global score by method
        print("5. OVERALL RANKING")
        print("-" * 50)
        method_scores = {}
        
        for method in self.embeddings.keys():
            scores = []
            
            # F1@k (weight 40%)
            if method in results.get('knn_retrieval_quality', {}):
                f1_score = results['knn_retrieval_quality'][method].get('f1@k', 0)
                scores.append(('f1@k', f1_score, 0.4))
            
            # Structure preservation (weight 30%)
            if method in results.get('structure_preservation', {}):
                struct_score = results['structure_preservation'][method].get('structure_preservation_score', 0)
                scores.append(('structure', struct_score, 0.3))
            
            # Robustness (weight 30%)
            if method in results.get('robustness', {}):
                robust_score = results['robustness'][method].get('robustness_score', 0)
                scores.append(('robustness', robust_score, 0.3))
            
            # Weighted score
            if scores:
                weighted_score = sum(score * weight for _, score, weight in scores)
                total_weight = sum(weight for _, _, weight in scores)
                overall_score = weighted_score / total_weight if total_weight > 0 else 0
            else:
                overall_score = 0
            
            method_scores[method] = {
                'overall_knn_effectiveness': overall_score,
                'component_scores': {name: score for name, score, _ in scores}
            }
        
        # Sort by descending score
        ranked_methods = sorted(method_scores.items(), key=lambda x: x[1]['overall_knn_effectiveness'], reverse=True)
        
        print("Ranking by k-NN effectiveness:")
        for i, (method, metrics) in enumerate(ranked_methods, 1):
            print(f"{i}. {method}: {metrics['overall_knn_effectiveness']:.4f}")
        
        results['overall_ranking'] = method_scores
        results['ranked_methods'] = ranked_methods
        
        return results