#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict

class SemanticFractureEvaluator:
    """
    Classe pour évaluer la fracture sémantique dans les embeddings de documents liés
    """
    
    def __init__(self, embeddings, adjacency_matrix, raw_texts=None):
        """
        Args:
            embeddings: dict avec les embeddings par méthode
            adjacency_matrix: matrice d'adjacence du graphe de documents
            raw_texts: textes bruts des documents (optionnel)
        """
        self.embeddings = embeddings
        self.adjacency_matrix = adjacency_matrix
        self.raw_texts = raw_texts
        self.graph = nx.from_scipy_sparse_array(adjacency_matrix)
        
    def method_1_linked_similarity_preservation(self):
        """
        Méthode 1: Préservation de la similarité pour les documents liés
        
        Mesure si les documents directement liés restent proches dans l'espace embedding
        """
        results = {}
        
        # Récupérer toutes les paires de documents directement liés
        linked_pairs = list(self.graph.edges())
        
        for method_name, emb in self.embeddings.items():
            similarities = []
            
            for doc_i, doc_j in linked_pairs:
                # Calculer la similarité cosinus entre les embeddings
                sim = cosine_similarity([emb[doc_i]], [emb[doc_j]])[0][0]
                similarities.append(sim)
            
            # Statistiques
            results[method_name] = {
                'mean_linked_similarity': np.mean(similarities),
                'median_linked_similarity': np.median(similarities),
                'std_linked_similarity': np.std(similarities),
                'min_linked_similarity': np.min(similarities),
                'low_similarity_count': sum(1 for s in similarities if s < 0.5),  # Seuil arbitraire
                'total_linked_pairs': len(similarities)
            }
            
        return results
    
    def method_2_link_distance_correlation(self):
        """
        Méthode 2: Corrélation entre distance dans le graphe et distance dans l'embedding
        
        Vérifie si la distance topologique (plus court chemin) corrèle avec 
        la distance sémantique dans l'embedding
        """
        results = {}
        
        # Calculer les distances de plus court chemin pour tous les pairs
        path_lengths = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=5))
        
        for method_name, emb in self.embeddings.items():
            graph_distances = []
            embedding_distances = []
            
            # Échantillonner des paires pour éviter une complexité trop élevée
            sample_nodes = list(self.graph.nodes())[:min(100, len(self.graph.nodes()))]
            
            for i, node_i in enumerate(sample_nodes):
                for node_j in sample_nodes[i+1:]:
                    if node_j in path_lengths.get(node_i, {}):
                        graph_dist = path_lengths[node_i][node_j]
                        emb_dist = cosine(emb[node_i], emb[node_j])  # Distance cosinus
                        
                        graph_distances.append(graph_dist)
                        embedding_distances.append(emb_dist)
            
            # Calculer la corrélation
            if len(graph_distances) > 0:
                correlation = np.corrcoef(graph_distances, embedding_distances)[0, 1]
                results[method_name] = {
                    'graph_embedding_correlation': correlation,
                    'sample_size': len(graph_distances)
                }
            else:
                results[method_name] = {
                    'graph_embedding_correlation': np.nan,
                    'sample_size': 0
                }
                
        return results
    
    def method_3_link_based_retrieval_evaluation(self, k=10):
        """
        Méthode 3: Évaluation de la récupération basée sur les liens
        
        Pour chaque document, vérifie si ses voisins dans le graphe 
        sont retrouvés dans ses k plus proches voisins dans l'embedding
        """
        results = {}
        
        for method_name, emb in self.embeddings.items():
            precision_scores = []
            recall_scores = []
            
            for node in self.graph.nodes():
                # Voisins réels dans le graphe
                true_neighbors = set(self.graph.neighbors(node))
                
                if len(true_neighbors) == 0:
                    continue
                
                # Calculer les similarités avec tous les autres documents
                similarities = []
                for other_node in self.graph.nodes():
                    if other_node != node:
                        sim = cosine_similarity([emb[node]], [emb[other_node]])[0][0]
                        similarities.append((other_node, sim))
                
                # Trier par similarité décroissante et prendre les k premiers
                similarities.sort(key=lambda x: x[1], reverse=True)
                retrieved_neighbors = set([item[0] for item in similarities[:k]])
                
                # Calculer précision et rappel
                intersection = true_neighbors.intersection(retrieved_neighbors)
                precision = len(intersection) / len(retrieved_neighbors) if len(retrieved_neighbors) > 0 else 0
                recall = len(intersection) / len(true_neighbors) if len(true_neighbors) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
            
            results[method_name] = {
                'mean_precision': np.mean(precision_scores),
                'mean_recall': np.mean(recall_scores),
                'f1_score': 2 * np.mean(precision_scores) * np.mean(recall_scores) / 
                          (np.mean(precision_scores) + np.mean(recall_scores)) 
                          if (np.mean(precision_scores) + np.mean(recall_scores)) > 0 else 0,
                'num_evaluated_nodes': len(precision_scores)
            }
            
        return results
    
    def method_4_semantic_coherence_clusters(self):
        """
        Méthode 4: Cohérence sémantique des clusters connectés
        
        Évalue la cohérence interne des composantes connexes du graphe
        """
        results = {}
        
        # Identifier les composantes connexes
        connected_components = list(nx.connected_components(self.graph))
        # Filtrer les composantes avec au moins 3 nœuds
        large_components = [comp for comp in connected_components if len(comp) >= 3]
        
        for method_name, emb in self.embeddings.items():
            coherence_scores = []
            
            for component in large_components:
                component_nodes = list(component)
                
                # Calculer toutes les similarités paires dans la composante
                similarities = []
                for i, node_i in enumerate(component_nodes):
                    for node_j in component_nodes[i+1:]:
                        sim = cosine_similarity([emb[node_i]], [emb[node_j]])[0][0]
                        similarities.append(sim)
                
                # Cohérence = similarité moyenne dans la composante
                if similarities:
                    coherence_scores.append(np.mean(similarities))
            
            results[method_name] = {
                'mean_cluster_coherence': np.mean(coherence_scores) if coherence_scores else 0,
                'std_cluster_coherence': np.std(coherence_scores) if coherence_scores else 0,
                'num_clusters_evaluated': len(coherence_scores),
                'cluster_sizes': [len(comp) for comp in large_components]
            }
            
        return results
    
    def method_5_fracture_severity_index(self, similarity_threshold=0.3):
        """
        Méthode 5: Indice de sévérité de la fracture sémantique
        
        Compte le pourcentage de liens où la similarité sémantique est 
        en dessous d'un seuil critique
        """
        results = {}
        linked_pairs = list(self.graph.edges())
        
        for method_name, emb in self.embeddings.items():
            fracture_count = 0
            total_links = len(linked_pairs)
            
            low_similarity_pairs = []
            
            for doc_i, doc_j in linked_pairs:
                sim = cosine_similarity([emb[doc_i]], [emb[doc_j]])[0][0]
                
                if sim < similarity_threshold:
                    fracture_count += 1
                    low_similarity_pairs.append((doc_i, doc_j, sim))
            
            fracture_severity = fracture_count / total_links if total_links > 0 else 0
            
            results[method_name] = {
                'fracture_severity_index': fracture_severity,
                'fractured_links_count': fracture_count,
                'total_links': total_links,
                'worst_fractures': sorted(low_similarity_pairs, key=lambda x: x[2])[:5]  # 5 pires cas
            }
            
        return results
    
    def comprehensive_evaluation(self):
        """
        Évaluation complète combinant toutes les méthodes
        """
        print("=== ÉVALUATION COMPLÈTE DE LA FRACTURE SÉMANTIQUE ===\n")
        
        # Méthode 1: Préservation des similarités
        print("1. PRÉSERVATION DE LA SIMILARITÉ POUR LES DOCUMENTS LIÉS")
        print("-" * 60)
        results_1 = self.method_1_linked_similarity_preservation()
        df_1 = pd.DataFrame(results_1).T
        print(df_1.round(4))
        print()
        
        # Méthode 2: Corrélation distance graphe/embedding
        print("2. CORRÉLATION DISTANCE GRAPHE / DISTANCE EMBEDDING")
        print("-" * 60)
        results_2 = self.method_2_link_distance_correlation()
        df_2 = pd.DataFrame(results_2).T
        print(df_2.round(4))
        print()
        
        # Méthode 3: Évaluation de récupération
        print("3. ÉVALUATION DE LA RÉCUPÉRATION BASÉE SUR LES LIENS")
        print("-" * 60)
        results_3 = self.method_3_link_based_retrieval_evaluation()
        df_3 = pd.DataFrame(results_3).T
        print(df_3.round(4))
        print()
        
        # Méthode 4: Cohérence des clusters
        print("4. COHÉRENCE SÉMANTIQUE DES CLUSTERS CONNECTÉS")
        print("-" * 60)
        results_4 = self.method_4_semantic_coherence_clusters()
        df_4 = pd.DataFrame(results_4).T
        print(df_4.round(4))
        print()
        
        # Méthode 5: Indice de sévérité
        print("5. INDICE DE SÉVÉRITÉ DE LA FRACTURE SÉMANTIQUE")
        print("-" * 60)
        results_5 = self.method_5_fracture_severity_index()
        df_5 = pd.DataFrame(results_5).T
        print(df_5.round(4))
        print()
        
        return {
            'linked_similarity': results_1,
            'distance_correlation': results_2,
            'retrieval_evaluation': results_3,
            'cluster_coherence': results_4,
            'fracture_severity': results_5
        }

# Exemple d'utilisation
def evaluate_semantic_fracture(embeddings, adjacency_matrix, raw_texts=None):
    """
    Fonction principale pour évaluer la fracture sémantique
    
    Args:
        embeddings: dict avec les embeddings par méthode
        adjacency_matrix: matrice d'adjacence du graphe
        raw_texts: textes bruts (optionnel)
    
    Returns:
        dict avec tous les résultats d'évaluation
    """
    evaluator = SemanticFractureEvaluator(embeddings, adjacency_matrix, raw_texts)
    return evaluator.comprehensive_evaluation()

# Intégration dans votre code principal


