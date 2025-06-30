"""
AUC-Focused Node Embedding Benchmark
===================================

This benchmark evaluates node embedding models with a focus on AUC metrics
at different training data percentages (50% and 100%).

Models evaluated:
- TADW (Text-Associated DeepWalk)
- SINE (Scalable Incomplete Network Embedding)
- DeepWalk

Metrics:
- ROC-AUC for node classification
- Training data splits: 50% and 100%
"""

import networkx as nx
import numpy as np
import pandas as pd
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from karateclub import TADW, SINE, DeepWalk


class AUCFocusedBenchmark:
    """
    Benchmark focused on AUC metrics with different training data percentages
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        
    def load_cora_dataset(self):
        """Load and preprocess the Cora dataset"""
        print("Loading Cora dataset...")
        
        # Load graph - nodes are already integers in the file
        graph = nx.read_edgelist("dataset/cora/graph.txt", nodetype=int, create_using=nx.Graph())
        
        # Ensure connected graph
        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()
            
        # Load labels - nodes in group.txt match the node IDs in graph.txt
        labels_df = pd.read_csv('dataset/cora/group.txt', sep='\t', header=None, names=['node', 'label'])
        
        # Create label array aligned with nodes
        nodes = sorted(graph.nodes())
        labels = np.zeros(len(nodes), dtype=int)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for _, row in labels_df.iterrows():
            node_id = int(row['node'])
            if node_id in node_to_idx:
                labels[node_to_idx[node_id]] = int(row['label'])
                
        # Ensure graph nodes are consecutive integers starting from 0
        if set(nodes) != set(range(len(nodes))):
            mapping = {node: i for i, node in enumerate(nodes)}
            graph = nx.relabel_nodes(graph, mapping)
        
        print(f"Dataset loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Classes: {len(np.unique(labels))}")
        
        return graph, labels
    
    def create_node_features(self, graph):
        """Create node features for attributed embedding models with more meaningful graph-based features"""
        print("Creating node features...")
        num_nodes = graph.number_of_nodes()
        
        # Calculate various centrality measures
        print("  Calculating centrality measures...")
        degrees = dict(graph.degree())
        in_degrees = dict(nx.DiGraph(graph).in_degree()) if not nx.is_directed(graph) else dict(graph.in_degree())
        out_degrees = dict(nx.DiGraph(graph).out_degree()) if not nx.is_directed(graph) else dict(graph.out_degree())
        
        # Use try-except for potentially expensive calculations
        try:
            print("  Calculating clustering coefficients...")
            clustering = nx.clustering(graph)
        except:
            clustering = {node: 0.0 for node in graph.nodes()}
            
        try:
            print("  Calculating pagerank...")
            pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except:
            pagerank = {node: 1.0/num_nodes for node in graph.nodes()}
        
        # Calculate local features that are less expensive
        print("  Calculating neighborhood features...")
        neighborhood_features = {}
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighborhood_features[node] = {
                    'neighbor_count': len(neighbors),
                    'avg_neighbor_degree': np.mean([degrees[n] for n in neighbors]),
                    'max_neighbor_degree': max([degrees[n] for n in neighbors]) if neighbors else 0,
                }
            else:
                neighborhood_features[node] = {
                    'neighbor_count': 0,
                    'avg_neighbor_degree': 0,
                    'max_neighbor_degree': 0,
                }
        
        # Create feature matrix
        print("  Building feature matrix...")
        features = []
        for i in range(num_nodes):
            node_features = [
                # Basic centrality
                degrees[i],
                np.log1p(degrees[i]),
                in_degrees[i],
                out_degrees[i],
                
                # Clustering and PageRank
                clustering[i],
                pagerank[i],
                
                # Neighborhood features
                neighborhood_features[i]['neighbor_count'],
                neighborhood_features[i]['avg_neighbor_degree'],
                neighborhood_features[i]['max_neighbor_degree'],
                
                # Interaction terms
                degrees[i] * clustering[i],
                pagerank[i] * degrees[i],
            ]
            
            # Add some random features to reach required dimensions
            # but fewer than before since we have more meaningful features
            node_features.extend(np.random.rand(53).tolist())
            features.append(node_features)
        
        return np.array(features)
    
    def compute_tadw_embedding(self, graph, features, dimensions=64):
        """Compute TADW embedding with improved robustness"""
        print("Computing TADW embedding...")
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        
        try:
            # Ensure graph has at least one node and edge
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                print("Error: Graph has no nodes or edges")
                return None, None, None
                
            # Check if graph has self-loops and remove them if necessary
            self_loops = list(nx.selfloop_edges(graph))
            if self_loops:
                print(f"  Removing {len(self_loops)} self-loops from graph")
                graph = graph.copy()
                graph.remove_edges_from(self_loops)
            
            # Normalize features for better numerical stability
            print("  Normalizing features...")
            features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            # Configure TADW with robust parameters
            print("  Creating TADW model...")
            model = TADW(
                dimensions=dimensions,
                iterations=10,        # More iterations for better convergence
                lambd=0.2,            # Regularization parameter
                alpha=0.1             # Learning rate
            )
            
            print("  Fitting TADW model...")
            model.fit(graph, features_norm)
            
            print("  Extracting embeddings...")
            embedding = model.get_embedding()
            
            runtime = time.time() - start_time
            memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
            
            print(f"  TADW completed in {runtime:.2f} seconds")
            return embedding, runtime, memory_mb
        except Exception as e:
            print(f"TADW failed: {str(e)}")
            
            # Try with fallback parameters
            try:
                print("  Attempting TADW with fallback parameters...")
                # TADW is sensitive to dimensions, try with smaller dimensions
                fallback_dimensions = max(16, dimensions // 2)
                model = TADW(
                    dimensions=fallback_dimensions,
                    iterations=5,
                    lambd=0.1
                )
                model.fit(graph, features)
                embedding = model.get_embedding()
                
                runtime = time.time() - start_time
                memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
                
                print(f"  TADW completed with fallback parameters in {runtime:.2f} seconds")
                return embedding, runtime, memory_mb
            except Exception as e2:
                print(f"TADW fallback also failed: {str(e2)}")
                return None, None, None
    
    def compute_sine_embedding(self, graph, features, dimensions=64):
        """Compute SINE embedding with improved robustness"""
        print("Computing SINE embedding...")
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        
        try:
            # Ensure graph has at least one node and edge
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                print("Error: Graph has no nodes or edges")
                return None, None, None
                
            # Check if graph has self-loops and remove them if necessary
            self_loops = list(nx.selfloop_edges(graph))
            if self_loops:
                print(f"  Removing {len(self_loops)} self-loops from graph")
                graph = graph.copy()
                graph.remove_edges_from(self_loops)
            
            # Convert features to sparse format
            print("  Converting features to sparse format...")
            features_sparse = coo_matrix(features)
            
            # Configure SINE with faster parameters
            print("  Creating SINE model...")
            model = SINE(
                dimensions=dimensions,
                walk_length=20,      # Shorter walks for faster processing
                walk_number=5,       # Fewer walks
                window_size=3,       # Smaller context window
                epochs=3,            # Fewer training epochs
                seed=42              # For reproducibility
            )
            
            print("  Fitting SINE model...")
            model.fit(graph, features_sparse)
            
            print("  Extracting embeddings...")
            embedding = model.get_embedding()
            
            runtime = time.time() - start_time
            memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
            
            print(f"  SINE completed in {runtime:.2f} seconds")
            return embedding, runtime, memory_mb
        except Exception as e:
            print(f"SINE failed: {str(e)}")
            
            # Try with fallback parameters
            try:
                print("  Attempting SINE with fallback parameters...")
                model = SINE(
                    dimensions=dimensions,
                    walk_length=10,
                    walk_number=3,
                    window_size=2,
                    epochs=2
                )
                model.fit(graph, features_sparse)
                embedding = model.get_embedding()
                
                runtime = time.time() - start_time
                memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
                
                print(f"  SINE completed with fallback parameters in {runtime:.2f} seconds")
                return embedding, runtime, memory_mb
            except Exception as e2:
                print(f"SINE fallback also failed: {str(e2)}")
                return None, None, None
    
    def compute_deepwalk_embedding(self, graph, dimensions=64):
        """Compute DeepWalk embedding with improved robustness"""
        print("Computing DeepWalk embedding...")
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        
        try:
            # Ensure graph has at least one node and edge
            if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
                print("Error: Graph has no nodes or edges")
                return None, None, None
                
            # Check if graph has self-loops and remove them if necessary
            self_loops = list(nx.selfloop_edges(graph))
            if self_loops:
                print(f"  Removing {len(self_loops)} self-loops from graph")
                graph = graph.copy()
                graph.remove_edges_from(self_loops)
            
            # Configure DeepWalk with robust parameters
            model = DeepWalk(
                dimensions=dimensions,
                walk_length=80,       # Longer walks for better context
                window_size=10,       # Wider context window
                workers=4,            # Parallel processing
                epochs=10,            # More training epochs
                seed=42               # For reproducibility
            )
            
            print("  Fitting DeepWalk model...")
            model.fit(graph)
            
            print("  Extracting embeddings...")
            embedding = model.get_embedding()
            
            runtime = time.time() - start_time
            memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
            
            print(f"  DeepWalk completed in {runtime:.2f} seconds")
            return embedding, runtime, memory_mb
        except Exception as e:
            print(f"DeepWalk failed: {str(e)}")
            
            # Try with fallback parameters if the first attempt failed
            try:
                print("  Attempting DeepWalk with fallback parameters...")
                model = DeepWalk(
                    dimensions=dimensions,
                    walk_length=40,
                    window_size=5,
                    workers=1,
                    epochs=5
                )
                model.fit(graph)
                embedding = model.get_embedding()
                
                runtime = time.time() - start_time
                memory_mb = (psutil.Process(os.getpid()).memory_info().rss - start_mem) / 1e6
                
                print(f"  DeepWalk completed with fallback parameters in {runtime:.2f} seconds")
                return embedding, runtime, memory_mb
            except Exception as e2:
                print(f"DeepWalk fallback also failed: {str(e2)}")
                return None, None, None
    
    def evaluate_auc_at_training_percentage(self, embedding, labels, training_percentage, model_name):
        """Evaluate AUC at specific training data percentage"""
        if embedding is None:
            return None
        
        print(f"Evaluating {model_name} at {training_percentage}% training data...")
        
        # Encode labels if they're strings
        if labels.dtype == 'object':
            le = LabelEncoder()
            labels = le.fit_transform(labels)
        
        # Split data based on training percentage
        if training_percentage == 100:
            # Use all data for training and testing (training set performance)
            X_train, X_test = embedding, embedding
            y_train, y_test = labels, labels
        else:
            # Split data according to training percentage
            train_size = training_percentage / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                embedding, labels, 
                train_size=train_size, 
                stratify=labels, 
                random_state=self.random_state
            )
        
        # Scale features for better convergence
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train classifier with increased max_iter and appropriate regularization
        clf = LogisticRegression(
            random_state=self.random_state, 
            max_iter=2000,
            C=1.0,  # Regularization strength (inverse)
            solver='lbfgs',
            n_jobs=-1  # Use all available CPUs
        )
        clf.fit(X_train_scaled, y_train)
        
        # Get predictions
        y_prob = clf.predict_proba(X_test_scaled)
        
        # Calculate AUC
        try:
            if len(np.unique(labels)) == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            auc = np.nan
        
        return auc
    
    def run_benchmark(self, dimensions=64):
        """Run the complete AUC-focused benchmark"""
        print("Starting AUC-Focused Benchmark")
        print("="*50)
        
        # Load dataset
        graph, labels = self.load_cora_dataset()
        features = self.create_node_features(graph)
        
        # Define models
        models = [
            ("TADW", lambda g, f: self.compute_tadw_embedding(g, f, dimensions)),
            ("SINE", lambda g, f: self.compute_sine_embedding(g, f, dimensions)),
            ("DeepWalk", lambda g, f: self.compute_deepwalk_embedding(g, dimensions))
        ]
        
        # Define training percentages
        training_percentages = [50, 100]
        
        # Run benchmark for each model and training percentage
        for model_name, embedding_func in models:
            print(f"\n{'-'*30}")
            print(f"Model: {model_name}")
            print(f"{'-'*30}")
            
            # Compute embedding
            if model_name == "DeepWalk":
                embedding, runtime, memory_mb = embedding_func(graph)
            else:
                embedding, runtime, memory_mb = embedding_func(graph, features)
            
            if embedding is None:
                print(f"Skipping {model_name} due to embedding failure")
                continue
            
            # Evaluate at different training percentages
            for train_pct in training_percentages:
                auc = self.evaluate_auc_at_training_percentage(
                    embedding, labels, train_pct, model_name
                )
                
                result = {
                    'model': model_name,
                    'training_percentage': train_pct,
                    'auc': auc,
                    'embedding_runtime': runtime,
                    'embedding_memory_mb': memory_mb,
                    'dimensions': dimensions,
                    'nodes': graph.number_of_nodes(),
                    'edges': graph.number_of_edges(),
                    'classes': len(np.unique(labels))
                }
                
                self.results.append(result)
                print(f"  {train_pct}% training: AUC = {auc:.4f}")
        
        return self.results
    
    def save_results(self, filename="auc_benchmark_results.csv"):
        """Save results to CSV"""
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to {filename}")
            return results_df
        else:
            print("No results to save")
            return None
    
    def display_results(self):
        """Display results in a formatted table"""
        if not self.results:
            print("No results to display")
            return
        
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*80)
        print("AUC-FOCUSED BENCHMARK RESULTS")
        print("="*80)
        
        # Create pivot table for better display
        pivot_df = results_df.pivot(index='model', columns='training_percentage', values='auc')
        pivot_df.columns = [f'{col}% Training' for col in pivot_df.columns]
        
        # Add performance metrics
        runtime_df = results_df.pivot(index='model', columns='training_percentage', values='embedding_runtime')
        memory_df = results_df.pivot(index='model', columns='training_percentage', values='embedding_memory_mb')
        
        # Get the best model by AUC
        best_model_50 = pivot_df['50% Training'].idxmax() if '50% Training' in pivot_df.columns else None
        best_model_100 = pivot_df['100% Training'].idxmax() if '100% Training' in pivot_df.columns else None
        
        print("\nAUC Scores by Model and Training Percentage:")
        print("-" * 60)
        print(pivot_df.round(4).to_string())
        
        if best_model_50:
            print(f"\nBest model at 50% training data: {best_model_50} (AUC: {pivot_df.loc[best_model_50, '50% Training']:.4f})")
        if best_model_100:
            print(f"Best model at 100% training data: {best_model_100} (AUC: {pivot_df.loc[best_model_100, '100% Training']:.4f})")
        
        print("\nPerformance Metrics:")
        print("-" * 60)
        
        # Create a combined metrics dataframe
        metrics_df = pd.DataFrame(index=pivot_df.index)
        metrics_df['AUC (50%)'] = pivot_df['50% Training'] if '50% Training' in pivot_df.columns else None
        metrics_df['AUC (100%)'] = pivot_df['100% Training'] if '100% Training' in pivot_df.columns else None
        metrics_df['Runtime (s)'] = runtime_df.iloc[:, 0] if not runtime_df.empty else None
        metrics_df['Memory (MB)'] = memory_df.iloc[:, 0] if not memory_df.empty else None
        
        print(metrics_df.round(4).to_string())
        
        print("\nDetailed Results:")
        print("-" * 60)
        display_cols = ['model', 'training_percentage', 'auc', 'embedding_runtime', 'embedding_memory_mb', 'dimensions', 'nodes', 'edges']
        print(results_df[display_cols].round(4).to_string(index=False))


def main():
    """Run the AUC-focused benchmark"""
    benchmark = AUCFocusedBenchmark()
    
    # Run benchmark
    results = benchmark.run_benchmark(dimensions=64)
    
    # Display and save results
    benchmark.display_results()
    results_df = benchmark.save_results()
    
    return results_df


if __name__ == "__main__":
    main() 