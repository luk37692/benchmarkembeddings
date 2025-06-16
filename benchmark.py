"""
This code benchmarks the performance of embedding models
models
- tadw
- sine
- deepwalk


We will use the following datasets:
- cora
- dblp
- nyt

We will use the following metrics:
- accuracy
- f1 score
- precision
- recall
- roc auc score


using Karate Club library models







"""

import networkx as nx
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import psutil
import os

from karateclub import TADW, SINE, DeepWalk


def tadw_embedding(graph):
    """
    """
    # Use synthetic features with more dimensions
    num_nodes = graph.number_of_nodes()
    # Create features with degree, clustering coefficient, and random features
    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)
    
    # Create feature matrix with sufficient dimensions (minimum 64 for TADW with 64 dimensions)
    features = []
    for i in range(num_nodes):
        node_features = [
            degrees[i], 
            degrees[i]**2, 
            clustering[i],
            degrees[i] * clustering[i]
        ]
        # Add random features to reach 64 dimensions
        node_features.extend(np.random.rand(60).tolist())
        features.append(node_features)
    
    features = np.array(features)
    tadw = TADW(dimensions=32, iterations=5)  # Reduced dimensions
    tadw.fit(graph, features)
    return tadw.get_embedding()


def sine_embedding(graph):
    """
    """
    from scipy.sparse import coo_matrix
    
    # Use synthetic features in sparse format
    num_nodes = graph.number_of_nodes()
    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)
    
    # Create feature matrix
    features = []
    for i in range(num_nodes):
        node_features = [
            degrees[i], 
            degrees[i]**2, 
            clustering[i],
            degrees[i] * clustering[i]
        ]
        # Add random features to reach decent dimensions
        node_features.extend(np.random.rand(60).tolist())
        features.append(node_features)
    
    features = np.array(features)
    # Convert to sparse matrix format
    features_sparse = coo_matrix(features)
    
    sine = SINE(dimensions=32, epochs=1)  # Reduced dimensions
    sine.fit(graph, features_sparse)
    return sine.get_embedding()


def deepwalk_embedding(graph):
    """
    """
    deepwalk = DeepWalk(dimensions=32, workers=4, epochs=1)  # Reduced to match other models
    deepwalk.fit(graph)
    return deepwalk.get_embedding()


def load_dataset(dataset_name):
    """
    """
    if dataset_name == "cora":
        graph = nx.read_edgelist("dataset/cora/graph.txt", create_using=nx.Graph())
        # Relabel nodes to consecutive integers starting from 0
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        graph = nx.relabel_nodes(graph, mapping)
        return graph
    elif dataset_name == "dblp":
        graph = nx.read_edgelist("dataset/dblp/graph.txt", create_using=nx.Graph())
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        graph = nx.relabel_nodes(graph, mapping)
        return graph
    elif dataset_name == "nyt":
        graph = nx.read_edgelist("dataset/nyt/graph.txt", create_using=nx.Graph())
        mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
        graph = nx.relabel_nodes(graph, mapping)
        return graph


def load_node_labels(dataset_name):
    """
    Load node labels for evaluation (assumes labels files exist)
    """
    try:
        if dataset_name == "cora":
            labels_df = pd.read_csv("dataset/cora/group.txt", sep='\t', header=None, names=['node', 'label'])
            # Map original node IDs to new consecutive IDs
            graph = nx.read_edgelist("dataset/cora/graph.txt", create_using=nx.Graph())
            mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
            labels_df['node'] = labels_df['node'].map(mapping)
            labels_df = labels_df.dropna()  # Remove unmapped nodes
            return labels_df
        elif dataset_name == "dblp":
            labels_df = pd.read_csv("dataset/dblp/group.txt", sep='\t', header=None, names=['node', 'label'])
            graph = nx.read_edgelist("dataset/dblp/graph.txt", create_using=nx.Graph())
            mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
            labels_df['node'] = labels_df['node'].map(mapping)
            labels_df = labels_df.dropna()
            return labels_df
        elif dataset_name == "nyt":
            labels_df = pd.read_csv("dataset/nyt/group.txt", sep='\t', header=None, names=['node', 'label'])
            graph = nx.read_edgelist("dataset/nyt/graph.txt", create_using=nx.Graph())
            mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
            labels_df['node'] = labels_df['node'].map(mapping)
            labels_df = labels_df.dropna()
            return labels_df
    except FileNotFoundError:
        print(f"Warning: No labels file found for {dataset_name}. Generating synthetic labels.")
        # Generate synthetic labels for demonstration
        graph = load_dataset(dataset_name)
        nodes = list(graph.nodes())
        labels = np.random.randint(0, 5, len(nodes))  # 5 classes
        return pd.DataFrame({'node': nodes, 'label': labels})


def evaluate_embedding(embeddings, labels_df, nodes, n_splits=5):
    """
    Evaluate embeddings using node classification
    """
    # Ensure we have the right number of embeddings for the nodes
    if len(embeddings) != len(nodes):
        print(f"Mismatch: {len(embeddings)} embeddings vs {len(nodes)} nodes")
        return None
        
    # Filter labels to only include nodes present in current graph
    valid_labels = labels_df[labels_df['node'] < len(nodes)].copy()
    
    if len(valid_labels) < 10:
        print("Warning: Not enough labeled nodes for evaluation")
        return None
    
    # Get embeddings and labels for valid nodes
    X = embeddings[valid_labels['node'].values]
    y = valid_labels['label'].values
    
    if len(X) < 10:
        print("Warning: Not enough matching nodes for evaluation")
        return None
        
    # Encode labels if they're strings
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc, f1_macro, f1_micro, f1_weighted = [], [], [], []
    prec, rec, roc, train_times = [], [], [], []
    mem_usage = []
    process = psutil.Process(os.getpid())

    for train_idx, test_idx in skf.split(X, y):
        start_time = time.time()
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        train_times.append(time.time() - start_time)
        mem_usage.append(process.memory_info().rss)

        y_pred = clf.predict(X[test_idx])
        y_prob = clf.predict_proba(X[test_idx])

        acc.append(accuracy_score(y[test_idx], y_pred))
        f1_macro.append(f1_score(y[test_idx], y_pred, average='macro'))
        f1_micro.append(f1_score(y[test_idx], y_pred, average='micro'))
        f1_weighted.append(f1_score(y[test_idx], y_pred, average='weighted'))
        prec.append(precision_score(y[test_idx], y_pred, average='weighted'))
        rec.append(recall_score(y[test_idx], y_pred, average='weighted'))

        try:
            if len(np.unique(y)) == 2:
                roc.append(roc_auc_score(y[test_idx], y_prob[:, 1]))
            else:
                roc.append(roc_auc_score(y[test_idx], y_prob, multi_class='ovr'))
        except Exception:
            roc.append(np.nan)

    metrics = {
        'accuracy': float(np.mean(acc)),
        'f1_macro': float(np.mean(f1_macro)),
        'f1_micro': float(np.mean(f1_micro)),
        'f1_weighted': float(np.mean(f1_weighted)),
        'precision': float(np.mean(prec)),
        'recall': float(np.mean(rec)),
        'roc_auc': float(np.nanmean(roc)),
        'train_time': float(np.sum(train_times)),
        'train_memory_mb': float(max(mem_usage) / 1e6)
    }

    return metrics


def benchmark_model(model_name, embedding_func, graph, labels_df):
    """
    Benchmark a single model
    """
    print(f"\nBenchmarking {model_name}...")
    
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem = process.memory_info().rss
    try:
        embeddings = embedding_func(graph)
        embedding_time = time.time() - start_time
        embedding_mem = (process.memory_info().rss - start_mem) / 1e6
        
        # Get node list (assuming embeddings are in node order)
        nodes = list(graph.nodes())
        
        # Evaluate embeddings
        metrics = evaluate_embedding(embeddings, labels_df, nodes)
        
        result = {
            'model': model_name,
            'embedding_time': embedding_time,
            'embedding_memory_mb': embedding_mem,
            'num_nodes': len(nodes),
            'num_edges': graph.number_of_edges()
        }
        
        if metrics:
            result.update(metrics)
        else:
            print(f"Could not evaluate {model_name} - insufficient labeled data")
            
        return result
        
    except Exception as e:
        print(f"Error with {model_name}: {str(e)}")
        return {
            'model': model_name,
            'error': str(e),
            'embedding_time': None,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges()
        }


def main():
    """
    Run complete benchmark across all models and datasets
    """
    # Datasets to benchmark
    datasets = ["cora", "dblp", "nyt"]
    models = [
        ("TADW", tadw_embedding),
        ("SINE", sine_embedding),
        ("DeepWalk", deepwalk_embedding)
    ]
    
    all_results = []
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Load dataset
            graph = load_dataset(dataset_name)
            labels_df = load_node_labels(dataset_name)
            
            print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
            
            # Ensure graph is connected (take largest component)
            if not nx.is_connected(graph):
                largest_cc = max(nx.connected_components(graph), key=len)
                graph = graph.subgraph(largest_cc).copy()
                print(f"Using largest connected component: {graph.number_of_nodes()} nodes")
                
                # Re-label nodes to be consecutive starting from 0
                node_mapping = {node: i for i, node in enumerate(sorted(graph.nodes()))}
                graph = nx.relabel_nodes(graph, node_mapping)
                
                # Update labels to match new node indexing
                labels_df = labels_df[labels_df['node'].isin(node_mapping.keys())]
                labels_df['node'] = labels_df['node'].map(node_mapping)
                labels_df = labels_df.dropna()
            
            # Benchmark each model
            for model_name, embedding_func in models:
                result = benchmark_model(model_name, embedding_func, graph, labels_df)
                result['dataset'] = dataset_name
                all_results.append(result)
                
        except FileNotFoundError as e:
            print(f"Dataset {dataset_name} not found: {e}")
            continue
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
            continue
    
    # Create results DataFrame and display
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*80}")
        print("BENCHMARK RESULTS SUMMARY")
        print(f"{'='*80}")
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('benchmark_results.csv', index=False)
        print(f"\nResults saved to benchmark_results.csv")
    else:
        print("No results to display. Please check that dataset files exist.")


if __name__ == "__main__":
    main()