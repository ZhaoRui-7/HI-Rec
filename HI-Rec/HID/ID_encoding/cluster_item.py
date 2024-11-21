import json
import numpy as np
from sklearn.cluster import KMeans
import os

def read_json_file(file_path):
    items = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            items.append(item)
    return items

def extract_data(items):
    embeddings = []
    item_ids = []
    for item in items:
        embeddings.append(item['item_emb'])
        item_ids.append(item['item_id'])
    return np.array(embeddings), item_ids

def recursive_clustering(embeddings, item_ids, num_clusters, depth, result, prefix=''):
    if depth == 0:
        # If depth is 0 or not enough samples to cluster, stop recursion
        cluster_name = prefix if prefix else 'root'
        result[cluster_name] = item_ids
        return
    if len(embeddings) < num_clusters:
        num_clusters = len(embeddings)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_item_ids = [item_ids[idx] for idx in cluster_indices]
        
        cluster_name = f"{prefix}_{i}" if prefix else str(i)
        result[cluster_name] = [item_id[0] for item_id in cluster_item_ids]  # Save item_ids for this cluster
        
        # Recursive clustering
        recursive_clustering(cluster_embeddings, cluster_item_ids, num_clusters, depth - 1, result, cluster_name)

def bottom_up_clustering(embeddings, item_ids, initial_num_clusters, final_num_clusters, depth, result):
    current_embeddings = embeddings.copy()
    current_item_ids = item_ids.copy()
    cluster_names = ['cluster_{}'.format(i) for i in range(initial_num_clusters)]
    
    # Initial clustering
    kmeans = KMeans(n_clusters=initial_num_clusters, random_state=0).fit(current_embeddings)
    labels = kmeans.labels_
    
    # Update the embeddings and item_ids with cluster centers and their items
    for i in range(initial_num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_item_ids = [item_ids[idx] for idx in cluster_indices]
        result[cluster_names[i]] = [item_id[0] for item_id in cluster_item_ids]
        
        # Replace embeddings and item_ids with cluster center and its items
        current_embeddings = np.vstack((current_embeddings, [kmeans.cluster_centers_[i]]))
        current_item_ids.extend([[cluster_names[i]] for _ in range(len(cluster_indices))])
    
    # Remove duplicates from current_item_ids
    current_item_ids = list(set(map(tuple, current_item_ids)))
    
    # Perform clustering iteratively until reaching the final number of clusters
    while len(current_embeddings) > final_num_clusters:
        kmeans = KMeans(n_clusters=len(current_embeddings)//2, random_state=0).fit(current_embeddings)
        labels = kmeans.labels_
        
        new_embeddings = []
        new_item_ids = []
        new_cluster_names = []
        
        for i in range(len(current_embeddings)//2):
            cluster_indices = np.where(labels == i)[0]
            new_embeddings.append(kmeans.cluster_centers_[i])
            new_item_ids.extend([current_item_ids[idx] for idx in cluster_indices])
            new_cluster_names.append('cluster_{}'.format(len(new_cluster_names)))
            
            # Update result with current cluster information
            result[new_cluster_names] = [item[0][0] for item in new_item_ids[-len(cluster_indices):]]
        
        current_embeddings = np.array(new_embeddings)
        current_item_ids = new_item_ids
        
    # Final clustering to reach the desired number of clusters
    kmeans = KMeans(n_clusters=final_num_clusters, random_state=0).fit(current_embeddings)
    labels = kmeans.labels_
    
    # Update result with final cluster information
    for i in range(final_num_clusters):
        cluster_indices = np.where(labels == i)[0]
        result[f'final_cluster_{i}'] = [current_item_ids[idx][0] for idx in cluster_indices]

def save_results(result, output_file):
    with open(output_file, 'w') as f:
        json.dump(result, f)

# def main(input_file, output_file, initial_num_clusters=1024, final_num_clusters=4, depth=6):
#     items = read_json_file(input_file)
#     embeddings, item_ids = extract_data(items)
    
#     result = {}
#     bottom_up_clustering(embeddings, item_ids, initial_num_clusters, final_num_clusters, depth, result)
    
#     save_results(result, output_file)


def main(input_file, output_file, num_clusters=6, depth=5):
    items = read_json_file(input_file)
    embeddings, item_ids = extract_data(items)
    
    result = {}
    recursive_clustering(embeddings, item_ids, num_clusters, depth, result)
    
    save_results(result, output_file)

if __name__ == "__main__":
    input_file = input("input file:")  # Replace with your input file path
    output_file = 'output.json'  # Replace with your desired output file path
    main(input_file, output_file)


