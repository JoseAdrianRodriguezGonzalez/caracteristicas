import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_samples,silhouette_score
from sklearn.preprocessing import label_binarize
def kmeans_train(Dataset:pd.DataFrame):
    kmeans=KMeans(n_clusters=2,random_state=42)
    kmeans.fit(Dataset)
    labels=kmeans.labels_
    unique,count=np.unique(labels,return_counts=True)
    for u,c in zip(unique,count):
        print(f"CLuster {u}:{c}")
    return kmeans.cluster_centers_,labels,kmeans
def calculate_score(Dataset:pd.DataFrame,kmeans_model):
    all_distances=kmeans_model.transform(Dataset)
    labels=kmeans_model.labels_
    assigned_distances = all_distances[np.arange(len(labels)), labels]
    distances_0=assigned_distances[labels==0]
    distances_1=assigned_distances[labels==1]
    print("Distancia promedio cluster 0:", np.mean(distances_0))
    print("Distancia promedio cluster 1:", np.mean(distances_1))
    return distances_0,distances_1

def distancia_cohesion(Dataset:pd.DataFrame,kmeans_model):
    labels = kmeans_model.labels_
    all_distances = kmeans_model.transform(Dataset)
    assigned_distances = all_distances[np.arange(len(labels)), labels]
    metricas={}
    for k in range(kmeans_model.n_clusters):
        cluster_mask = (labels == k)
        cluster_distances = assigned_distances[cluster_mask]
        
        metricas[k] = {
            "mean": np.mean(cluster_distances),
            "max": np.max(cluster_distances),
            "min": np.min(cluster_distances),
            "std": np.std(cluster_distances),
            "all_distances": cluster_distances,
            "n_points": np.sum(cluster_mask)
        }
    return metricas
def Inertia(kmeans):
    return kmeans.inertia_ 

def get_cluster_distances(kmeans_model, Dataset):
    # Distancias a todos los centroides
    all_distances = kmeans_model.transform(Dataset)
    labels=kmeans_model.labels_
    # Distancia al centroide asignado (mínima)
    assigned_distances = all_distances[np.arange(len(all_distances)), labels]
    second_distances = []
    for i, label in enumerate(labels):
        # Excluimos el centroide asignado
        other_centroids = np.delete(all_distances[i], label)
        second_distances.append(np.min(other_centroids))
    second_distances = np.array(second_distances)
    confidence = 1 - (assigned_distances / second_distances)

    return confidence,all_distances

def kmeans_metrics(Dataset: pd.DataFrame):
    """Función principal mejorada"""
    centroids, labels, kmeans = kmeans_train(Dataset)
    silhouette = silhouette_score(Dataset, labels)
    inercia = Inertia(kmeans)
    cohesion_metrics = distancia_cohesion(Dataset, kmeans)
    dist_cluster_0, dist_cluster_1 = calculate_score(Dataset, kmeans)
    confidence,all_distances = get_cluster_distances(kmeans, Dataset)
    
    return {
        "cohesion": cohesion_metrics,
        "punto_centro": (np.mean(dist_cluster_0), np.mean(dist_cluster_1)),
        "silueta": silhouette,
        "inercia": inercia, # <---- aquí está la métrica nueva
        "all_distances": all_distances,
        "model": kmeans,
        "confianza":confidence,
        "confianza_promedio":np.mean(confidence)*100,
        "labels":labels
    }
