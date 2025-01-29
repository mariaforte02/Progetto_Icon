import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator  # Libreria per trovare il "gomito" nel grafico Elbow

# Classe per l'apprendimento non supervisionato con K-Means per il dataset di rete
class UnsupervisedLearning:
    def __init__(self, file_path):
        self.file_path = file_path

    def find_optimal_clusters(self, dataset):
        # Calcola l'inertia per diversi valori di k (numero di cluster)
        inertia = []
        K_range = range(1, 10)  # Valori da 1 a 10 cluster
        for k in K_range:
            kmeans = KMeans(n_clusters=k, n_init=10, init='k-means++', random_state=42)
            kmeans.fit(dataset)
            inertia.append(kmeans.inertia_)

        # Usa kneed per trovare il "gomito" (elbow) nel grafico
        kneedle = KneeLocator(K_range, inertia, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        # Visualizza il grafico Elbow per scegliere il numero ottimale di cluster
        plt.figure(figsize=(10, 8))
        plt.plot(K_range, inertia, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--')
        plt.xlabel('Numero di Cluster')
        plt.ylabel('Inertia')
        plt.title('Grafico Elbow per trovare il numero ottimale di cluster')
        plt.show()

        print(f"Numero ottimale di cluster: {optimal_k}")
        return optimal_k

    def perform_kmeans_clustering(self):
        # Carica il dataset
        df = pd.read_csv(self.file_path)

        # Seleziona solo le colonne numeriche rilevanti per l'analisi
        features = df[['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent']]

        # Standardizzazione dei dati
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Trova il numero ottimale di cluster (grafico gomito)
        optimal_k = self.find_optimal_clusters(features_scaled)

        # Esegui K-Means con il numero ottimale di cluster
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features_scaled)

        # Ottieni i centroidi dei cluster
        centroids = kmeans.cluster_centers_

        # Mostra i centroidi con le feature originali (in scala standardizzata)
        print("\nValori dei centroidi per ogni cluster (standardizzati):")
        print(centroids)

        # Per facilitare la lettura, deseleziona la standardizzazione
        centroids_unscaled = scaler.inverse_transform(centroids)
        print("\nValori dei centroidi per ogni cluster (in valori originali):")
        print(pd.DataFrame(centroids_unscaled, columns=features.columns))

        # Mostra il numero di connessioni in ciascun cluster
        cluster_counts = df['Cluster'].value_counts()

        # Visualizza i cluster assegnati
        print("\nNumero di connessioni in ciascun cluster:")
        print(cluster_counts)

