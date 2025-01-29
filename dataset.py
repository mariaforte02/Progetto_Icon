import dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def get_dataset(file_path):
    """Carica il dataset e prepara i dati per l'elaborazione."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste nella directory.")
    
    print(f"Caricamento dataset da {file_path}...")
    dataset = pd.read_csv(file_path)
    print("Dataset caricato con successo!")

    # Rimozione dei record con valori nulli in colonne essenziali
    columns_to_check = ['protocol_type', 'service', 'src_bytes', 'dst_bytes', 'duration']
    dataset = dataset.dropna(subset=columns_to_check)
    print(f"Righe dopo la rimozione dei nulli: {dataset.shape[0]}")

    return dataset

def preprocess_data(dataset):
    print("Inizio del preprocessing...")

    # Rimozione dei duplicati
    dataset = dataset.drop_duplicates().copy()

    # Codifica delle variabili categoriche
    categorical_columns = ['protocol_type', 'service']
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        dataset.loc[:, col] = le.fit_transform(dataset[col].astype(str))
        label_encoders[col] = le

    # Standardizzazione delle variabili numeriche con conversione preventiva a float64
    numerical_columns = ['src_bytes', 'dst_bytes', 'duration']
    scaler = StandardScaler()
    dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns].astype(float)).astype(float)  # Converti prima e dopo in float

    # Aggiungere la colonna anomaly_probability
    dataset["anomaly_probability"] = 0.0  # Qui non serve usare loc

    return dataset




def save_dataset(dataset, output_path):
    """Salva il dataset pre-elaborato."""
    dataset.to_csv(output_path, index=False)
    print(f"Dataset pulito e pre-elaborato salvato in: {output_path}")

# Esecuzione del processo
if __name__ == "__main__":
    file_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Test_data.csv" 
    output_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"

    # Caricamento e preprocessing del dataset
    dataset = get_dataset(file_path)
    processed_dataset = preprocess_data(dataset)

    # Salvataggio del dataset finale
    save_dataset(processed_dataset, output_path)
