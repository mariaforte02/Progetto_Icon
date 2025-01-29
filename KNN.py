import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class KNNIntrusionDetection:
    def __init__(self, dataset):
        # Controlla che il dataset sia un DataFrame
        if isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            raise ValueError("Il dataset deve essere un DataFrame di Pandas.")

        # Inizializza il modello KNN con k=3
        self.model = KNeighborsClassifier(n_neighbors=3)

    def preprocess_data(self):
        # Codifica variabili categoriali (ad esempio, protocol_type, service)
        label_encoders = {}
        categorical_columns = ["protocol_type", "service"]
        
        for col in categorical_columns:
            le = LabelEncoder()
            self.dataset[col] = le.fit_transform(self.dataset[col])
            label_encoders[col] = le
        
        # Normalizza le variabili numeriche
        scaler = StandardScaler()
        numeric_columns = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent"]
        self.dataset[numeric_columns] = scaler.fit_transform(self.dataset[numeric_columns])
        
        # Seleziona le colonne desiderate per le feature (X)
        X = self.dataset[["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "protocol_type", "service"]]
        
        # Definisci la colonna target (y)
        y = self.dataset["flag"]  # La variabile target rappresenta lo stato della connessione
        
        return X, y

    def train_model(self):
        X, y = self.preprocess_data()
        
        # Suddividi i dati in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Addestra il modello
        self.model.fit(X_train, y_train)
        
        # Fai previsioni e calcola l'accuratezza
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Accuracy:", accuracy)

# Carica il tuo dataset (sostituisci il percorso con quello corretto)
dataset_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\ProgettoIcon\Processed_Test_data_cleaned_unified.csv"
dataset = pd.read_csv(dataset_path)


# Crea un'istanza di KNNIntrusionDetection e addestra il modello
knn_intrusion = KNNIntrusionDetection(dataset)
knn_intrusion.train_model()
