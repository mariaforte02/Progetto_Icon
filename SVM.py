import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

class SVMPacketAnalysis:
    def __init__(self, data_path):
        try:
            print("Caricamento del dataset...")
            self.dataset = pd.read_csv(data_path)
            print("Dataset caricato con successo.")
            self.model = SVC(kernel='rbf', C=1, gamma='scale')  # Modificato per kernel RBF
        except FileNotFoundError:
            print(f"Errore: il file {data_path} non esiste.")
            raise
        except Exception as e:
            print(f"Errore durante il caricamento del dataset: {e}")
            raise

    def preprocess_data(self):
        try:
            print("Inizio preprocessamento...")

            # Codifica delle variabili categoriali
            label_encoders = {}
            categorical_columns = ["protocol_type", "service"]

            for col in categorical_columns:
                if col not in self.dataset.columns:
                    raise KeyError(f"Colonna {col} non trovata nel dataset.")

                le = LabelEncoder()
                self.dataset[col] = le.fit_transform(self.dataset[col])
                label_encoders[col] = le

            print("Codifica delle variabili categoriali completata.")

            # Standardizzazione delle variabili numeriche
            numeric_columns = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent"]

            for col in numeric_columns:
                if col not in self.dataset.columns:
                    raise KeyError(f"Colonna {col} non trovata nel dataset.")

            scaler = StandardScaler()
            self.dataset[numeric_columns] = scaler.fit_transform(self.dataset[numeric_columns])
            print("Standardizzazione delle variabili numeriche completata.")

            # Seleziona le feature (X) e il target (y)
            X = self.dataset[["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "protocol_type", "service"]]
            y = self.dataset["flag"]

            print("Preprocessamento completato con successo.")
            return X, y

        except Exception as e:
            print(f"Errore durante il preprocessamento: {e}")
            raise

    def train_model(self):
        try:
            print("Inizio addestramento SVM...")
            X, y = self.preprocess_data()

            # Suddivide i dati in training e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"Dati suddivisi: {len(X_train)} training, {len(X_test)} test.")

            # DEBUG: Riduci il numero di dati per testare più velocemente
            X_train = X_train[:5000]
            y_train = y_train[:5000]

            start_time = time.time()

            print("Avvio training SVM...")
            self.model.fit(X_train, y_train)

            end_time = time.time()
            print(f"Training completato in {end_time - start_time:.2f} secondi.")

            # Predizione e accuratezza
            print("Eseguo predizioni...")
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy * 100:.2f}%")

            # Messaggio finale per confermare che è terminato
            print("\nSVM completata. Premi Invio per tornare al menu principale...")
            input()  # Attende un input per consentire all'utente di vedere i risultati

        except Exception as e:
            print(f"Errore durante l'addestramento SVM: {e}")
            raise

# Esempio di utilizzo della classe SVMPacketAnalysis
if __name__ == "__main__":
    try:
        file_path = r"C:\\Users\\meryf\\OneDrive\\desktop\\PROGETTO_ICON\\Processed_Test_data_cleaned_unified.csv"
        svm_analysis = SVMPacketAnalysis(file_path)
        svm_analysis.train_model()
    except Exception as e:
        print(f"Errore nell'esecuzione: {e}")
