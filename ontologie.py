import os
import pandas as pd
from owlready2 import *
import dataset as ds

def load_onto():
    file_path = "./archivio/network_intrusion_ontology.rdf"
    if not os.path.exists(file_path):
        print("File non trovato. Creazione di una nuova ontologia...")
        return create_ontology()
    else:
        return get_ontology(file_path).load()


def create_ontology():
    # Crea un'ontologia vuota
    onto = get_ontology("http://example.org/network_intrusion_ontology")

    with onto:
        # Classi per l'ontologia
        class Connection(Thing):
            pass

        class Protocol(Thing):
            pass

        class Anomaly(Thing):
            pass

        # Proprietà
        class uses_protocol(ObjectProperty):
            domain = [Connection]
            range = [Protocol]

        class is_an_anomaly(ObjectProperty):
            domain = [Connection]
            range = [Anomaly]

        class src_bytes(DataProperty):
            domain = [Connection]
            range = [int]

        class dst_bytes(DataProperty):
            domain = [Connection]
            range = [int]

        class duration(DataProperty):
            domain = [Connection]
            range = [float]

        class protocol_type(DataProperty):
            domain = [Protocol]
            range = [str]

        class anomaly_probability(DataProperty):
            domain = [Anomaly]
            range = [float]

    # Carica il dataset
    file_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"
    dataset = ds.get_dataset(file_path)

    # Aggiunta delle colonne mancanti
    if "connection_id" not in dataset.columns:
        dataset.insert(0, "connection_id", dataset.index)
    dataset["anomaly_probability"] = 0.0

    print("Colonne nel dataset dopo l'aggiunta di 'connection_id':", dataset.columns.tolist())

    # Controllo colonne richieste
    required_columns = ['connection_id', 'protocol_type', 'duration', 'src_bytes', 'dst_bytes', 'anomaly_probability']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Il file CSV manca delle seguenti colonne richieste: {missing_columns}")

    # Popolamento delle classi
    with onto:
        for _, row in dataset.iterrows():
            # Crea un'istanza di Connection
            connection = Connection(f"Connection_{row['connection_id']}")
            connection.src_bytes = [row['src_bytes']]
            connection.dst_bytes = [row['dst_bytes']]
            connection.duration = [row['duration']]

            # Crea un'istanza di Protocol
            protocol = Protocol(f"Protocol_{row['protocol_type']}")
            protocol.protocol_type = [row['protocol_type']]

            # Collega la connessione al protocollo
            connection.uses_protocol = [protocol]

            # Crea un'istanza di Anomaly
            anomaly = Anomaly(f"Anomaly_{row['connection_id']}")
            anomaly.anomaly_probability = [row['anomaly_probability']]

            # Collega la connessione all'anomalia
            connection.is_an_anomaly = [anomaly]

    # Crea la directory "archivio" se non esiste
    os.makedirs("./archivio", exist_ok=True)

    # Salva l'ontologia in formato OWL
    onto.save(file=r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\archivio\network_intrusion_ontology.rdf", format="rdfxml")
    print("Ontologia salvata con successo come network_intrusion_ontology.rdf")
    return onto

# Query 1: Connessioni con un numero elevato di byte inviati
def connections_high_src_bytes(byte_threshold, csv_file=r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"):
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(csv_file)

    # Verifica i nomi delle colonne nel dataset
    print("Colonne presenti nel dataset:", df.columns.tolist())

    # Controlla se connection_id esiste nel dataset
    if "connection_id" not in df.columns:
        df.insert(0, "connection_id", df.index)

    # Filtra i dati per i byte inviati superiori alla soglia
    high_src_bytes = df[df['src_bytes'] >= byte_threshold]

    # Visualizza le connessioni
    if high_src_bytes.empty:
        print(f"Nessuna connessione trovata con byte inviati superiori a {byte_threshold}.")
    else:
        print(f"Connessioni con byte inviati superiori a {byte_threshold}:")
        for _, row in high_src_bytes.iterrows():
            print(f"ID Connessione: {row['connection_id']}, Protocollo: {row['protocol_type']}, "
                  f"Durata: {row['duration']}, Byte inviati: {row['src_bytes']}, "
                  f"Byte ricevuti: {row['dst_bytes']}, Probabilità di anomalia: {row['anomaly_probability']}")

# Query 2: Connessioni con un numero basso di byte inviati
def connections_low_src_bytes(byte_threshold, csv_file=r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"):
    # Carica il file CSV in un DataFrame
    df = pd.read_csv(csv_file)

    # Controlla se connection_id esiste nel dataset
    if "connection_id" not in df.columns:
        df.insert(0, "connection_id", df.index)

    # Filtra i dati per i byte inviati inferiori alla soglia
    low_src_bytes = df[df['src_bytes'] <= byte_threshold]

    # Visualizza le connessioni
    if low_src_bytes.empty:
        print(f"Nessuna connessione trovata con byte inviati inferiori a {byte_threshold}.")
    else:
        print(f"Connessioni con byte inviati inferiori a {byte_threshold}:")
        for _, row in low_src_bytes.iterrows():
            print(f"ID Connessione: {row['connection_id']}, Protocollo: {row['protocol_type']}, "
                  f"Durata: {row['duration']}, Byte inviati: {row['src_bytes']}, "
                  f"Byte ricevuti: {row['dst_bytes']}, Probabilità di anomalia: {row['anomaly_probability']}")
