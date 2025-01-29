import pandas as pd
from owlready2 import *

def create_rdf(csv_file_path):
    # Crea l'ontologia
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
    dataset = pd.read_csv(csv_file_path)

    # Aggiunta delle colonne mancanti
    dataset["ConnectionID"] = dataset.index  # Genera un identificativo univoco
    dataset["anomaly_probability"] = 0.0

    # Controllo delle colonne richieste
    required_columns = ['ConnectionID', 'protocol_type', 'duration', 'src_bytes', 'dst_bytes', 'AnomalyProbability']
    missing_columns = [col for col in required_columns if col not in dataset.columns]

    if missing_columns:
        raise ValueError(f"Il file CSV manca delle seguenti colonne richieste: {missing_columns}")

    # Popolamento delle classi
    with onto:
        for _, row in dataset.iterrows():
            # Crea un'istanza di Connection
            connection = Connection(f"Connection_{row['ConnectionID']}")
            connection.src_bytes = [row['src_bytes']]
            connection.dst_bytes = [row['dst_bytes']]
            connection.duration = [row['duration']]

            # Crea un'istanza di Protocol
            protocol = Protocol(f"Protocol_{row['protocol_type']}")
            protocol.protocol_type = [row['protocol_type']]

            # Collega la connessione al protocollo
            connection.uses_protocol = [protocol]

            # Crea un'istanza di Anomaly
            anomaly = Anomaly(f"Anomaly_{row['ConnectionID']}")
            anomaly.anomaly_probability = [row['AnomalyProbability']]

            # Collega la connessione all'anomalia
            connection.is_an_anomaly = [anomaly]

    # Salva l'ontologia in formato RDF
    onto.save(file="network_intrusion_ontology.rdf", format="rdfxml")
    print("Ontologia salvata con successo come network_intrusion_ontology.rdf")

    return onto

# Percorso del file CSV
csv_file_path = "C:/Users/meryf/OneDrive/desktop/PROGETTO_ICON/Processed_Test_data_cleaned_unified.csv"

# Creazione dell'ontologia e salvataggio come RDF
create_rdf(csv_file_path)
