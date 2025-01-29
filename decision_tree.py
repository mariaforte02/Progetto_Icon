import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import accuracy_score

# Dizionario degli iperparametri per la ricerca ottimale
DecisionTreeHyperparameters = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
}

class DecisionTree:
    def __init__(self, file_path, hyperparameters=None):
        # Carica il dataset
        self.dataset = pd.read_csv(file_path)

        if not isinstance(self.dataset, pd.DataFrame):
            raise ValueError("Il dataset deve essere un DataFrame di Pandas.")

        # Stampa la distribuzione delle classi
        print("Distribuzione delle classi nel dataset (colonna 'flag'):")
        print(self.dataset['flag'].value_counts())  # La colonna target è 'flag'

        # Filtra classi con almeno 2 campioni
        class_counts = self.dataset['flag'].value_counts()
        classes_to_keep = class_counts[class_counts >= 2].index
        self.dataset = self.dataset[self.dataset['flag'].isin(classes_to_keep)]

        if hyperparameters is None:
            hyperparameters = {}

        self.model = DecisionTreeClassifier(**hyperparameters)

    def preprocess_data(self):
        # Seleziona caratteristiche e target
        X = self.dataset[['duration', 'src_bytes', 'dst_bytes']]  # Caratteristiche principali
        y = self.dataset['flag']  # Variabile target
        return X, y

    def optimize_hyperparameters(self):
        X, y = self.preprocess_data()

        # Suddivide i dati in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Ricerca degli iperparametri
        grid_search = GridSearchCV(estimator=DecisionTreeClassifier(),
                                   param_grid=DecisionTreeHyperparameters,
                                   cv=3,
                                   scoring='accuracy',
                                   return_train_score=True)

        grid_search.fit(X_train, y_train)

        # Mostra i migliori parametri
        results = pd.DataFrame(grid_search.cv_results_)
        print("Tabella degli iperparametri ottimali:")
        print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

        self.plot_best_params_table(results)

        return grid_search.best_params_

    def plot_best_params_table(self, results):
        # Filtra i migliori parametri
        best_params = results.loc[results['rank_test_score'] == 1].iloc[0]

        # Prepara i dati per la tabella
        params = {
            'Iperparametro': ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
            'Valore': [best_params['params'][key] for key in best_params['params']]
        }

        params_df = pd.DataFrame(params)

        # Crea una tabella per visualizzare i migliori iperparametri
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=params_df.values,
                         colLabels=params_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.title("Migliori Iperparametri del Decision Tree", fontsize=14)
        plt.show()

    def train_model(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Addestra il modello
        self.model.fit(X_train, y_train)

        # Previsioni e accuratezza
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

    def plot_decision_tree(self):
        if self.model is None:
            raise ValueError("Il modello non è stato ancora addestrato.")

        _, y = self.preprocess_data()
        if len(y.unique()) <= 1:
            raise ValueError("Il target ha solo una classe, non è possibile visualizzare l'albero decisionale.")

        max_depth = 3
        plt.figure(figsize=(20, 12))

        # Visualizzazione dell'albero decisionale
        plot_tree(self.model,
                  filled=True,
                  feature_names=['duration', 'src_bytes', 'dst_bytes'],
                  class_names=list(self.model.classes_),
                  rounded=True,
                  fontsize=10,
                  max_depth=max_depth)

        plt.title("Albero Decisionale - Decision Tree")
        plt.show()

    def export_tree_with_margins(self, output_file="tree_with_margin", graphviz=None):
        """Esporta l'albero decisionale con margini personalizzati usando Graphviz."""
        if self.model is None:
            raise ValueError("Il modello non è stato ancora addestrato.")

        X, y = self.preprocess_data()
        if len(y.unique()) <= 1:
            raise ValueError("Il target ha solo una classe, non è possibile esportare l'albero decisionale.")

        # Esporta il modello in formato DOT
        dot_data = export_graphviz(
            self.model,
            out_file=None,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=['duration', 'src_bytes', 'dst_bytes'],
            class_names=self.model.classes_
        )

        # Personalizza il file DOT con margini personalizzati
        customized_dot_data = f"""
        digraph Tree {{
            graph [ranksep=1.5, nodesep=1.0]  // Aumenta la separazione tra nodi
            node [shape=box, style="filled, rounded", color="black", fontname="helvetica"]
            {dot_data}
        }}
        """

        # Crea il grafico con Graphviz
        graph = graphviz.Source(customized_dot_data)
        graph.format = "png"  # Formato immagine
        graph.render(output_file)  # Salva il file

        print(f"Albero decisionale esportato in {output_file}.png")

# Esempio di utilizzo
if __name__ == "__main__":
    file_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"

    # Inizializza il modello
    tree_model = DecisionTree(file_path)

    print("Ricerca degli iperparametri migliori...")
    best_params = tree_model.optimize_hyperparameters()

    print("Addestramento del modello con i migliori iperparametri...")
    tree_model.train_model()

    print("Visualizzazione dell'albero decisionale...")
    tree_model.plot_decision_tree()

    print("Esportazione dell'albero decisionale con margini personalizzati...")
    tree_model.export_tree_with_margins()
