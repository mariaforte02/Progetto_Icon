import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Dizionario degli iperparametri per RandomForest
RandomForestHyperparameters = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
}

def load_and_preprocess_data(file_path):
    # Carica il dataset
    df = pd.read_csv(file_path)

    # Verifica se la colonna 'flag' esiste nel dataset
    if 'flag' not in df.columns:
        raise ValueError("La colonna 'flag' non è presente nel dataset. Verifica il file CSV.")

    # Usa 'flag' come target per la predizione
    target_column = 'flag'

    # Definisci le caratteristiche (features) e il target
    X = df.drop(columns=[target_column])  # Tutte le colonne tranne il target
    y = df[target_column]  # La colonna target

    # Preprocessing per le variabili categoriche
    label_encoder = LabelEncoder()
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = label_encoder.fit_transform(X[col])

    # Dividi i dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, df

def optimize_random_forest_hyperparameters(X_train, y_train):
    # Esegui la ricerca degli iperparametri per Random Forest
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=RandomForestHyperparameters,
                               cv=2,  # Cambia da 3 a 2
                               scoring='accuracy',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Crea un DataFrame con i risultati della GridSearchCV
    results = pd.DataFrame(grid_search.cv_results_)
    print("Tabella degli iperparametri ottimali per Random Forest:")
    print(results[['params', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score']])

    return grid_search.best_params_

def train_random_forest(X_train, X_test, y_train, y_test):
    print("Ricerca degli iperparametri migliori per Random Forest...")
    best_params = optimize_random_forest_hyperparameters(X_train, y_train)

    # Addestra il modello Random Forest con i migliori iperparametri
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)

    # Predizione
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello Random Forest: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Calcolo del rischio
    if hasattr(model, "predict_proba"):
        risk_scores = model.predict_proba(X_test)[:, 0]
        X_test_with_risk = X_test.copy()
        X_test_with_risk['Rischio'] = risk_scores

        # Distribuzione del rischio
        plt.figure(figsize=(10, 6))
        sns.histplot(risk_scores, bins=20, kde=True, color='blue')
        plt.title('Distribuzione del rischio predetto')
        plt.xlabel('Rischio')
        plt.ylabel('Frequenza')
        plt.show()

        # Calcolo e visualizzazione del rischio medio
        if 'src_bytes' in X_test_with_risk.columns:
            avg_risk_per_feature = X_test_with_risk.groupby('src_bytes')['Rischio'].mean().reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(data=avg_risk_per_feature, x='src_bytes', y='Rischio', palette='viridis')
            plt.title('Rischio medio in funzione dei byte inviati')
            plt.xlabel('Byte inviati (src_bytes)')
            plt.ylabel('Rischio medio')
            plt.xticks(rotation=45)
            plt.show()

    return model

def train_svm(X_train, X_test, y_train, y_test):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello SVM: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model

def train_knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello KNN: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model

def train_decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuratezza del modello Decision Tree: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    return model

# Esempio di utilizzo
if __name__ == "__main__":
    file_path = r"C:\Users\meryf\OneDrive\desktop\PROGETTO_ICON\Processed_Test_data_cleaned_unified.csv"
    X_train, X_test, y_train, y_test, df = load_and_preprocess_data(file_path)

    print("\n--- Training Random Forest ---")
    train_random_forest(X_train, X_test, y_train, y_test)

    print("\n--- Training SVM ---")
    train_svm(X_train, X_test, y_train, y_test)

    print("\n--- Training KNN ---")
    train_knn(X_train, X_test, y_train, y_test)

    print("\n--- Training Decision Tree ---")
    train_decision_tree(X_train, X_test, y_train, y_test)