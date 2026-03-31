from utils import db_connect
engine = db_connect()

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_and_clean_data(url):
    """Carga los datos y realiza la limpieza de ceros."""
    df = pd.read_csv(url)
    cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_con_ceros] = df[cols_con_ceros].replace(0, np.nan)
    for col in cols_con_ceros:
        df[col] = df[col].fillna(df[col].median())
    return df

def train_optimized_model(X_train, y_train):
    """Entrena el modelo con los mejores parámetros encontrados en el GridSearchCV."""
    model = DecisionTreeClassifier(
        criterion='gini', 
        max_depth=5, 
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Calcula e imprime las métricas de rendimiento."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*30)
    print("REPORTE DE EVALUACIÓN")
    print("="*30)
    print(f"Precisión Total (Accuracy): {accuracy:.2%}")
    print("\nDetalle por Clase:")
    print(classification_report(y_test, predictions, target_names=['No Diabetes', 'Diabetes']))
    print("="*30 + "\n")

def save_model(model, filename):
    """Guarda el modelo en un archivo .sav"""
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    URL = "https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv"
    
    # 1. Preparación
    print("Iniciando proceso...")
    df = load_and_clean_data(URL)
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Entrenamiento
    print("Entrenando modelo optimizado...")
    final_model = train_optimized_model(X_train, y_train)
    
    # 3. Evaluación (Nueva función)
    evaluate_model(final_model, X_test, y_test)
    
    # 4. Guardado
    print("Guardando modelo final...")
    save_model(final_model, "data/diabetes_decision_tree_optimized.sav")
    
    print("¡Proceso completado con éxito!")
