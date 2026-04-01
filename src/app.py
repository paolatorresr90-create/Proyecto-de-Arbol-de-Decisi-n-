import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# 1. Carga de datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv"
df = pd.read_csv(url)

# 2. Limpieza básica (Imputación de medianas para valores 0 que no tienen sentido)
cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_con_ceros] = df[cols_con_ceros].replace(0, np.nan)
for col in cols_con_ceros:
    df[col] = df[col].fillna(df[col].median())

# 3. División de datos
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenamiento de los 3 modelos
print("Entrenando modelos...")

# Modelo 1: Árbol de Decisión (Básico)
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

# Modelo 2: Random Forest (El que mejor te funcionó)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)

# Modelo 3: Gradient Boosting (El "hermano" de XGBoost)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)

# 5. Comparativa de resultados
print("\n--- Resultados Finales ---")
print(f"Decision Tree Accuracy:   {accuracy_score(y_test, tree_model.predict(X_test)):.2%}")
print(f"Random Forest Accuracy:   {accuracy_score(y_test, rf_model.predict(X_test)):.2%}")
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_model.predict(X_test)):.2%}")

# 6. Guardar el mejor modelo (Random Forest)
if not os.path.exists('src/models'):
    os.makedirs('src/models')

with open('src/models/random_forest_final.sav', 'wb') as f:
    pickle.dump(rf_model, f)

print("\n✅ Proyecto completado. El mejor modelo ha sido guardado en src/models/")