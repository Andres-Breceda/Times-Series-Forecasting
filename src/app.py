import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv")

data.info()

data.describe()

data.head()

data = data[['Latitude', 'Longitude', 'MedInc']]
data.head()

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Escalar los datos
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Crear un DataFrame con los datos escalados y conservar los nombres de las columnas originales
data_s = pd.DataFrame(data_scaled, columns=data.columns)

# (Opcional) Ver estadísticos descriptivos
data_s.describe()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Dividir datos y variables de entrenamiento
x_train, x_test = train_test_split(data_s, test_size=0.2, random_state=42)


# Crear y entrenar modelo
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(x_train)  # <- obtenemos las etiquetas

# Predecir los clusters de test con el modelo entrenado
predict = kmeans.predict(x_test)

# Crear copias para añadir clusters
x_train_clustered = x_train.copy()
x_train_clustered["clustered"] = clusters  # etiquetas de entrenamiento

x_test_clustered = x_test.copy()
x_test_clustered["clustered"] = predict  # etiquetas del test

import seaborn as sns

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=x_test.iloc[:, 0],
    y=x_test.iloc[:, 1],
    hue=x_test_clustered["clustered"],
    palette='tab10'
)
plt.title("Clusters en conjunto de prueba (KMeans)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.figure(figsize=(10,7))

# Graficar puntos de entrenamiento (train) con su cluster
sns.scatterplot(
    x=x_train.iloc[:, 0], y=x_train.iloc[:, 1],
    hue=x_train_clustered["clustered"],
    palette='Set2', alpha=0.6, 
)

# Graficar puntos de test con cluster predicho, con otro marcador
sns.scatterplot(
    x=x_test.iloc[:, 0], y=x_test.iloc[:, 1],
    hue=x_test_clustered["clustered"],
    palette='Set2', marker='X', s=100, legend=None
)

plt.title("Clusters KMeans: Train y Test")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Leyenda personalizada para distinguir train y test
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='gray', markersize=10, alpha=0.6),
    Line2D([0], [0], marker='X', color='w', label='Test', markerfacecolor='gray', markersize=10)
]
plt.legend(handles=legend_elements, title='Conjunto')

plt.show()

x_train_clustered.head()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Datos para entrenamiento
X_train_rf = x_train_clustered.drop(columns=["clustered"])
y_train_rf = x_train_clustered["clustered"]


# Datos para test
X_test_rf = x_test_clustered.drop(columns=["clustered"])
y_test_rf = x_test_clustered["clustered"]

# Crear y entrenar modelo
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train_rf, y_train_rf)

# Predecir
y_pred_rf = rf_model.predict(X_test_rf)

# Evaluar
print("Accuracy:", accuracy_score(y_test_rf, y_pred_rf))
print(classification_report(y_test_rf, y_pred_rf))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test_rf, y_pred_rf)
print(cm)

import joblib

# Guardar modelo KMeans
joblib.dump(kmeans, "/workspaces/machine-learning-python-linear-regression/models/kmeans_model.pkl")

# Guardar modelo Random Forest
joblib.dump(rf_model, "/workspaces/machine-learning-python-linear-regression/models/random_forest_model.pkl")