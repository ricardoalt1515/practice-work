import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Crear un DataFrame con características categóricas y la etiqueta
data = {
    'Color': ['Rojo', 'Verde', 'Azul', 'Rojo', 'Verde'],
    'Tamaño': ['Grande', 'Pequeño', 'Mediano', 'Pequeño', 'Grande'],
    'Jugar': ['Sí', 'No', 'Sí', 'Sí', 'No']
}

df = pd.DataFrame(data)

# Separar características y etiquetas
X = df[['Color', 'Tamaño']]
y = df['Jugar']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el transformador para la codificación one-hot
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Color', 'Tamaño'])
    ])

# Crear el pipeline con el preprocesador y el modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', DecisionTreeClassifier())])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Hacer predicciones
y_pred = pipeline.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')