import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Crear el conjunto de datos
data = {
    'Clima': ['Soleado', 'Soleado', 'Nublado', 'Lluvioso', 'Lluvioso', 'Lluvioso', 'Nublado', 'Soleado', 'Soleado', 'Lluvioso', 'Soleado', 'Nublado', 'Nublado', 'Lluvioso'],
    'Humedad': ['Alta', 'Alta', 'Alta', 'Baja', 'Baja', 'Alta', 'Baja', 'Baja', 'Baja', 'Baja', 'Alta', 'Alta', 'Baja', 'Alta'],
    'Viento': ['Débil', 'Fuerte', 'Débil', 'Débil', 'Fuerte', 'Débil', 'Débil', 'Débil', 'Fuerte', 'Débil', 'Fuerte', 'Fuerte', 'Fuerte', 'Débil'],
    'Jugar': ['No', 'No', 'Sí', 'Sí', 'No', 'Sí', 'Sí', 'No', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'No']
}

df = pd.DataFrame(data)

# Convertir atributos categóricos a numéricos
df['Clima'] = df['Clima'].astype('category').cat.codes
df['Humedad'] = df['Humedad'].astype('category').cat.codes
df['Viento'] = df['Viento'].astype('category').cat.codes
df['Jugar'] = df['Jugar'].astype('category').cat.codes

# Separar características y etiquetas
X = df[['Clima', 'Humedad', 'Viento']]
y = df['Jugar']

# Entrenar el árbol de decisión
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))  # Ajustar el tamaño de la figura
tree.plot_tree(clf, feature_names=['Clima', 'Humedad', 'Viento'], class_names=['No', 'Sí'], filled=True)
plt.show()
