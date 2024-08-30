import pandas as pd
# Crear un Dataframe con una columna categorica
data = {'Color': ['Rojo', 'Verde', 'Azul', 'Rojo', 'Verde']}
df = pd.DataFrame(data)

# Aplicar codificacion one-hot
one_hot_encoded = pd.get_dummies(df, columns=['Color'])
print(one_hot_encoded)