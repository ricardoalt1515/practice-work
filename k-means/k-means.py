import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

clientes = pd.DataFrame({"saldo" : [5000,45000, 4800, 43500, 47000, 52000,
                                    20000, 26000, 25000, 23000, 214000, 18000,
                                    8000, 12000, 6000, 14500, 12600, 7000],
                                    
                                    "transacciones": [25, 20, 16, 23, 25, 18,
                                                      23, 11, 24, 21, 27, 18,
                                                      8, 3, 6, 4, 9, 3]})

escalador = MinMaxScaler().fit(clientes.values)

clientes = pd.DataFrame(escalador.transform(clientes.values),
                        columns=["saldo", "transacciones"])

clientes