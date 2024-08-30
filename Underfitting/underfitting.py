import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Generar datos
np.random.seed(0)
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Modelos
degrees = [1, 3, 10]
plt.figure(figsize=(14, 4))

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    X_test = np.linspace(0, 1, 100)[:, np.newaxis]
    plt.scatter(X, y, color='blue', s=10)
    plt.plot(X_test, model.predict(X_test), color='red')
    plt.title(f'Grado {degree}')
    plt.ylim((-2, 2))

plt.tight_layout()
plt.show()