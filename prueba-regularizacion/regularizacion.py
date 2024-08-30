import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generar datos sintéticos
np.random.seed(0)
X = np.random.randn(100, 10)
y = X.dot(np.random.randn(10)) + np.random.randn(100)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Aplicar Ridge Regression (L2)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train, y_train)
y_pred_ridge = ridge_reg.predict(X_test)
print(f'Ridge Regression MSE: {mean_squared_error(y_test, y_pred_ridge)}')

# Aplicar Lasso Regression (L1)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
y_pred_lasso = lasso_reg.predict(X_test)
print(f'Lasso Regression MSE: {mean_squared_error(y_test, y_pred_lasso)}')
