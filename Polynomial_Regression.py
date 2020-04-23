from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import random

# Training data
X = [i for i in range(10)]
Y = [random.gauss(x, 0.75) for x in X]
print(Y)
X = np.asarray(X)
Y = np.asarray(Y)
X = X[:, np.newaxis]
Y = Y[:, np.newaxis]

plt.scatter(X, Y)


# Data preparation
nb_degree = 4
polynomial_features = PolynomialFeatures(degree=nb_degree)
X_TRANSF = polynomial_features.fit_transform(X)

# Define and train a model
model = LinearRegression()
model.fit(X_TRANSF, Y)

# Calculate bias and variance
Y_NEW = model.predict(X_TRANSF)
rmse = np.sqrt(mean_squared_error(Y, Y_NEW))
r2 = r2_score(Y, Y_NEW)
print('RMSE:', rmse)
print('R2:', r2)

# Prediction
x_new_min = 0.0
x_new_max = 10.0

X_NEW = np.linspace(x_new_min, x_new_max, 100)
X_NEW = X_NEW[:, np.newaxis]

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)

Y_NEW = model.predict(X_NEW_TRANSF)

plt.plot(X_NEW, Y_NEW, color='coral', linewidth=3)
plt.grid()
plt.xlim(x_new_min, x_new_max)
plt.ylim(0, 10)
title = 'Degree = {}; RMSE = {}'.format(nb_degree, round(rmse, 2), round(r2, 2))
plt.title("Polynomial Linear Regression using scikit-learn\n" + title, fontsize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("polynomial_linear_regression.png", bbox_inches='tight')
plt.show()

