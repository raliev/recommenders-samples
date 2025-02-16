import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, 2].reshape(-1, 1) # petal length
y = iris.data[:, 3]  # petal width

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train - features from train (petal lengths)
# X_test - features from test (petal lengths)
# Y_train - values from train (petal widths)
# Y_test - values from test (petal widths)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred, color='blue', linewidth=2, label='Regression line')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Linear Regression: Petal Length vs Petal Width')
plt.legend()
plt.show()

print("Coefficients:", reg.coef_)
