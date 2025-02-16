import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, 2].reshape(-1, 1)  # petal length
y = iris.data[:, 3]  # petal width

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = linear_model.LinearRegression()
ridge_reg = linear_model.Ridge(alpha=5.0)

lin_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)

y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)

plt.scatter(X_test, y_test, color='black', label='Actual data')
plt.plot(X_test, y_pred_lin, color='red', linewidth=2, label='Linear Regression')
plt.plot(X_test, y_pred_ridge, color='blue', linewidth=2, linestyle='dashed', label='Ridge Regression (alpha=5.0)')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Comparison of Linear and Ridge Regression')
plt.legend()
plt.show()

print("Linear Regression Coefficients:", lin_reg.coef_)
print("Ridge Regression Coefficients:", ridge_reg.coef_)