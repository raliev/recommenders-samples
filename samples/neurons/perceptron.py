import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

class Perceptron:

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        #self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = X.shape[1])
        self.w_ = np.zeros(X.shape[1])
        self.b_ = float (0.)
        self.errors_ = [];

        for _ in range (self.n_iter):
            errors = 0;
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self;

    def net_input (self, X):
        return np.dot (X, self.w_) + self.b_;

    def predict (self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)


# Plot decision boundary
def plot_decision_regions(X, y, classifier, resolution=0.01):
    markers = ('o', 's', '^', 'v', "<")
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = mcolors.ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max());
    plt.ylim(xx2.min(), xx2.max());

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0],
                    X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black');

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [2, 3]] # petal length и petal width
y = (iris.target != 0).astype(int)  # Convert to binary classification (setosa vs non-setosa)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for better convergence
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the Perceptron
ppn = Perceptron(eta=0.01, n_iter=15, random_state=1)
ppn.fit(X_train, y_train)

# Make predictions
y_pred = ppn.predict(X_test)
accuracy = np.mean(y_pred == y_test)

# Visualize the errors per iteration
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron Training Convergence')
plt.show()

# Standardize full dataset for plotting
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
X_combined_std = sc.transform(X_combined)

# Plot decision boundary
plot_decision_regions(X_combined_std, y_combined, classifier=ppn)
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.show()

# Print accuracy
print(f'Perceptron classification accuracy: {accuracy * 100:.2f}%')