import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

class AdalineSGD:

    def __init__(self, eta = 0.01, n_iter = 10, random_state = 1, shuffle = True):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
        self.losses_ = [];

    def fit(self, X, y):
        self._initialize_weights(X.shape[1]);
        self.losses_ = [];
        for i in range (self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = [];
            for xi, target in zip (X, y):
                losses.append(self._update_weights(xi, target))
            loss = np.mean(losses);
            self.losses_.append(loss)
        return self;

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self;
    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_= float(0.)
        self.w_initialized = True;

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = target - output;
        self.w_ += self.eta * 2.0 * xi * error;
        self.b_ += self.eta * 2.0 * error
        loss = error ** 2
        return loss;
    def net_input (self, X):
        return np.dot (X, self.w_) + self.b_;

    def activation(self, X):
        return X;
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

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
adal = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
adal.fit(X_train, y_train)

# Make predictions
y_pred = adal.predict(X_test)
accuracy = np.mean(y_pred == y_test)

# Visualize the errors per iteration
plt.plot(range(1, len(adal.losses_) + 1), adal.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron Training Convergence')
plt.show()

# Plot decision boundary
def plot_decision_regions(X, y, classifier, resolution=0.02):
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

# Standardize full dataset for plotting
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
X_combined_std = sc.transform(X_combined)

# Plot decision boundary
plot_decision_regions(X_combined_std, y_combined, classifier=adal)
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.title('Perceptron Decision Boundary')
plt.legend()
plt.show()

# Print accuracy
print(f'Perceptron classification accuracy: {accuracy * 100:.2f}%')