import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import os


X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)

y = y.reshape(-1, 1)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15/0.85), random_state=42
)

def plot_data(X, y, title="Dataset"):
    """Helper function to visualize the dataset."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], label="Class 0", alpha=0.7)
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], label="Class 1", alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if not os.environ.get('HIDE_PLOTS', False): plt.show()

print("--- Visualizing Initial Data ---")
plot_data(X, y, "Non-linearly Separable Circular Data")


def plot_decision_boundary(model, X, y, title="Decision Boundary", save_fname=None):
    """Helper function to plot the decision boundary of a trained model."""
    plt.figure(figsize=(8, 6))

    plt.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], label="Class 0", alpha=0.7)
    plt.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], label="Class 1", alpha=0.7)

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))


    grid_X = np.c_[xx1.ravel(), xx2.ravel()]

    if hasattr(model, 'poly_features') and model.poly_features:
        grid_X_poly = model.transform(grid_X)
    else:
        grid_X_poly = grid_X

    Z = model.predict(grid_X_poly)
    Z = Z.reshape(xx1.shape)

 
    plt.contourf(xx1, xx2, Z, alpha=0.3, levels=[0, 0.5, 1], colors=['#3B82F6', '#F472B6'])

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_fname:
      if not os.environ.get('HIDE_PLOTS', False): plt.savefig(save_fname) 

    if not os.environ.get('HIDE_PLOTS', False): plt.show()

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=10000, lambda_val=0.1, regularization=None, poly_features=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_val = lambda_val  
        self.regularization = regularization  
        self.poly_features = poly_features
        self.bias = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def transform(self, X):
        X1 = X[:, 0:1]
        X2 = X[:, 1:2]
        X1_squared = X1 ** 2
        X2_squared = X2 ** 2
        X1_X2 = X1 * X2
        return np.hstack((X1, X2, X1_squared, X2_squared, X1_X2))

    def _compute_cost(self, y, y_pred):
        m = len(y)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        cost = -(1 / m) * np.sum(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )

        reg_cost = 0
        if self.regularization == 'L2':
            reg_cost = (self.lambda_val / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'L1':
            reg_cost = (self.lambda_val / m) * np.sum(np.abs(self.weights))

        return cost + reg_cost

    def fit(self, X, y):
        if self.poly_features:
            X = self.transform(X)

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        self.costs = []

        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            cost = self._compute_cost(y, y_pred)
            self.costs.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            if self.regularization == 'L2':
                dw += (self.lambda_val / n_samples) * self.weights
            elif self.regularization == 'L1':
                dw += (self.lambda_val / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if i % 1000 == 0:
                print(f"Cost after iteration {i}: {cost:.4f}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred > 0.5).astype(int)


linear_model = LogisticRegression(learning_rate=0.1, n_iterations=20000)
linear_model.fit(X_train, y_train)
plot_decision_boundary(linear_model, X, y, "Linear Decision Boundary (Fails)", "submission_4a_linear_decision_boundary.png")

nonlinear_model = LogisticRegression(learning_rate=0.1, n_iterations=20000, poly_features=True)
nonlinear_model.fit(X_train, y_train)
plot_decision_boundary(nonlinear_model, X, y, "Non-linear Decision Boundary (Succeeds)", "submission_4a_nonlinear_decision_boundary.png")


print("Training L2 (Ridge) Regularized Model...")
l2_model = LogisticRegression(learning_rate=0.1, n_iterations=20000,
                              regularization='L2', lambda_val=0.5, poly_features=True)
l2_model.fit(X_train, y_train)
plot_decision_boundary(l2_model, X, y, "Decision Boundary with L2 Regularization")

print("\nTraining L1 (Lasso) Regularized Model...")
l1_model = LogisticRegression(learning_rate=0.1, n_iterations=20000,
                              regularization='L1', lambda_val=0.5, poly_features=True)
l1_model.fit(X_train, y_train)
plot_decision_boundary(l1_model, X, y, "Decision Boundary with L1 Regularization")

print("\n--- Comparing Model Weights ---")
print(f"L2 Model Weights:\n{l2_model.weights.flatten()}")
print(f"L1 Model Weights:\n{l1_model.weights.flatten()}")

feature_names = ['X1', 'X2', 'X1^2', 'X2^2', 'X1*X2']
non_reg_weights = nonlinear_model.weights.flatten()
l2_weights = l2_model.weights.flatten()
l1_weights = l1_model.weights.flatten()

x = np.arange(len(feature_names))  
width = 0.35  

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharey=True)


rects1 = ax1.bar(x, non_reg_weights, width, label='Non-Regularized', color='tomato')
ax1.set_ylabel('Weight Value')
ax1.set_title('Non-Regularized Weights')
ax1.set_xticks(x)
ax1.set_xticklabels(feature_names, rotation=45, ha="right")
ax1.axhline(0, color='grey', linewidth=0.8)
ax1.grid(True, axis='y', linestyle='--', alpha=0.6)


rects2 = ax2.bar(x, l2_weights, width, label='L2 Weights', color='royalblue')
ax2.set_title('L2 (Ridge) Regularization Weights')
ax2.set_xticks(x)
ax2.set_xticklabels(feature_names, rotation=45, ha="right")
ax2.axhline(0, color='grey', linewidth=0.8)
ax2.grid(True, axis='y', linestyle='--', alpha=0.6)


rects3 = ax3.bar(x, l1_weights, width, label='L1 Weights', color='seagreen')
ax3.set_title('L1 (Lasso) Regularization Weights')
ax3.set_xticks(x)
ax3.set_xticklabels(feature_names, rotation=45, ha="right")
ax3.axhline(0, color='grey', linewidth=0.8)
ax3.grid(True, axis='y', linestyle='--', alpha=0.6)

fig.suptitle('Side-by-Side Comparison of Model Weights', fontsize=16)
if not os.environ.get('HIDE_PLOTS', False): plt.savefig("submission_4b.png") 

if not os.environ.get('HIDE_PLOTS', False): plt.show()


learning_rates = [0.01, 0.05, 0.1]
lambda_values = [0.01, 0.1, 0.5]
best_accuracy = 0
best_params = {}
best_model = None


grid_search_results = np.zeros((len(learning_rates), len(lambda_values)))
X_val_poly = nonlinear_model.transform(X_val)

for i, lr in enumerate(learning_rates):
    for j, lmbda in enumerate(lambda_values):
        print(f"Training with lr={lr}, lambda={lmbda}...")
        model = LogisticRegression(learning_rate=lr, n_iterations=5000,
                                   regularization='L2', lambda_val=lmbda, poly_features=True)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val_poly)
        accuracy = np.mean(predictions == y_val)
        grid_search_results[i, j] = accuracy
        print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'learning_rate': lr, 'lambda_val': lmbda}
            best_model = model

print("\n--- Grid Search Results ---")
print(f"Best parameters found: {best_params}")
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")


print("\nEvaluating best model on the test set...")
X_test_poly = best_model.transform(X_test)
test_predictions = best_model.predict(X_test_poly)
test_accuracy = np.mean(test_predictions == y_test)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")
plot_decision_boundary(best_model, X, y, f"Best Model (lr={best_params['learning_rate']}, lambda={best_params['lambda_val']})", 'submission_4c.png')


X_train_poly = nonlinear_model.transform(X_train)

def run_gd_variant(optimizer='batch', batch_size=32, n_epochs=100):
    n_samples, n_features = X_train_poly.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    costs = []

    if optimizer == 'batch':
        step_size = n_samples
    elif optimizer == 'stochastic':
        step_size = 1
    elif optimizer == 'mini-batch':
        step_size = batch_size

    iterations_per_epoch = int(np.ceil(n_samples / step_size))

    for epoch in range(n_epochs):
        epoch_cost = 0

        permutation = np.random.permutation(n_samples)
        X_shuffled = X_train_poly[permutation]
        y_shuffled = y_train[permutation]

        for i in range(0, n_samples, step_size):
            
            if optimizer == 'stochastic':
                 X_batch = X_shuffled[i:i+1]
                 y_batch = y_shuffled[i:i+1]
            elif optimizer == 'mini-batch':
                 X_batch = X_shuffled[i:i+batch_size]
                 y_batch = y_shuffled[i:i+batch_size]
            else: 
                 X_batch = X_shuffled
                 y_batch = y_shuffled

            m_batch = len(y_batch)
            if m_batch == 0: continue

            linear_model = np.dot(X_batch, weights) + bias
            y_pred = 1 / (1 + np.exp(-linear_model))

            cost = -1/m_batch * np.sum(y_batch * np.log(y_pred) + (1 - y_batch) * np.log(1 - y_pred))
            epoch_cost += cost

            dw = (1 / m_batch) * np.dot(X_batch.T, (y_pred - y_batch))
            db = (1 / m_batch) * np.sum(y_pred - y_batch)

            weights -= 0.1 * dw
            bias -= 0.1 * db

        costs.append(epoch_cost / iterations_per_epoch)

    return costs

cost_batch = run_gd_variant(optimizer='batch', n_epochs=500)
cost_stochastic = run_gd_variant(optimizer='stochastic', n_epochs=500)
cost_mini_batch = run_gd_variant(optimizer='mini-batch', batch_size=32, n_epochs=500)


plt.figure(figsize=(12, 8))
plt.plot(cost_batch, label='Batch GD', linewidth=2)
plt.plot(cost_stochastic, label='Stochastic GD (SGD)', alpha=0.8)
plt.plot(cost_mini_batch, label='Mini-batch GD', linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Function Convergence for Different GD Variants")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
if not os.environ.get('HIDE_PLOTS', False): plt.savefig("submission_4d.png") 
if not os.environ.get('HIDE_PLOTS', False): plt.show()
