import numpy as np
import unittest
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class Node:
    """A node in the decision tree."""

    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    """A simplified Decision Tree Classifier with detailed TODOs."""

    def __init__(self, max_depth=1, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        is_max_depth_reached = depth >= self.max_depth
        is_one_class = n_labels == 1
        is_min_samples_not_met = n_samples < self.min_samples_split

        if is_max_depth_reached or is_one_class or is_min_samples_not_met:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is not None:
            left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
            left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
            right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
            return Node(best_feature, best_threshold, left_child, right_child)
        leaf_value = self._most_common_label(y)

        return Node(value=leaf_value)

    def _find_best_split(self, X, y):
        best_gain = 0.0
        best_feature, best_threshold = None, None
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            for thr in thresholds:
                gain = self._information_gain(y, X[:, feat_idx], thr)

                is_better_gain = gain > best_gain
                if is_better_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = thr

        return best_feature, best_threshold

    def _information_gain(self, y, X_column, threshold):
        parent_impurity = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        gini_left = self._gini(y[left_idxs])
        gini_right = self._gini(y[right_idxs])
        child_impurity = (n_l / n) * gini_left + (n_r / n) * gini_right
        information_gain = parent_impurity - child_impurity

        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs
    def _gini(self, y):
    
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_ or 0)

        probabilities = counts/len(y)
        gini = 1.0 - np.sum(probabilities ** 2)

        return gini

    def _most_common_label(self, y):

        most_common = int(np.argmax(np.bincount(y)))
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
        if is_leaf_node:
            return node.value
        go_left = x[node.feature] <= node.threshold

        if go_left:
            predicted_label = self._traverse_tree(x, node.left)
        else:
            predicted_label = self._traverse_tree(x, node.right)


        return predicted_label


if __name__ == "__main__":
    print("--- 1. Simple 1D Smoke Test ---")
    X_simple = np.array([[1], [2], [3], [4]])
    y_simple = np.array([0, 0, 1, 1])
    print(f"Simple X:\n{X_simple}")
    print(f"Simple y:\n{y_simple}")


    clf_simple = DecisionTreeClassifier(max_depth=1)
    clf_simple.fit(X_simple, y_simple)

    if clf_simple.root and not clf_simple.root.is_leaf_node():
        print("\nFitted Tree (Root Node):")
        print(f"  Split Feature: {clf_simple.root.feature}")
        print(f"  Split Threshold: {clf_simple.root.threshold}")
        print(f"  Left Child Value: {clf_simple.root.left.value}")
        print(f"  Right Child Value: {clf_simple.root.right.value}")
    else:
        print("\nTree has no splits (root is a leaf). Check implementation.")


    X_test_simple = np.array([[1.5], [2.0], [2.5], [5.0]])
    y_pred_simple = clf_simple.predict(X_test_simple)
    print(f"\nTest Data:\n{X_test_simple}")
    print(f"Predictions: {y_pred_simple}")
    print(f"Expected: [0 0 1 1]") 

    print("\n\n--- 2. Sklearn make_classification Test ---")
    X_complex, y_complex = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, n_classes=2, random_state=0)
    clf_complex = DecisionTreeClassifier(max_depth=1)
    clf_complex.fit(X_complex, y_complex)
    y_pred_complex = clf_complex.predict(X_complex)
    accuracy = accuracy_score(y_complex, y_pred_complex)
    print(f"Complex Test - Weak Learner | Training Accuracy: {accuracy:.4f}")
    print("A correct implementation should achieve moderate accuracy here.")
    clf_complex = DecisionTreeClassifier(max_depth=5)
    clf_complex.fit(X_complex, y_complex)
    y_pred_complex = clf_complex.predict(X_complex)
    accuracy = accuracy_score(y_complex, y_pred_complex)
    print(f"Complex Test - Better Learner | Training Accuracy: {accuracy:.4f}")
    print("A correct implementation should achieve better accuracy here.")