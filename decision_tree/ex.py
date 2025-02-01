import numpy as np

# Example dataset: [color, size]
# 0 = green, 1 = red
# 0 = small, 1 = large

X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])  # Features
y = np.array(["Apple", "Apple", "Orange", "Orange"])  # Labels

# A simple decision tree function
def decision_tree(fruit):
    color, size = fruit
    if color == 1:  # If the fruit is red
        return "Apple"
    else:  # If the fruit is green
        return "Orange"

# Test the decision tree
for i in range(len(X)):
    print(f"Features: {X[i]}, Prediction: {decision_tree(X[i])}")