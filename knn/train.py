from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
# Features: [weight (grams), diameter (cm)]
# Labels: 0 = Apple, 1 = Orange

X = np.array([
    [140, 7.5],  # Apple
    [130, 7.0],  # Apple
    [150, 7.8],  # Apple
    [120, 6.8],  # Apple
    [170, 8.0],  # Orange
    [180, 8.5],  # Orange
    [160, 8.2],  # Orange
    [190, 8.8]   # Orange
])

y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

def classify_fruit(weight, diameter):
    new_fruit = np.array([[weight, diameter]])
    prediction = knn.predict(new_fruit)
    return "Apple" if prediction[0] == 0 else "Orange"

test_fruits = [
    [145, 7.4],
    [175, 8.3] 
]

print("Predictions:")
for fruit in test_fruits:
    result = classify_fruit(fruit[0], fruit[1])
    print(f"Fruit with weight {fruit[0]}g and diameter {fruit[1]}cm is a {result}")

plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Apples')
plt.scatter(X[y==1, 0], X[y==1, 1], c='orange', label='Oranges')
plt.scatter([t[0] for t in test_fruits], [t[1] for t in test_fruits], 
           c='blue', label='Test Fruits')
plt.xlabel('Weight (grams)')
plt.ylabel('Diameter (cm)')
plt.legend()
plt.title('Fruit Classification using k-NN')
plt.show()