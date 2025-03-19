# k_nearest_neighbors

def k_nearest_neighbors(X_train, y_train, X_test, k):
    def euclidean_distance(point1, point2):
        distance = 0
        for i in range(len(point1)):
            distance += (point1[i] - point2[i]) ** 2
        return distance ** 0.5

    def get_neighbors(train_data, train_labels, test_point, k):
        distances = []
        for i in range(len(train_data)):
            dist = euclidean_distance(test_point, train_data[i])
            distances.append((dist, train_labels[i]))
        distances.sort(key=lambda x: x[0])  # Sort by distance
        neighbors = [label for dist, label in distances[:k]]
        return neighbors

    def predict(neighbors):
        counts = {}
        for neighbor in neighbors:
            counts[neighbor] = counts.get(neighbor, 0) + 1
        return max(counts, key=counts.get)  # Return label with highest count

    predictions = []
    for test_point in X_test:
        neighbors = get_neighbors(X_train, y_train, test_point, k)
        predictions.append(predict(neighbors))

    return predictions

# Example Usage:
X_train = [[1, 2], [2, 3], [3, 1], [4, 5], [5, 4]]
y_train = [0, 0, 1, 1, 1]
X_test = [[2, 2], [4, 3]]
k = 3

predictions = k_nearest_neighbors(X_train, y_train, X_test, k)
print(predictions)  # Output: [0, 1]