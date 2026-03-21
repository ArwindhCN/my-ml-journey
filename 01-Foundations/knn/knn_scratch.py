import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    """
    The core math: Pythagorean theorem for N-dimensions.
    Calculates the straight-line physical distance between two points.
    """
    return np.sqrt(np.sum((point1 - point2)**2))

def knn_predict(X_train, y_train, new_point, k=3):
    """
    Predicts the class of a new point based on majority vote of nearest neighbors.
    """
    distances = []
    
    # 1. Calculate distance from the new_point to EVERY point in the dataset
    for i in range(len(X_train)):
        dist = euclidean_distance(new_point, X_train[i])
        # Store a tuple of (distance, label)
        distances.append((dist, y_train[i]))
        
    # 2. Sort the list by distance (shortest distance first)
    distances.sort(key=lambda x: x[0])
    
    # 3. Slice the list to grab only the 'K' closest neighbors
    k_nearest_labels = [label for distance, label in distances[:k]]
    
    # 4. Take a majority vote
    vote_counts = Counter(k_nearest_labels)
    winner = vote_counts.most_common(1)[0][0]
    
    return winner

# --- Testing the Engine ---
if __name__ == "__main__":
    # Dummy Data: [Feature 1, Feature 2]
    # Class 0 (Safe) is clustered around lower numbers (e.g., [1,1], [2,2])
    # Class 1 (Phishing) is clustered around higher numbers (e.g., [8,8], [9,9])
    X_data = np.array([
        [1, 2], [2, 3], [3, 1], [2, 2], # Safe Cluster
        [8, 9], [9, 8], [9, 9], [8, 8]  # Phishing Cluster
    ])
    y_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    # Drop a new unknown point right in the middle of the Phishing cluster
    unknown_point = np.array([8.5, 8.5])
    
    print(f"Dropping new point at coordinates: {unknown_point}")
    
    # Run the prediction
    prediction = knn_predict(X_data, y_labels, unknown_point, k=3)
    
    class_name = "Phishing" if prediction == 1 else "Safe"
    print(f"\nThe K=3 Nearest Neighbors voted: This is {class_name} (Class {prediction})")