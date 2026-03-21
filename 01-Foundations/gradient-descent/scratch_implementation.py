import numpy as np

def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    Finds the line of best fit (y = mx + b) using raw calculus.
    """
    # 1. Initialization: Start at a random coordinate
    m = 0.0
    b = 0.0
    n = len(X)

    for i in range(iterations):
        # 2. Calculate Predictions
        y_pred = m * X + b
        
        # 3. Calculate Gradients (Partial Derivatives of Mean Squared Error)
        dm = (2/n) * np.sum(X * (y_pred - y))
        db = (2/n) * np.sum(y_pred - y)
        
        # 4. Update Parameters (Take a step downhill)
        m = m - (learning_rate * dm)
        b = b - (learning_rate * db)
        
        # Print progress to watch the descent
        if i % 200 == 0:
            print(f"Iteration {i}: m = {m:.4f}, b = {b:.4f}")
            
    return m, b

if __name__ == "__main__":
    # Test Data: The true relationship is roughly y = 2x + 3
    X_data = np.array([1, 2, 3, 4, 5])
    y_data = np.array([5, 7, 9, 11, 13]) 

    print("Starting Descent...\n")
    final_m, final_b = gradient_descent(X_data, y_data, learning_rate=0.01, iterations=1000)

    print(f"\nFinal Engine Output: y = {final_m:.2f}x + {final_b:.2f}")